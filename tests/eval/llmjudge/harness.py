# -*- coding: utf-8 -*-
"""SUT 运行器与证据包构建。

每个 trial：
  1. 建立独立临时工作区（office/skills/tasks/日志采集互不串扰）；
  2. 按用例 host 配置组装 FluxAgent 图（默认工具集 / 技能 / 委派 / 审批开关）；
  3. 用替身模型（dry-run）或真实模型跑完整图；
  4. 采集证据：审计事件、工具结果、最终回答、office 前后哈希、耗时；
  5. 执行宿主程序断言（independent_checks）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import threading
from typing import Any
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from fluxagent.core.tools import builtins as builtins_mod
from fluxagent.core.tools.builtins import BUILTIN_TOOLS
from fluxagent.core import skill_loader as skill_loader_mod
from fluxagent.core import subagents as subagents_mod
from fluxagent.core.tools import sandbox_tools
from fluxagent.core.tools.builtins import tasks_lock
from fluxagent.capability.context_compiler import MemoryScope, MemoryStore, ProfileCompatProvider
from fluxagent.core.approval import approve, pending_approval
from fluxagent.core.agent import _grounded_memory_evidence


class StubModel:
    """按脚本顺序执行的替身模型：工具调用 / 最终回答 / 宿主动作。"""

    def __init__(self, script: list[dict], harness_cb=None,
                 turn_scripts: list[list[dict]] | None = None):
        self.script = list(script)
        self.turn_scripts = [list(items) for items in (turn_scripts or [])]
        self.index = 0
        self.turn_indexes: dict[int, int] = {}
        self.harness_cb = harness_cb or (lambda action: None)
        self.calls: list[tuple[list, dict | None]] = []
        self.context: dict[str, str] = {}
        self.child_fails = False  # 宿主故障注入：子 Agent 模型不可用

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def _resolve(self, args: dict) -> dict:
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value == "$token":
                resolved[key] = self.context.get("token", "")
            else:
                resolved[key] = value
        return resolved

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append((messages, config))
        if self.child_fails and (config or {}).get("configurable", {}).get("subagent_depth"):
            raise RuntimeError("provider down (harness)")
        # 从最近的 ToolMessage 捕获 help_token 供后续步骤引用
        for msg in reversed(messages):
            if getattr(msg, "type", "") == "tool":
                match = re.search(r"help_token: ([A-Za-z0-9_-]+)", str(msg.content))
                if match:
                    self.context["token"] = match.group(1)
                break
        configurable = (config or {}).get("configurable", {})
        configured_turn = configurable.get("stub_turn_index")
        if configured_turn is None:
            human_count = sum(1 for message in messages if getattr(message, "type", "") == "human")
            turn_index = max(0, human_count - 1)
        else:
            turn_index = max(0, int(configured_turn))
        if self.turn_scripts:
            script = self.turn_scripts[min(turn_index, len(self.turn_scripts) - 1)]
            index = self.turn_indexes.get(turn_index, 0)
        else:
            script = self.script
            index = self.index
        while index < len(script):
            step = script[index]
            index += 1
            if self.turn_scripts:
                self.turn_indexes[turn_index] = index
            else:
                self.index = index
            if "harness" in step:
                self.harness_cb(step)
                continue
            if "tool" in step:
                return AIMessage(content="", tool_calls=[{
                    "name": step["tool"],
                    "args": self._resolve(step.get("args", {})),
                    "id": f"stub-call-{self.index}",
                    "type": "tool_call",
                }])
            if "say" in step:
                return AIMessage(content=step["say"])
        return AIMessage(content="done")

    async def ainvoke(self, messages, config=None, **kwargs):
        return self.invoke(messages, config, **kwargs)


class _EventRecorder:
    """替身 audit_logger：同步记录事件（替代异步写盘，保证证据完整）。"""

    def __init__(self):
        self.events: list[dict] = []

    def log_event(self, thread_id, event, **kwargs):
        self.events.append({"thread_id": thread_id, "event": event, **kwargs})
        return True

    def shutdown(self):
        return None


def _hash_office(office_dir: str) -> dict[str, str]:
    snapshot = {}
    for root, _dirs, files in os.walk(office_dir):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, office_dir).replace(os.sep, "/")
            try:
                with open(path, "rb") as fh:
                    snapshot[rel] = hashlib.sha256(fh.read()).hexdigest()[:12]
            except OSError:
                snapshot[rel] = "unreadable"
    return snapshot


def _materialize_fixtures(office_dir: str, skills_dir: str, fixtures: dict) -> None:
    for rel, content in (fixtures.get("office_files") or {}).items():
        path = os.path.join(office_dir, rel)
        os.makedirs(os.path.dirname(path) or office_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    for skill in fixtures.get("skills") or []:
        folder = os.path.join(skills_dir, skill["folder"])
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "SKILL.md"), "w", encoding="utf-8") as fh:
            fh.write(skill["skill_md"])
        for script_name, script_body in (skill.get("scripts") or {}).items():
            script_path = os.path.join(folder, script_name)
            os.makedirs(os.path.dirname(script_path) or folder, exist_ok=True)
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(script_body)


class TrialRunner:
    """跑单个 trial 并产出证据包 + 程序断言结果。"""

    def __init__(self, provider_name: str | None = None, model_name: str | None = None,
                 key_pool=None, base_url: str | None = None, sut_model: str | None = None,
                 run_id: str = ""):
        self.provider_name = provider_name
        self.model_name = model_name
        self.key_pool = key_pool
        self.base_url = base_url
        self.sut_model = sut_model
        self.run_id = str(run_id or "").strip()

    def run(self, case: dict, trial_index: int = 0, stub: bool = True) -> dict:
        started = time.time()
        prefix = f"{self.run_id}-" if self.run_id else ""
        trial_id = f"{prefix}{case['case_id']}-t{trial_index}-{'stub' if stub else 'real'}"
        with tempfile.TemporaryDirectory(prefix="fluxjudge-") as tmp:
            office_dir = os.path.join(tmp, "office")
            skills_dir = os.path.join(office_dir, "skills")
            memory_dir = os.path.join(tmp, "memory")
            os.makedirs(office_dir, exist_ok=True)
            os.makedirs(skills_dir, exist_ok=True)
            os.makedirs(memory_dir, exist_ok=True)
            _materialize_fixtures(office_dir, skills_dir, case["fixtures"])
            profile_content = case["fixtures"].get("memory_profile")
            if profile_content is not None:
                with open(os.path.join(memory_dir, "user_profile.md"), "w", encoding="utf-8") as fh:
                    fh.write(str(profile_content))

            recorder = _EventRecorder()
            office_before = _hash_office(office_dir)
            profile_before = ""
            profile_path = os.path.join(memory_dir, "user_profile.md")
            if os.path.exists(profile_path):
                with open(profile_path, encoding="utf-8") as fh:
                    profile_before = fh.read()

            host = case["host"]
            patches = [
                patch.object(sandbox_tools, "OFFICE_DIR", office_dir),
                patch.object(skill_loader_mod, "SKILLS_DIR", skills_dir),
                patch.object(subagents_mod, "audit_logger", recorder),
                patch("fluxagent.core.agent.audit_logger", recorder),
                # 每个 trial 独立画像目录：真实运行不得读写 workspace/memory
                patch("fluxagent.core.agent.MEMORY_DIR", memory_dir),
            ]
            tasks_file = None
            if host.get("tasks_file"):
                tasks_file = os.path.join(tmp, "tasks.json")
                patches.append(patch.object(
                    builtins_mod,
                    "TASKS_FILE", tasks_file))

            if stub:
                def harness_cb(step: dict) -> None:
                    action = step.get("harness")
                    if action == "mutate_skill":
                        folder = os.path.join(skills_dir, step.get("skill", "demo"))
                        md = os.path.join(folder, "SKILL.md")
                        with open(md, "r", encoding="utf-8") as fh:
                            text = fh.read()
                        with open(md, "w", encoding="utf-8") as fh:
                            fh.write(text.replace("v1", "v2") + "\nMUTATED\n")
                    elif action == "fail_child_provider":
                        # 子 Agent 模型在子图内（depth>=1）直接抛错，
                        # 用于验证主 Agent 对子任务 failed 状态的处理
                        stub_model.child_fails = True
                stub_model = StubModel(
                    case["stub"], harness_cb,
                    turn_scripts=case.get("stub_turns"),
                )
                patches.append(patch(
                    "fluxagent.core.agent.get_provider", return_value=stub_model))
                # 子 Agent 的模型也用同一替身：脚本会在子图内继续执行
                patches.append(patch.object(
                    subagents_mod, "get_provider", return_value=stub_model))
            else:
                stub_model = None

            import os as _os
            import functools
            import fluxagent.core.agent as fluxagent_core_agent
            from fluxagent.core.agent import create_agent_app
            trial_memory_store = MemoryStore(
                persist_path=os.path.join(memory_dir, "structured_memory.json")
            )
            if os.path.isfile(profile_path):
                ProfileCompatProvider(profile_path).migrate_once(trial_memory_store)
            for item in case["fixtures"].get("memory_records", []):
                scope = MemoryScope(str(item["scope"]))
                trial_memory_store.add(
                    scope,
                    str(item["value"]),
                    key=f"profile.{item['semantic_key']}",
                    source_id="synthetic_eval_fixture",
                )
            kwargs: dict[str, Any] = {
                "require_approval": bool(host.get("require_approval", False)),
                "enable_planner": bool(host.get("enable_planner", True)),
                # 真实运行使用 .env 配置的提供商/模型；替身模式下被 patch 覆盖
                "provider_name": self.provider_name or _os.getenv("DEFAULT_PROVIDER", "openai"),
                "model_name": self.model_name or _os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
                # The harness owns the trial lifecycle. Use the same durable
                # projection without starting a daemon worker per trial;
                # worker recovery is covered by dedicated unit tests.
                "memory_store": trial_memory_store,
                "checkpointer": MemorySaver(),
            }
            if host.get("context_trigger_turns"):
                kwargs["context_trigger_turns"] = host["context_trigger_turns"]
                kwargs["context_keep_turns"] = host.get("context_keep_turns", 1)
            if host.get("enable_replan") is False:
                kwargs["enable_replan"] = False
            if host.get("memory_retrieval_mode"):
                kwargs["memory_retrieval_mode"] = host["memory_retrieval_mode"]
            if not stub:
                from fluxagent.core.budget import RunBudget
                kwargs["budget"] = RunBudget(max_llm_calls=24, max_wall_seconds=240)
                # 网络防护：包装 get_provider，单请求超时 45s、最多重试 1 次。
                # 必须在 patches.start() 之前入列，否则包装不生效。
                trial_key = self.key_pool.acquire() if self.key_pool else None
                # 多 key 模式：SUT 每个 trial 独占一个账号 key，账号间限流互不影响；
                # 主图与子图的 get_provider 都强注入 trial 的 key/base_url/model
                def _inject(model_get, key=trial_key):
                    def wrapped(**call_kwargs):
                        if key:
                            call_kwargs["api_key"] = key
                            call_kwargs["base_url"] = self.base_url
                            call_kwargs["model_name"] = self.sut_model
                            call_kwargs["provider_name"] = "openai"
                        call_kwargs.setdefault("timeout", 90)
                        call_kwargs["max_retries"] = 1
                        return model_get(**call_kwargs)
                    return wrapped
                patches.append(patch("fluxagent.core.agent.get_provider",
                                     new=_inject(fluxagent_core_agent.get_provider)))
                patches.append(patch.object(subagents_mod, "get_provider",
                                            new=_inject(subagents_mod.get_provider)))

            try:
                for patcher in patches:
                    patcher.start()

                if host.get("load_skills"):
                    skill_loader_mod._lazy_loader.clear_cache()
                    tools = list(BUILTIN_TOOLS) + skill_loader_mod.load_dynamic_skills(force_rescan=True)
                else:
                    tools = list(BUILTIN_TOOLS)

                # 故障注入：目标 office 工具首次调用抛瞬时错误，验证恢复链路
                scripted_steps = list(case.get("stub") or [])
                for turn_script in case.get("stub_turns") or []:
                    scripted_steps.extend(turn_script)
                harness_actions = [s.get("harness") for s in scripted_steps]
                if stub and "inject_error_once" in harness_actions:
                    fault_step = next(s for s in case["stub"] if s.get("harness") == "inject_error_once")
                    target_name = fault_step.get("target", "read_office_file")
                    state = {"failed": False}
                    original = next(t for t in tools if t.name == target_name)
                    from langchain_core.tools import StructuredTool

                    def flaky(**tool_kwargs):
                        if not state["failed"]:
                            state["failed"] = True
                            # 产品风格的结构化错误信封：与真实瞬态故障在恢复
                            # 管线中的表现一致（extract_tool_failure_texts 可识别）
                            return json.dumps(
                                {"ok": False, "display": "❌ 工具错误",
                                 "error": {"code": "transient_error",
                                           "message": fault_step.get("error", "transient harness fault"),
                                           "retryable": True}},
                                ensure_ascii=False)
                        return original.invoke(tool_kwargs)

                    flaky_tool = StructuredTool.from_function(
                        func=flaky, name=target_name,
                        description=str(getattr(original, "description", "")),
                        args_schema=getattr(original, "args_schema", None))
                    tools[tools.index(original)] = flaky_tool

                enable_sub = host.get("enable_subagents", False)
                app = create_agent_app(tools=tools, enable_subagents=enable_sub, **kwargs)

                config = {"configurable": {"thread_id": f"judge-{case['case_id']}-{trial_index}"}}
                turns = list(case.get("turns") or [case["user_prompt"]])
                final_answer = ""
                tool_messages = []
                model_tool_calls = []
                turn_evidence = []
                previous_message_count = 0
                result = {}
                total_llm_calls = 0
                for turn_index, turn in enumerate(turns):
                    turn_config = {
                        "configurable": {
                            **config["configurable"],
                            "stub_turn_index": turn_index,
                        }
                    }
                    result = app.invoke(
                        {"messages": [HumanMessage(content=str(turn))]}, turn_config
                    )
                    if case.get("family") == "M":
                        pending = pending_approval(app, turn_config)
                        if pending:
                            result = approve(app, turn_config, pending)
                    messages = list(result.get("messages") or [])
                    # A checkpoint returns the complete thread. Only collect
                    # the newly produced messages for this turn, otherwise
                    # earlier tool results would be counted repeatedly.
                    new_messages = messages[previous_message_count:]
                    for message in new_messages:
                        if case.get("family") == "M" and isinstance(message, AIMessage):
                            for call in getattr(message, "tool_calls", None) or []:
                                if call.get("name") == "propose_memory_change":
                                    raw_args = dict(call.get("args") or {})
                                    model_tool_calls.append({
                                        "name": "propose_memory_change",
                                        "args": {
                                            key: str(raw_args.get(key) or "")[:500]
                                            for key in (
                                                "operation", "scope", "semantic_key", "value", "evidence"
                                            )
                                        },
                                        "turn_index": turn_index,
                                    })
                    turn_tools = [
                        {"tool": m.name, "content": str(m.content)[:600]}
                        for m in new_messages if getattr(m, "type", "") == "tool"
                    ]
                    tool_messages.extend(turn_tools)
                    final_answer = str(messages[-1].content) if messages else ""
                    turn_llm_calls = int(result.get("llm_calls") or 0)
                    total_llm_calls += turn_llm_calls
                    turn_evidence.append({
                        "turn_index": turn_index,
                        "user_text": str(turn),
                        "final_answer": final_answer,
                        "tool_messages": turn_tools,
                        "llm_calls": turn_llm_calls,
                    })
                    previous_message_count = len(messages)
                llm_calls = total_llm_calls
            finally:
                worker = getattr(locals().get("app"), "memory_worker", None)
                if worker is not None:
                    worker.stop(timeout=10.0)
                for patcher in reversed(patches):
                    patcher.stop()
                office_after = _hash_office(office_dir)
                profile_after = ""
                if os.path.exists(profile_path):
                    with open(profile_path, encoding="utf-8") as fh:
                        profile_after = fh.read()

            evidence = {
                "trial_id": trial_id,
                "case_id": case["case_id"],
                "track": case["track"],
                "stub": stub,
                "events": recorder.events,
                "final_answer": final_answer,
                "tool_messages": tool_messages,
                "model_tool_calls": model_tool_calls,
                "turns": turn_evidence,
                "office_before": office_before,
                "office_after": office_after,
                "profile_before": profile_before,
                "profile_after": profile_after,
                "observed_metrics": {
                    "wall_time_seconds": round(time.time() - started, 3),
                    "llm_calls": llm_calls,
                },
            }
            evidence["independent_checks"] = self._run_checks(case, evidence, tasks_file)
            return evidence

    # ---------------- 程序断言 ----------------

    def _run_checks(self, case: dict, evidence: dict, tasks_file: str | None) -> list[dict]:
        checks: list[dict] = []
        prog = case.get("programmatic") or {}
        all_results: dict[str, list[str]] = {}
        for tm in evidence["tool_messages"]:
            all_results.setdefault(tm["tool"], []).append(tm["content"])
        tool_results = {name: contents[-1] for name, contents in all_results.items()}

        def add(check_id, status, expected, actual):
            checks.append({"check_id": check_id, "status": status,
                           "expected": expected, "actual": str(actual)[:200]})

        proposal_expectation = prog.get("memory_proposal")
        if proposal_expectation:
            proposals = [call for call in evidence.get("model_tool_calls", [])
                         if call.get("name") == "propose_memory_change"]
            should_propose = bool(proposal_expectation.get("should_propose"))
            add("memory_proposal_presence",
                "pass" if bool(proposals) == should_propose else "fail",
                "proposal expected" if should_propose else "no proposal expected",
                f"{len(proposals)} proposal(s)")
            if should_propose and proposals:
                expected_operation = proposal_expectation.get("operation")
                expected_scope = proposal_expectation.get("scope")
                current_user = str((case.get("turns") or [case["user_prompt"]])[-1])
                matching = [
                    proposal for proposal in proposals
                    if (not expected_operation or (proposal.get("args") or {}).get("operation") == expected_operation)
                    and (not expected_scope or (proposal.get("args") or {}).get("scope") == expected_scope)
                ]
                grounded = [
                    proposal for proposal in matching
                    if _grounded_memory_evidence(
                        str((proposal.get("args") or {}).get("evidence") or ""),
                        current_user,
                    )
                ] if matching else []
                # 模型可能先发出格式错误的调用，再根据 ToolMessage 修正。
                # 断言应检查最终可执行的提案，而不是第一条失败尝试。
                matching = grounded or matching
                args = (matching[0] if matching else proposals[0]).get("args") or {}
                if expected_operation:
                    add("memory_proposal_operation",
                        "pass" if args.get("operation") == expected_operation else "fail",
                        expected_operation, args.get("operation"))
                if expected_scope:
                    add("memory_proposal_scope",
                        "pass" if args.get("scope") == expected_scope else "fail",
                        expected_scope, args.get("scope"))
                evidence_text = str(args.get("evidence") or "")
                add("memory_proposal_evidence",
                    "pass" if _grounded_memory_evidence(evidence_text, current_user) else "fail",
                    "evidence is a substring of the latest user message", evidence_text)
            writes = [event for event in evidence.get("events", [])
                      if event.get("event") == "memory_event_appended"
                      and event.get("event_type") == "memory_change"]
            add("memory_commit_after_review",
                "pass" if bool(writes) == should_propose else "fail",
                "approved proposal is committed" if should_propose else "no memory event is committed",
                len(writes))

        if "final_contains" in prog:
            for token in prog["final_contains"]:
                add(f"final::{token[:20]}",
                    "pass" if token in evidence["final_answer"] else "fail",
                    f"final_answer contains {token!r}", evidence["final_answer"][:120])

        if "tool_result_contains" in prog:
            for tool_name, tokens in prog["tool_result_contains"].items():
                contents = all_results.get(tool_name, [])
                for token in tokens:
                    hit = any(token in content for content in contents)
                    add(f"tool_result::{tool_name}::{token[:16]}",
                        "pass" if hit else "fail",
                        f"{tool_name} any result contains {token!r}",
                        (contents[-1] if contents else "")[:120])

        if "profile_after_contains" in prog:
            for token in prog["profile_after_contains"]:
                add(
                    f"profile::{token[:20]}",
                    "pass" if token in evidence.get("profile_after", "") else "fail",
                    f"profile contains {token!r}",
                    evidence.get("profile_after", "")[:160],
                )

        if prog.get("profile_unchanged"):
            add(
                "profile_unchanged",
                "pass" if evidence.get("profile_before") == evidence.get("profile_after") else "fail",
                "profile remains unchanged",
                "unchanged" if evidence.get("profile_before") == evidence.get("profile_after") else "changed",
            )

        if prog.get("context_memory_zero"):
            context_events = [
                event for event in evidence.get("events", [])
                if event.get("event") == "context_compiled"
            ]
            leaked = [event.get("memory_records") for event in context_events if event.get("memory_records")]
            add(
                "context_memory_zero",
                "pass" if not leaked else "fail",
                "no memory records injected",
                "none" if not leaked else leaked[-1],
            )

        if "executor_calls_zero" in prog:
            for tool_name in prog["executor_calls_zero"]:
                dispatched = any(
                    ev["event"] == "tool_call" and ev.get("tool") == tool_name
                    for ev in evidence["events"]
                )
                add(f"executor_zero::{tool_name}",
                    "fail" if dispatched else "pass",
                    f"no dispatch of {tool_name}", "dispatched" if dispatched else "none")

        if "executor_calls_exact" in prog:
            for tool_name, expected_n in prog["executor_calls_exact"].items():
                n = sum(1 for ev in evidence["events"]
                        if ev["event"] == "tool_call" and ev.get("tool") == tool_name)
                add(f"executor_exact::{tool_name}",
                    "pass" if n == expected_n else "fail",
                    f"{expected_n} dispatch(es) of {tool_name}", n)

        if "shell_exec_max" in prog:
            # 真实 shell 执行的唯一标记：sandbox 输出的 "● 当前系统" 行
            n = sum(1 for tm in evidence["tool_messages"] if "● 当前系统" in tm["content"])
            add("shell_exec_max",
                "pass" if n <= prog["shell_exec_max"] else "fail",
                f"shell executions <= {prog['shell_exec_max']}", n)

        if "audit_events_min" in prog:
            seen = {ev["event"] for ev in evidence["events"]}
            for event_name in prog["audit_events_min"]:
                add(f"audit::{event_name}",
                    "pass" if event_name in seen else "fail",
                    f"event {event_name} present", event_name in seen)

        # 通用：office 快照对比（作为"零副作用"证据）
        if prog.get("office_unchanged"):
            add("office_unchanged",
                "pass" if evidence["office_before"] == evidence["office_after"] else "fail",
                "office snapshot identical",
                "identical" if evidence["office_before"] == evidence["office_after"] else "changed")

        # tasks.json 检查（I 族）
        if tasks_file and os.path.exists(tasks_file):
            try:
                with open(tasks_file, encoding="utf-8") as fh:
                    records = json.load(fh)
                add("tasks_json_valid", "pass" if isinstance(records, list) and records else "fail",
                    "tasks.json contains records", len(records) if isinstance(records, list) else "invalid")
            except (OSError, json.JSONDecodeError) as exc:
                add("tasks_json_valid", "fail", "tasks.json readable", str(exc)[:80])
        return checks
