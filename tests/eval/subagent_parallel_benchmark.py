# -*- coding: utf-8 -*-
"""Real-LLM benchmark for autonomous delegation on realistic work requests.

The hierarchical arm exposes delegation but never tells the model to use it.
Correctness is judged from evidence and output; delegation is an observation,
except that delegating a simple lookup is counted as avoidable over-delegation.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics
import tempfile
import threading
import time
from typing import Any
from unittest.mock import patch

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver

from fluxagent.capability.context_compiler import ContextCompiler, MemoryStore
from fluxagent.core.agent import create_agent_app
from fluxagent.core.approval import apending_approval, areject_prepare
from fluxagent.core.permission_broker import PermissionBroker
from fluxagent.core.policy import PolicyConfig, RiskLevel
from fluxagent.core.subagents import SubagentSpec
from fluxagent.core.tools import sandbox_tools


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

TRANSIENT_MARKERS = (
    "429", "ratelimit", "rate limit", "concurrent limit", "timeout",
    "temporarily unavailable", "connection error", "502", "503", "504",
)


def _is_transient_text(value: Any) -> bool:
    text = str(value or "").casefold()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _noise(prefix: str, count: int = 140) -> list[str]:
    return [
        f"{prefix} background paragraph {index:03d}: ordinary operational notes "
        "without acceptance data or final conclusions."
        for index in range(count)
    ]


def _document(label: str, facts: list[tuple[str, str]]) -> str:
    lines = _noise(label)
    slots = (17, 49, 91, 126)
    for slot, (field, value) in zip(slots, facts):
        lines.insert(slot, f"VERIFIED {field} = {value}")
    return "\n".join(lines)


def _case(
    case_id: str,
    family: str,
    prompt: str,
    *,
    fixture: dict[str, str] | None = None,
    checks: dict[str, dict[str, Any]] | None = None,
    expected_tokens: tuple[str, ...] = (),
    expected_any: tuple[tuple[str, ...], ...] = (),
    required_reads: tuple[str, ...] = (),
    required_checks: tuple[str, ...] = (),
    delegation_policy: str = "optional",
    parallelizable: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "family": family,
        "prompt": prompt,
        "fixture": fixture or {},
        "checks": checks or {},
        "expected_tokens": expected_tokens,
        "expected_any": expected_any,
        "required_reads": required_reads,
        "required_checks": required_checks,
        "delegation_policy": delegation_policy,
        "parallelizable": parallelizable,
    }


def build_cases(per_family: int = 3) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(1, per_family + 1):
        suffix = f"{index:02d}"
        approval_code = f"src/approval_{suffix}.py"
        approval_policy = f"config/approval_policy_{suffix}.yaml"
        approval_tests = f"reports/approval_tests_{suffix}.txt"
        recovery_code = f"src/recovery_{suffix}.py"
        recovery_tests = f"reports/recovery_tests_{suffix}.txt"
        approval_doc = f"docs/approval_release_{suffix}.md"
        risk_id = f"RISK-REPLAY-{suffix}"
        failed_test = f"test_replay_guard_{suffix}"
        action_rule = f"one-shot-action-{suffix}"
        recovery_risk = f"RISK-UNKNOWN-STATE-{suffix}"
        recovery_failed_test = f"test_crash_recovery_{suffix}"
        recovery_rule = f"reconcile-before-retry-{suffix}"
        cases.append(_case(
            f"release-review-{suffix}",
            "release_readiness_review",
            (
                "请完成审批模块上线评审。第一条调查线核对"
                f" {approval_code}、{approval_policy}、{approval_tests} 的重复执行保护；"
                f"第二条调查线核对 {recovery_code}、{recovery_tests}、{approval_doc} 的崩溃恢复。"
                "最终给出统一上线结论，列全两个阻断风险、对应失败测试、必须满足的规则，并逐项标注来源文件。"
            ),
            fixture={
                approval_code: _document("APPROVAL-CODE", [
                    ("replay_risk", risk_id),
                    ("implemented_state", "approved-to-executing-without-compare-and-set"),
                ]),
                approval_policy: _document("APPROVAL-POLICY", [
                    ("required_rule", action_rule),
                    ("current_guard", "plain-update-without-version-check"),
                ]),
                approval_tests: _document("APPROVAL-TESTS", [
                    ("failed_test", failed_test),
                    ("test_result", "FAILED-duplicate-execution-observed"),
                ]),
                recovery_code: _document("RECOVERY-CODE", [
                    ("recovery_risk", recovery_risk),
                    ("current_behavior", "retry-unknown-action-without-reconcile"),
                ]),
                recovery_tests: _document("RECOVERY-TESTS", [
                    ("failed_test", recovery_failed_test),
                    ("test_result", "FAILED-unknown-action-replayed"),
                ]),
                approval_doc: _document("APPROVAL-DOC", [
                    ("recovery_rule", recovery_rule),
                    ("release_gate", "zero-duplicate-side-effects"),
                ]),
            },
            expected_tokens=(
                risk_id, failed_test, action_rule,
                recovery_risk, recovery_failed_test, recovery_rule,
            ),
            expected_any=((
                "不可上线", "不能上线", "不建议上线", "不予上线",
                "阻断上线", "禁止上线", "不得上线", "不予通过",
                "不通过上线评审",
            ),),
            required_reads=(
                approval_code, approval_policy, approval_tests,
                recovery_code, recovery_tests, approval_doc,
            ),
            parallelizable=True,
        ))

        gateway_log = f"logs/gateway_incident_{suffix}.log"
        gateway_code = f"src/gateway_{suffix}.py"
        gateway_config = f"config/gateway_{suffix}.yaml"
        worker_log = f"logs/worker_incident_{suffix}.log"
        worker_code = f"src/worker_{suffix}.py"
        worker_config = f"config/worker_{suffix}.yaml"
        incident_id = f"INC-DEADLETTER-{suffix}"
        config_fault = f"ack-before-persist-{suffix}"
        fix_id = f"persist-before-ack-{suffix}"
        gateway_fault = f"retry-without-idempotency-{suffix}"
        gateway_fix = f"stable-request-key-{suffix}"
        cases.append(_case(
            f"incident-{suffix}",
            "incident_diagnosis",
            (
                "线上订单出现重复受理后又丢消息。请分别调查网关重试链路"
                f"（{gateway_log}、{gateway_code}、{gateway_config}）和 worker 持久化链路"
                f"（{worker_log}、{worker_code}、{worker_config}），"
                "最后说明两个根因如何共同导致事故，并给出两项修复及逐项证据来源。"
            ),
            fixture={
                gateway_log: _document("GATEWAY-LOG", [
                    ("incident", incident_id),
                    ("symptom", "same-order-accepted-twice-after-timeout"),
                ]),
                gateway_code: _document("GATEWAY-CODE", [
                    ("retry_fault", gateway_fault),
                    ("required_fix", gateway_fix),
                ]),
                gateway_config: _document("GATEWAY-CONFIG", [
                    ("retry_count", "three-without-stable-key"),
                    ("idempotency", "disabled"),
                ]),
                worker_log: _document("WORKER-LOG", [
                    ("incident", incident_id),
                    ("symptom", "acknowledged-message-missing-after-crash"),
                ]),
                worker_code: _document("WORKER-CODE", [
                    ("current_order", config_fault),
                    ("required_fix", fix_id),
                ]),
                worker_config: _document("WORKER-CONFIG", [
                    ("delivery_mode", config_fault),
                    ("durability", "disabled"),
                ]),
            },
            expected_tokens=(incident_id, gateway_fault, gateway_fix, config_fault, fix_id),
            required_reads=(
                gateway_log, gateway_code, gateway_config,
                worker_log, worker_code, worker_config,
            ),
            parallelizable=True,
        ))

        api_path = f"api/orders_{suffix}.py"
        api_test_path = f"reports/api_contract_{suffix}.txt"
        contract_path = f"docs/orders_contract_{suffix}.md"
        web_path = f"clients/web_orders_{suffix}.ts"
        mobile_path = f"clients/mobile_orders_{suffix}.kt"
        migration_path = f"docs/orders_migration_{suffix}.md"
        suite = f"orders-contract-{suffix}"
        server_field = f"payment_state_{suffix}"
        stale_field = f"payment_status_{suffix}"
        failed_contract_test = f"test_orders_contract_{suffix}"
        mobile_field = f"paymentStateV2_{suffix}"
        migration_rule = f"dual-read-window-{suffix}"
        cases.append(_case(
            f"change-impact-{suffix}",
            "change_impact_analysis",
            (
                "订单接口准备发布。请先审查服务端契约线"
                f"（{api_path}、{api_test_path}、{contract_path}），再审查客户端兼容线"
                f"（{web_path}、{mobile_path}、{migration_path}），并运行检查套件 {suite}。"
                "最终说明各端字段差异、失败检查、迁移要求和具体修改位置，所有判断都要带来源。"
            ),
            fixture={
                api_path: _document("ORDER-API", [("response_field", server_field)]),
                api_test_path: _document("ORDER-API-TEST", [("failed_test", failed_contract_test)]),
                contract_path: _document("ORDER-CONTRACT", [("required_field", server_field)]),
                web_path: _document("ORDER-WEB", [("consumed_field", stale_field)]),
                mobile_path: _document("ORDER-MOBILE", [("consumed_field", mobile_field)]),
                migration_path: _document("ORDER-MIGRATION", [("required_rule", migration_rule)]),
            },
            checks={suite: {
                "ok": False,
                "failed_test": failed_contract_test,
                "reason": f"client-reads-{stale_field}-but-api-returns-{server_field}",
            }},
            expected_tokens=(
                server_field, stale_field, mobile_field,
                failed_contract_test, migration_rule,
            ),
            required_reads=(
                api_path, api_test_path, contract_path,
                web_path, mobile_path, migration_path,
            ),
            required_checks=(suite,),
            parallelizable=True,
        ))

        oncall_path = f"docs/oncall_{suffix}.md"
        owner = f"payments-team-{suffix}"
        command = f"flux status payments-{suffix}"
        cases.append(_case(
            f"simple-lookup-{suffix}",
            "simple_lookup_control",
            f"查看 {oncall_path}，告诉我支付服务的负责人和状态检查命令。",
            fixture={oncall_path: _document("ONCALL", [
                ("owner", owner),
                ("status_command", command),
            ])},
            expected_tokens=(owner, command),
            required_reads=(oncall_path,),
            delegation_policy="avoid",
            parallelizable=False,
        ))
    return cases


class MetricsCallback(BaseCallbackHandler):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        with self._lock:
            self.llm_calls += 1

    def on_llm_end(self, response, **kwargs):
        usage = getattr(response, "llm_output", None) or {}
        usage = usage.get("token_usage") or usage.get("usage") or {}
        with self._lock:
            self.input_tokens += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)


class EventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def log_event(self, **event: Any) -> None:
        self.events.append(event)


class BenchmarkTools:
    def __init__(self, case: dict[str, Any], delay_seconds: float) -> None:
        self.case = case
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.calls: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0

    def _begin(self, name: str, args: dict[str, Any]) -> float:
        started = time.monotonic()
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append({"name": name, "args": dict(args), "started": started})
        return started

    def _finish(self, started: float) -> None:
        finished = time.monotonic()
        with self._lock:
            self.active -= 1
            for call in reversed(self.calls):
                if call.get("started") == started and "finished" not in call:
                    call["finished"] = finished
                    call["duration"] = finished - started
                    break

    def tools(self) -> list[StructuredTool]:
        def read(filepath: str) -> str:
            started = self._begin("read_office_file", {"filepath": filepath})
            try:
                return sandbox_tools.read_office_file.invoke({"filepath": filepath})
            finally:
                self._finish(started)

        def run_project_checks(suite: str) -> str:
            started = self._begin("run_project_checks", {"suite": suite})
            try:
                time.sleep(self.delay_seconds)
                value = self.case["checks"].get(suite)
                if value is None:
                    return json.dumps({
                        "ok": False,
                        "error": {"code": "not_found", "message": "unknown check suite", "retryable": False},
                    }, ensure_ascii=False)
                return json.dumps(value, ensure_ascii=False)
            finally:
                self._finish(started)

        read_tool = StructuredTool.from_function(
            func=read,
            name="read_office_file",
            description="Read one UTF-8 file from the isolated office workspace.",
            args_schema=getattr(sandbox_tools.read_office_file, "args_schema", None),
            metadata={"risk": "read"},
        )
        checks_tool = StructuredTool.from_function(
            func=run_project_checks,
            name="run_project_checks",
            description="Run one named project check suite and return its structured result.",
            metadata={"risk": "read"},
        )
        return [read_tool, checks_tool]


def _materialize(directory: Path, fixture: dict[str, str]) -> None:
    for relative, content in fixture.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _final_text(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", None) == "ai" and getattr(message, "content", None):
            return str(message.content)
    return ""


def _evaluate(case: dict[str, Any], text: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    tokens_ok = all(token in text for token in case["expected_tokens"])
    alternatives_ok = all(
        any(candidate in text for candidate in alternatives)
        for alternatives in case["expected_any"]
    )
    read_paths = {
        str(call.get("args", {}).get("filepath") or "").replace("\\", "/")
        for call in calls if call.get("name") == "read_office_file"
    }
    check_suites = {
        str(call.get("args", {}).get("suite") or "")
        for call in calls if call.get("name") == "run_project_checks"
    }
    reads_ok = all(path.replace("\\", "/") in read_paths for path in case["required_reads"])
    checks_ok = all(suite in check_suites for suite in case["required_checks"])
    citations_ok = all(path in text for path in case["required_reads"])
    evidence_ok = reads_ok and checks_ok
    return {
        "correct": tokens_ok and alternatives_ok and evidence_ok and citations_ok,
        "tokens_ok": tokens_ok,
        "alternatives_ok": alternatives_ok,
        "evidence_correct": evidence_ok,
        "citations_ok": citations_ok,
        "reads_ok": reads_ok,
        "checks_ok": checks_ok,
    }


async def run_trial(
    case: dict[str, Any],
    arm: str,
    repeat: int,
    provider: str,
    model: str,
    delay_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    callback = MetricsCallback()
    recorder = EventRecorder()
    benchmark_tools = BenchmarkTools(case, delay_seconds)

    with tempfile.TemporaryDirectory(
        prefix="fluxagent-parallel-benchmark-", ignore_cleanup_errors=True
    ) as raw_directory:
        root = Path(raw_directory)
        office = root / "office"
        memory_dir = root / "memory"
        office.mkdir()
        memory_dir.mkdir()
        _materialize(office, case["fixture"])
        tools = benchmark_tools.tools()
        broker = PermissionBroker(root / "state.sqlite3")
        store = MemoryStore()
        compiler = ContextCompiler(store, token_budget=2400)
        try:
            with patch.object(sandbox_tools, "OFFICE_DIR", str(office)), \
                 patch("fluxagent.core.agent.MEMORY_DIR", str(memory_dir)), \
                 patch("fluxagent.core.agent.audit_logger", recorder), \
                 patch("fluxagent.core.subagents.audit_logger", recorder):
                subagent_specs = [
                    SubagentSpec(
                        "code_reviewer",
                        "审查代码实现、缺陷和变更影响。",
                        "阅读必要代码并返回结论、证据路径、风险和缺失信息。",
                        tuple(tools),
                    ),
                    SubagentSpec(
                        "document_analyst",
                        "核对文档、日志、配置和验收要求。",
                        "只读取任务必要材料，返回有来源的事实、矛盾和缺失信息。",
                        tuple(tools),
                    ),
                ]
                app = create_agent_app(
                    provider_name=provider,
                    model_name=model,
                    tools=tools,
                    checkpointer=MemorySaver(),
                    permission_broker=broker,
                    memory_store=store,
                    context_compiler=compiler,
                    enable_subagents=arm == "hierarchical",
                    subagent_specs=subagent_specs,
                    dynamic_subagent_tools=tools,
                    enable_planner=True,
                    policy=PolicyConfig(
                        tool_base_overrides={
                            "read_office_file": RiskLevel.READ,
                            "run_project_checks": RiskLevel.READ,
                        }
                    ),
                )
                config = {
                    "configurable": {
                        "thread_id": f"parallel-{case['case_id']}-{arm}-{repeat}",
                        "run_id": f"parallel-parent-{case['case_id']}-{repeat}",
                    },
                    "callbacks": [callback],
                }
                state = await asyncio.wait_for(
                    app.ainvoke({"messages": [HumanMessage(content=case["prompt"])]}, config),
                    timeout=300,
                )
                while True:
                    pending = await apending_approval(app, config)
                    if not pending:
                        break
                    await areject_prepare(app, config, pending, reason="benchmark_read_only")
                    state = await asyncio.wait_for(app.ainvoke(None, config), timeout=300)

                transient_graph_failures = [
                    event for event in recorder.events
                    if event.get("action") in {"plan_failed", "replan_failed"}
                    and _is_transient_text(event.get("error"))
                ]
                if transient_graph_failures:
                    raise RuntimeError(str(transient_graph_failures[-1].get("error") or "provider unavailable"))

                text = _final_text(state)
                checks = _evaluate(case, text, benchmark_tools.calls)
                delegation_count = sum(
                    1 for event in recorder.events if event.get("action") == "subagent_started"
                )
                delegation_appropriate = not (
                    arm == "hierarchical"
                    and case["delegation_policy"] == "avoid"
                    and delegation_count > 0
                )
                child_durations = [
                    float(event.get("duration_ms") or 0) / 1000
                    for event in recorder.events
                    if str(event.get("action") or "").startswith("subagent_")
                    and event.get("duration_ms") is not None
                    and event.get("action") != "subagent_approval_required"
                ]
                elapsed = time.monotonic() - started
                critical_path = max(child_durations, default=0.0)
                summary = state.get("last_run_summary") or {}
                return {
                    "benchmark_version": "autonomous-delegation-v2",
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "prompt": case["prompt"],
                    "parallelizable": case["parallelizable"],
                    "arm": arm,
                    "repeat": repeat,
                    "result": checks,
                    "qualified_success": checks["correct"] and delegation_appropriate,
                    "delegation_count": delegation_count,
                    "delegated": delegation_count > 0,
                    "delegation_policy": case["delegation_policy"],
                    "delegation_appropriate": delegation_appropriate,
                    "final_text": text[:4000],
                    "task_status": state.get("task_status", ""),
                    "task_plan": [
                        {
                            "action": str(step.get("action") or ""),
                            "status": str(step.get("status") or ""),
                        }
                        for step in (state.get("task_plan") or [])
                        if isinstance(step, dict)
                    ],
                    "elapsed_seconds": round(elapsed, 3),
                    "subagent_critical_path_seconds": round(critical_path, 3),
                    "parent_overhead_seconds": round(max(0.0, elapsed - critical_path), 3),
                    "parent_wait_fraction": round(critical_path / elapsed, 4) if critical_path else 0.0,
                    "child_total_seconds": round(sum(child_durations), 3),
                    "llm_calls": callback.llm_calls or int(summary.get("llm_calls") or 0),
                    "input_tokens": callback.input_tokens,
                    "output_tokens": callback.output_tokens,
                    "tool_calls": [
                        {
                            "name": item["name"],
                            "args": item["args"],
                            "started_offset": round(float(item.get("started") or started) - started, 4),
                            "finished_offset": round(float(item.get("finished") or started) - started, 4),
                            "duration": round(float(item.get("duration") or 0), 4),
                        }
                        for item in benchmark_tools.calls
                    ],
                    "configured_tool_delay_seconds": delay_seconds,
                    "tool_max_concurrency": benchmark_tools.max_active,
                    "subagent_statuses": [
                        str(event.get("action") or "").removeprefix("subagent_")
                        for event in recorder.events
                        if str(event.get("action") or "").startswith("subagent_")
                        and event.get("action") != "subagent_started"
                    ],
                }
        finally:
            broker.close()


async def _run_with_backoff(
    case: dict[str, Any], arm: str, repeat: int, provider: str, model: str, delay_seconds: float
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            return await run_trial(case, arm, repeat, provider, model, delay_seconds)
        except Exception as exc:  # noqa: BLE001 - live provider boundary
            text = str(exc).casefold()
            quota_window = "period request limit" in text or "effective_max" in text
            if quota_window:
                delay = 600.0
                print(f"[quota-wait] {case['case_id']} {arm}; sleep={delay:.0f}s", flush=True)
                await asyncio.sleep(delay)
                continue
            rate_limited = "429" in text or "rate limit" in text or "concurrent limit" in text
            if rate_limited:
                delay = min(60.0, 5.0 * (2 ** min(attempt, 4)))
                print(f"[rate-wait] {case['case_id']} {arm}; sleep={delay:.0f}s", flush=True)
                await asyncio.sleep(delay)
                attempt += 1
                continue
            if not _is_transient_text(exc) or attempt == 4:
                raise
            delay = min(60.0, 5.0 * (2 ** attempt))
            print(f"[retry] {case['case_id']} {arm}; sleep={delay:.0f}s", flush=True)
            await asyncio.sleep(delay)
            attempt += 1


async def run_all(
    cases: list[dict[str, Any]],
    repeat: int,
    provider: str,
    model: str,
    delay_seconds: float,
    output_jsonl: Path,
    resume: bool,
) -> list[dict[str, Any]]:
    expected = {
        f"{case['case_id']}:{arm}:{trial}"
        for case in cases for trial in range(repeat) for arm in ("single_agent", "hierarchical")
    }
    records_by_key: dict[str, dict[str, Any]] = {}
    if resume and output_jsonl.exists():
        for line in output_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = f"{record['case_id']}:{record['arm']}:{record['repeat']}"
            if key in expected:
                records_by_key[key] = record
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("a", encoding="utf-8") as handle:
        for case in cases:
            for trial in range(repeat):
                for arm in ("single_agent", "hierarchical"):
                    key = f"{case['case_id']}:{arm}:{trial}"
                    if key in records_by_key:
                        continue
                    record = await _run_with_backoff(
                        case, arm, trial, provider, model, delay_seconds
                    )
                    records_by_key[key] = record
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(
                        f"{len(records_by_key):>3}/{len(expected)} {record['case_id']:<22} "
                        f"{arm:<13} ok={record['result']['correct']} "
                        f"delegate={record['delegation_count']} t={record['elapsed_seconds']}s",
                        flush=True,
                    )
    return [records_by_key[key] for key in sorted(records_by_key)]


def _rate(rows: list[dict[str, Any]], predicate) -> float | None:
    return round(sum(1 for row in rows if predicate(row)) / len(rows), 4) if rows else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return round(ordered[index], 3)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    return {
        "n": len(rows),
        "correct_rate": _rate(rows, lambda row: row["result"]["correct"]),
        "evidence_correct_rate": _rate(rows, lambda row: row["result"]["evidence_correct"]),
        "qualified_success_rate": _rate(rows, lambda row: row["qualified_success"]),
        "delegation_rate": _rate(rows, lambda row: row.get("delegated", False)),
        "delegation_appropriate_rate": _rate(rows, lambda row: row["delegation_appropriate"]),
        "mean_elapsed_seconds": round(statistics.mean(elapsed), 3) if elapsed else None,
        "p50_elapsed_seconds": _percentile(elapsed, 0.5),
        "p95_elapsed_seconds": _percentile(elapsed, 0.95),
        "mean_llm_calls": round(statistics.mean(row["llm_calls"] for row in rows), 3) if rows else None,
        "mean_input_tokens": round(
            statistics.mean(row["input_tokens"] for row in rows), 3
        ) if rows else None,
        "mean_output_tokens": round(
            statistics.mean(row["output_tokens"] for row in rows), 3
        ) if rows else None,
        "total_input_tokens": sum(row["input_tokens"] for row in rows),
        "total_output_tokens": sum(row["output_tokens"] for row in rows),
        "mean_parent_wait_fraction": round(
            statistics.mean(row["parent_wait_fraction"] for row in rows), 4
        ) if rows else None,
        "mean_tool_max_concurrency": round(
            statistics.mean(row["tool_max_concurrency"] for row in rows), 3
        ) if rows else None,
    }


def summarize(records: list[dict[str, Any]], provider: str, model: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "benchmark_version": "autonomous-delegation-v2",
        "provider": provider,
        "model": model,
        "records": len(records),
        "arms": {},
        "families": {},
    }
    for arm in ("single_agent", "hierarchical"):
        result["arms"][arm] = _arm_summary([row for row in records if row["arm"] == arm])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[row["family"]].append(row)
    for family, rows in sorted(grouped.items()):
        result["families"][family] = {
            arm: _arm_summary([row for row in rows if row["arm"] == arm])
            for arm in ("single_agent", "hierarchical")
        }

    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        pairs[(row["case_id"], int(row["repeat"]))][row["arm"]] = row
    paired_rows = [pair for pair in pairs.values() if {"single_agent", "hierarchical"} <= pair.keys()]
    speedups = [
        float(pair["single_agent"]["elapsed_seconds"])
        / max(0.001, float(pair["hierarchical"]["elapsed_seconds"]))
        for pair in paired_rows
    ]
    parallel_pairs = [pair for pair in paired_rows if pair["hierarchical"]["parallelizable"]]
    parallel_speedups = [
        float(pair["single_agent"]["elapsed_seconds"])
        / max(0.001, float(pair["hierarchical"]["elapsed_seconds"]))
        for pair in parallel_pairs
    ]
    delegated_pairs = [
        pair for pair in paired_rows if pair["hierarchical"].get("delegated", False)
    ]
    direct_pairs = [
        pair for pair in paired_rows if not pair["hierarchical"].get("delegated", False)
    ]

    def pair_speedups(pairs_to_measure: list[dict[str, dict[str, Any]]]) -> list[float]:
        return [
            float(pair["single_agent"]["elapsed_seconds"])
            / max(0.001, float(pair["hierarchical"]["elapsed_seconds"]))
            for pair in pairs_to_measure
        ]

    def input_ratio(pairs_to_measure: list[dict[str, dict[str, Any]]]) -> float | None:
        single_tokens = sum(
            int(pair["single_agent"]["input_tokens"]) for pair in pairs_to_measure
        )
        hierarchical_tokens = sum(
            int(pair["hierarchical"]["input_tokens"]) for pair in pairs_to_measure
        )
        if not pairs_to_measure or single_tokens <= 0:
            return None
        return round(hierarchical_tokens / single_tokens, 4)

    delegated_speedups = pair_speedups(delegated_pairs)
    direct_speedups = pair_speedups(direct_pairs)
    result["paired"] = {
        "n": len(paired_rows),
        "mean_speedup_single_over_hierarchical": round(statistics.mean(speedups), 3) if speedups else None,
        "p50_speedup": _percentile(speedups, 0.5),
        "hierarchical_faster_rate": round(sum(value > 1 for value in speedups) / len(speedups), 4) if speedups else None,
        "parallel_n": len(parallel_pairs),
        "parallel_mean_speedup": round(statistics.mean(parallel_speedups), 3) if parallel_speedups else None,
        "parallel_p50_speedup": _percentile(parallel_speedups, 0.5),
        "parallel_hierarchical_faster_rate": round(
            sum(value > 1 for value in parallel_speedups) / len(parallel_speedups), 4
        ) if parallel_speedups else None,
        "both_correct_rate": round(
            sum(
                pair["single_agent"]["result"]["correct"]
                and pair["hierarchical"]["result"]["correct"]
                for pair in paired_rows
            ) / len(paired_rows), 4
        ) if paired_rows else None,
        "actually_delegated_n": len(delegated_pairs),
        "actually_delegated_mean_speedup": round(
            statistics.mean(delegated_speedups), 3
        ) if delegated_speedups else None,
        "actually_delegated_input_token_ratio": input_ratio(delegated_pairs),
        "not_delegated_n": len(direct_pairs),
        "not_delegated_mean_speedup": round(
            statistics.mean(direct_speedups), 3
        ) if direct_speedups else None,
    }
    hierarchical_rows = [row for row in records if row["arm"] == "hierarchical"]
    complex_rows = [row for row in hierarchical_rows if row["parallelizable"]]
    simple_rows = [row for row in hierarchical_rows if not row["parallelizable"]]
    result["autonomy"] = {
        "complex_n": len(complex_rows),
        "complex_delegation_rate": _rate(
            complex_rows, lambda row: row.get("delegated", False)
        ),
        "simple_n": len(simple_rows),
        "simple_overdelegation_rate": _rate(
            simple_rows, lambda row: row.get("delegated", False)
        ),
    }
    return result


async def amain(args: argparse.Namespace) -> None:
    cases = build_cases(args.per_family)
    if args.families:
        selected = {item.strip() for item in args.families.split(",") if item.strip()}
        cases = [case for case in cases if case["family"] in selected]
    records = await run_all(
        cases,
        args.repeat,
        args.provider,
        args.model,
        args.delay_seconds,
        PROJECT_ROOT / args.output_jsonl,
        args.resume,
    )
    summary = summarize(records, args.provider, args.model)
    output = PROJECT_ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real LLM benchmark for autonomous single-agent versus hierarchical work"
    )
    parser.add_argument("--provider", default=os.getenv("DEFAULT_PROVIDER", "other"))
    parser.add_argument("--model", default=os.getenv("DEFAULT_MODEL", "Deepseek-v4-flash"))
    parser.add_argument("--per-family", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=0.6)
    parser.add_argument("--families", default="", help="Comma-separated family filter")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-jsonl", default="artifacts/subagent_autonomy_benchmark.jsonl"
    )
    parser.add_argument(
        "--output-json", default="artifacts/subagent_autonomy_benchmark.json"
    )
    args = parser.parse_args()
    if args.per_family < 1 or args.repeat < 1 or args.delay_seconds < 0:
        parser.error("per-family/repeat must be positive and delay-seconds must be non-negative")
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
