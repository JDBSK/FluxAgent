# -*- coding: utf-8 -*-
"""Agent 状态图骨架——FluxAgent 的决策大脑。

本模块用 LangGraph StateGraph 搭建 Agent 工作流：
  initialize -> (planner) -> agent -> approval/tools/policy_block -> progress -> replan -> agent
  形成思考-调工具-看结果-再思考的循环，直到没有工具调用出答案。

节点职责：
  - initialize：每次运行前重置计数字段（llm_calls、tool_rounds、retry_count 等）
  - planner：复杂任务拆成 2-6 步计划，简单问答跳过
  - agent：LLM 推理，读记忆+画像+摘要，决定答题还是调工具
  - approval：副作用工具的人工审批闸门（interrupt_before 硬中断）
  - policy_block：高危参数的硬拒绝（执行前拦截）
  - tools：ToolNode 执行工具调用
  - progress：检查工具结果，决定成功/重试/重规划/预算熔断
  - replan：重新规划（整表替换 task_plan，不追加）

工作流 = 这张 StateGraph 状态图（有条件分支、有循环、有状态持久化）。
"""

import json
import os
import re
import sys
import threading
import time
from typing import Any, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .budget import (
    DEFAULT_BUDGET,
    RunBudget,
    apply_budget_fail,
    budget_violation,
    build_run_summary,
    bump_llm_calls,
    bump_tool_rounds,
    init_run_clock,
)
from .config import CONTEXT_KEEP_TURNS, CONTEXT_TRIGGER_TURNS, MEMORY_DIR
from .context import (
    STEP_DONE,
    STEP_FAILED,
    STEP_RUNNING,
    TASK_DONE,
    TASK_FAILED,
    TASK_CANCELLED,
    TASK_PENDING,
    TASK_REPLANNING,
    TASK_RUNNING,
    AgentState,
    advance_plan,
    make_plan_step,
    measure_context,
    repair_unpaired_tool_calls,
    render_plan,
    trim_context_messages,
)
from .logger import audit_logger
from .policy import SIDE_EFFECT_TOOLS, PolicyConfig, assess_tool_calls, build_policy_config, render_policy_decision
from .provider import get_provider
from .recovery import (
    ACTION_ABORT,
    ACTION_REPLAN,
    ACTION_RETRY,
    any_tool_failure,
    build_replan_prompt,
    classify_failure,
    extract_tool_failure_texts,
    fallback_replan_steps,
    failed_step_action,
    validate_replan_steps,
)


from .skill_loader import load_dynamic_skills
from .tools.builtins import BUILTIN_TOOLS

MAX_RETRY              = 3                                                   # 单个步骤的最大重试次数，超过则进入 replan
MAX_REPLAN             = 2                                                   # 单个任务的最大重规划次数，超过则失败
# These tools terminate an installation *step* after a successful transaction;
# they do not terminate the enclosing plan when later steps remain.
TERMINAL_INSTALL_TOOLS = frozenset({"install_skill", "install_mcp_server"})

POLICY_DENY_NOTE = (
    "策略拒绝：检测到高危/越权参数，已在执行前拦截。"
    "请不要重试同一非法参数，向用户说明并给出安全替代方案。"
)


def _needs_plan(text: str) -> bool:
    # 当前使用的是规则分类器，根据文本长度、动作数量、是否涉及多文件、是否需要工具等特征判断，减少简单请求的一次模型调用
    # 更高级的方法是混合路由，就是先用规则分类器快速判断，再用小模型（如 gpt-3.5-turbo-mini）做二次确认，最后再决定是否规划
    # 判断原理：如果文本长度超过60个字符，或者包含多个步骤关键词，则需要规划
    """Rule-based plan router. Avoid an extra LLM hop just to decide planning."""
    if not text:
        return False
    if len(text) > 60:
        return True
    multi_step = ["然后", "接着", "再", "并且", "之后", "先", "最后", "步骤", "依次"]
    hits = sum(1 for kw in multi_step if kw in text)
    if hits >= 2:
        return True

    # 如果文本中包含高危操作关键词，并且至少出现一次，则需要规划
    heavy = ["写", "创建", "删除", "修改", "部署", "重构", "执行", "运行", "清理", "整理", "生成"]
    return any(kw in text for kw in heavy) and hits >= 1


_UNVERIFIED_ACTION_CLAIMS = (
    "已创建",
    "创建成功",
    "已写入",
    "写入成功",
    "已保存",
    "保存成功",
    "已删除",
    "删除成功",
    "已修改",
    "修改成功",
    "已调用工具",
    "我已调用",
    "后台自动进行",
    "安装流程正在",
    "正在下载并解析",
    "我会为您监控进度",
    "预计剩余时间",
)


def _has_unverified_action_claim(content: Any) -> bool:
    """Detect success/tool-call claims that have no structured tool result."""
    if not isinstance(content, str):
        return False
    return any(claim in content for claim in _UNVERIFIED_ACTION_CLAIMS)


def _metadata_aliases(value: Any) -> set[str]:
    """Expand a capability/name into stable, metadata-derived aliases."""
    raw = str(value or "").strip()
    if not raw:
        return set()
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    spaced = re.sub(r"[_./:-]+", " ", spaced)
    variants = {raw.casefold(), spaced.casefold().strip()}
    # Component aliases make generated MCP names routable without a service
    # specific keyword table (e.g. ``mcp_vendor_tool`` -> ``vendor``).
    routing_stopwords = {
        "mcp", "skill", "server", "backend", "service", "tool", "sdk",
        "服务", "后端", "工具", "技能",
    }
    variants.update(
        part
        for part in spaced.casefold().split()
        if len(part) >= 3 and part not in routing_stopwords
    )
    if spaced.casefold().endswith(" mcp"):
        variants.add(spaced[:-4].strip().casefold())
    # Chinese users commonly shorten a declared phrase (for example,
    # ``深情祖师爷`` -> ``祖师爷``). Generate contiguous CJK fragments from the
    # metadata itself; this is not a service-specific keyword table.
    for match in re.finditer(r"[\u4e00-\u9fff]{2,}", raw):
        phrase = match.group(0)
        for start in range(len(phrase)):
            for end in range(start + 2, len(phrase) + 1):
                variants.add(phrase[start:end])
    return {
        item
        for item in variants
        if len(item) >= 2 and (any("\u4e00" <= char <= "\u9fff" for char in item) or len(item) >= 3)
    }


def _explicit_backend_hint(user_text: str) -> str:
    """Extract a generic backend name when a request explicitly names one.

    This intentionally knows only routing grammar ("use X MCP/Skill/service"),
    never concrete service names.  It lets us fail closed when a named backend
    is not loaded instead of silently exposing unrelated tools.
    """
    text = str(user_text or "").strip()
    pattern = re.compile(
        r"(?:用|使用|调用|切换到|选择|use|call|switch\s+to)\s*"
        r"([\w./:-]+(?:\s+[\w./:-]+){0,2})\s*"
        r"(mcp|skill|服务|后端)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return ""
    return f"{match.group(1).strip()} {match.group(2).strip()}".strip()


def _explicit_tool_scope(user_text: str, tools: list[BaseTool]) -> tuple[list[BaseTool], str]:
    """Restrict model-visible tools when the user explicitly names a backend.

    Binding every installed MCP at once leaves selection to fuzzy model
    matching.  A direct request naming one capability is an unambiguous routing
    instruction, so expose only that backend for the whole turn.  Unnamed
    requests keep the existing all-tools behavior.
    """
    text = str(user_text or "").casefold()
    groups: dict[str, list[BaseTool]] = {}
    aliases: dict[str, set[str]] = {}
    for tool in tools:
        metadata = getattr(tool, "metadata", None) or {}
        capability = str(metadata.get("capability") or "").strip()
        if not capability:
            # Generic fallback for tools created outside our loaders: group by
            # the MCP/Skill prefix, without naming any concrete service.
            name = str(tool.name)
            capability = name.split("_", 2)[1] if name.startswith("mcp_") else name
        groups.setdefault(capability, []).append(tool)
        candidates = metadata.get("aliases") or []
        alias_set = aliases.setdefault(capability, set())
        for candidate in [capability, str(tool.name), *candidates]:
            alias_set.update(_metadata_aliases(candidate))

    matches: list[tuple[int, str]] = []
    for capability, terms in aliases.items():
        for term in terms:
            normalized = term.strip().strip("[](){}<>「」『』\"'")
            if (
                len(normalized) >= 2
                and (any("\u4e00" <= char <= "\u9fff" for char in normalized) or len(normalized) >= 3)
                and normalized in text
            ):
                matches.append((len(normalized), capability))
    if matches:
        # Longest alias wins; ties are ambiguous. Explicit routing then fails
        # closed below so the model can ask for clarification instead of
        # guessing between capabilities.
        matches.sort(reverse=True)
        best_len = matches[0][0]
        best = {capability for length, capability in matches if length == best_len}
        if len(best) == 1:
            capability = next(iter(best))
            return groups.get(capability, []), capability
    # An explicit routing phrase with no matching loaded capability must fail
    # closed.  Otherwise a request for an unavailable backend could fall back
    # to an unrelated MCP merely because it is installed.
    hint = _explicit_backend_hint(user_text)
    if hint:
        return [], hint
    return tools, ""


def _routing_terms(value: Any) -> set[str]:
    """Build language-agnostic lexical terms from capability metadata."""
    text = str(value or "").casefold()
    terms = set(re.findall(r"[a-z0-9][a-z0-9_.-]{1,}", text))
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        # CJK has no whitespace tokenizer; overlapping bigrams/trigrams keep
        # this generic while allowing phrases such as ``中餐菜谱`` to match a
        # metadata description containing ``菜谱``.
        for size in (2, 3):
            terms.update(run[i : i + size] for i in range(len(run) - size + 1))
    return {term for term in terms if len(term) >= 2}


def _relevant_tool_subset(user_text: str, tools: list[BaseTool]) -> list[BaseTool]:
    """Retrieve the most relevant capability groups from tool metadata.

    This is deliberately service-agnostic. It is a lightweight local retrieval
    layer; an embedding or enterprise search index can replace the scoring
    implementation without changing loaders or routing contracts.
    """
    query_terms = _routing_terms(user_text)
    if not query_terms:
        return tools
    groups: dict[str, list[BaseTool]] = {}
    profiles: dict[str, set[str]] = {}
    for tool in tools:
        metadata = getattr(tool, "metadata", None) or {}
        capability = str(metadata.get("capability") or "").strip()
        if not capability:
            capability = str(tool.name)
        groups.setdefault(capability, []).append(tool)
        profile = profiles.setdefault(capability, set())
        profile.update(_routing_terms(capability))
        profile.update(_routing_terms(tool.name))
        profile.update(_routing_terms(getattr(tool, "description", "")))
        for alias in metadata.get("aliases") or []:
            profile.update(_routing_terms(alias))

    scores = {
        capability: len(query_terms & profile)
        for capability, profile in profiles.items()
    }
    best_score = max(scores.values(), default=0)
    if best_score <= 0:
        return tools

    selected: list[BaseTool] = []
    for capability, group in groups.items():
        # Built-ins without capability metadata remain available for ordinary
        # local actions; external capability groups are narrowed by relevance.
        has_external_metadata = any(
            (getattr(tool, "metadata", None) or {}).get("kind") in {"mcp", "skill"}
            for tool in group
        )
        if not has_external_metadata or scores.get(capability, 0) == best_score:
            selected.extend(group)
    return selected or tools


def _route_observation(user_text: str, tools: list[BaseTool], selected: list[BaseTool]) -> dict[str, Any]:
    """Return audit-safe evidence for a routing decision."""
    query_terms = _routing_terms(user_text)
    groups: dict[str, set[str]] = {}
    for tool in tools:
        metadata = getattr(tool, "metadata", None) or {}
        capability = str(metadata.get("capability") or tool.name)
        profile = groups.setdefault(capability, set())
        profile.update(_routing_terms(capability))
        profile.update(_routing_terms(tool.name))
        profile.update(_routing_terms(getattr(tool, "description", "")))
        for alias in metadata.get("aliases") or []:
            profile.update(_routing_terms(alias))
    scores = {
        capability: len(query_terms & profile)
        for capability, profile in groups.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best = ranked[0][1] if ranked else 0
    second = ranked[1][1] if len(ranked) > 1 else 0
    selected_capabilities = sorted(
        {
            str((getattr(tool, "metadata", None) or {}).get("capability") or tool.name)
            for tool in selected
        }
    )
    return {
        "query_terms": sorted(query_terms),
        "candidates": [{"capability": cap, "score": score} for cap, score in ranked[:8]],
        "selected_capabilities": selected_capabilities,
        "best_score": best,
        "score_margin": best - second,
        "confidence": round(best / max(best + second, 1), 3),
    }


def _tool_scope_for_turn(user_text: str, tools: list[BaseTool]) -> tuple[list[BaseTool], str]:
    """Apply explicit routing first, then metadata retrieval for implicit asks."""
    scoped, label = _explicit_tool_scope(user_text, tools)
    if label:
        return scoped, label
    return _relevant_tool_subset(user_text, tools), ""


def _latest_user_turn(messages: list[Any]) -> list[Any]:
    """Return only the messages belonging to the most recent user turn."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _install_call_from_user_request(messages: list[Any], user_text: str) -> Optional[dict[str, Any]]:
    """Build a structured install call from an explicit GitHub install request."""
    text = str(user_text or "")
    if not (("安装" in text or "install" in text.casefold()) and ("github.com/" in text.casefold())):
        return None
    # A short "继续安装" message can refer to the URL in the preceding turn.
    search_text = "\n".join(str(getattr(message, "content", "")) for message in messages[-12:])
    search_text += "\n" + text
    match = re.search(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", search_text, re.IGNORECASE)
    if not match:
        return None
    source = match.group(0).rstrip(".,;:)]}")
    repo = source.rstrip("/").rsplit("/", 1)[-1]
    if repo.casefold().endswith(".git"):
        repo = repo[:-4]
    name = re.sub(r"[^A-Za-z0-9_-]+", "-", repo).strip("-")
    if not name:
        return None
    return {
        "name": "install_mcp_server",
        "args": {"source_type": "git", "source": source, "name": name},
        "id": "install_" + name.casefold().replace("-", "_") + "_request",
    }


def _parse_plan(raw: str) -> list[dict[str, Any]]:
    """Extract plan steps from LLM output; tolerate markdown fences."""
    text = raw.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            steps = [str(s).strip() for s in data if str(s).strip()]
            if steps:
                return [make_plan_step(i + 1, s) for i, s in enumerate(steps[:8])]
            if isinstance(data, list):
                return []
        except (json.JSONDecodeError, TypeError):
            pass
    steps = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)、])\s*", "", line).strip()
        if line and not line.startswith("```"):
            steps.append(line)
    return [make_plan_step(i + 1, s) for i, s in enumerate(steps[:8])]


def _build_planner_prompt(goal: str) -> str:
    """Build the planner prompt shared by runtime code and live evaluation."""
    return (
        "You are the planner. Split the user goal into 2-6 executable steps.\n"
        f"Goal: {goal}\n\n"
        "Rules:\n"
        "1. Output ONLY a JSON string array\n"
        "2. One action per step, no numbering, no explanation\n"
        "3. Do not invent steps the user did not ask for"
    )


def create_agent_app(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    tools: Optional[List[BaseTool]] = None,
    checkpointer=None,
    require_approval: bool = True,
    approval_tools: Optional[set] = None,
    enable_planner: bool = True,
    enable_replan: bool = True,
    budget: Optional[RunBudget] = None,
    policy: Optional[PolicyConfig] = None,
    hard_deny_critical: bool = True,
    permission_manager=None,
    context_trigger_turns: int = CONTEXT_TRIGGER_TURNS,
    context_keep_turns: int = CONTEXT_KEEP_TURNS,
):
    """创建并编译 Agent 状态图应用。

    搭建 LangGraph StateGraph 工作流：
      initialize -> (planner) -> agent -> approval/tools/policy_block -> progress -> replan
    形成思考-调工具-看结果-再思考的循环。所有节点闭包共享 llm/tools/policy 配置。

    Args:
        provider_name: LLM 提供商名称，默认 "openai"。
        model_name: 模型名称，默认 "gpt-4o-mini"。
        tools: 工具列表，None 则自动加载内置工具 + 动态技能。
        checkpointer: LangGraph checkpointer（同步用 MemorySaver，异步用 AsyncSqliteSaver）。
        require_approval: 是否启用人工审批闸门（副作用工具执行前中断）。
        approval_tools: 需要审批的工具名集合，None 则用策略引擎默认判定。
        enable_planner: 是否启用 planner 节点（复杂任务自动拆步骤）。
        enable_replan: 是否启用失败后重规划。
        budget: 运行时预算（LLM 调用次数/工具轮次/重规划次数/墙钟时间）。
        policy: 风险策略配置，None 则用默认策略。
        hard_deny_critical: 是否对 critical 级别工具调用硬拒绝。
        permission_manager: 可选权限管理器，提供持续授权但不绕过 critical 硬拒绝。
        context_trigger_turns: 上下文裁剪高水位，默认 20 个用户回合。
        context_keep_turns: 裁剪后的低水位，默认保留最近 10 个用户回合。

    Returns:
        编译后的 LangGraph 应用，支持 astream/aget_state/aupdate_state。
    """
    if context_trigger_turns <= 0 or context_keep_turns <= 0:
        raise ValueError("context turn thresholds must be positive")
    if context_keep_turns >= context_trigger_turns:
        raise ValueError(
            "context_keep_turns must be smaller than context_trigger_turns")

    if tools is None:
        dynamic_tools = load_dynamic_skills()
        actual_tools = BUILTIN_TOOLS + dynamic_tools
    else:
        actual_tools = tools

    run_budget = budget or DEFAULT_BUDGET
    policy_cfg = policy or build_policy_config(
        approval_tools=approval_tools,
        hard_deny_critical=hard_deny_critical,
    )

    tool_node = ToolNode(actual_tools)

    # LangGraph 的固定边不会经过路由函数，单独维护每个会话的上一节点，
    # 让审计日志可以完整还原 START、条件分支、固定边和 END。
    transition_lock = threading.Lock()
    last_graph_nodes: dict[str, str] = {}
    approval_wait_started_at: dict[str, float] = {}

    def _thread_id(config: RunnableConfig | None) -> str:
        return (config or {}).get("configurable", {}).get(
            "thread_id", "system_default"
        )

    def _record_transition(
        state: AgentState,
        config: RunnableConfig | None,
        from_node: str | None,
        to_node: str,
    ) -> None:
        """记录图的转换。"""
        thread_id = _thread_id(config)
        with transition_lock:
            previous = last_graph_nodes.get(thread_id)
            # 进程重启后内存中的上一节点可能不存在；此时不要把恢复运行误报成 START。
            source = from_node or previous or (START if to_node == "initialize" else "unknown")
            source_label = "START" if source == START else "END" if source == END else source
            target_label = "END" if to_node == END else to_node
            audit_logger.log_event(
                thread_id=thread_id,
                event="graph_transition",
                from_node=source_label,
                to_node=target_label,
                task_status=state.get("task_status", ""),
                step_index=state.get("step_index", 0),
                retry_count=state.get("retry_count", 0),
                replan_count=state.get("replan_count", 0),
            )
            print(
                f"[Graph] {source_label} -> {target_label}"
                f" | status={state.get('task_status', '') or '-'}"
                f" step={state.get('step_index', 0)}",
                file=sys.stderr,
                flush=True,
            )
            if to_node == END:
                last_graph_nodes.pop(thread_id, None)
            else:
                last_graph_nodes[thread_id] = to_node
            if to_node == "approval":
                approval_wait_started_at[thread_id] = time.time()

    def _instrument_node(name: str, node_fn):
        """记录固定边进入节点；条件边已在路由函数中记录。"""
        def wrapped(state: AgentState, config: RunnableConfig | None = None):
            with transition_lock:
                previous = last_graph_nodes.get(_thread_id(config))
            if previous != name:
                _record_transition(state, config, previous, name)
            return node_fn(state, config)

        return wrapped

    def tools_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        """执行工具，并兼容旧 checkpoint 中的 ``v__args`` 参数名。

        旧版 ``install_mcp_server(args=...)`` 曾被 LangChain 生成成 ``v__args``。
        新调用统一使用 ``launch_args``，但已持久化的审批仍可能携带旧字段，
        因此在真正执行前做一次窄范围迁移。
        """
        messages = list(state.get("messages") or [])
        if messages:
            last = messages[-1]
            calls = getattr(last, "tool_calls", None) or []
            normalized_calls = []
            changed = False
            for call in calls:
                args = dict(call.get("args") or {})
                if (
                    call.get("name") == "install_mcp_server"
                    and "v__args" in args
                    and "launch_args" not in args
                ):
                    args["launch_args"] = args.pop("v__args")
                    normalized = dict(call)
                    normalized["args"] = args
                    normalized_calls.append(normalized)
                    changed = True
                else:
                    normalized_calls.append(call)
            if changed and hasattr(last, "model_copy"):
                messages[-1] = last.model_copy(
                    update={"tool_calls": normalized_calls})
            elif changed:
                last.tool_calls = normalized_calls
        return tool_node.invoke({"messages": messages})

    def approval_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        """标记审批闸门；真正的暂停由 interrupt_before=approval 完成。"""
        thread_id = _thread_id(config)
        with transition_lock:
            wait_started = approval_wait_started_at.pop(thread_id, None)
        updates = {
            "approval_interrupts": int(state.get("approval_interrupts") or 0) + 1,
            "task_status": "waiting_approval",
        }
        # 人工阅读和输入 y/s/a/n 的时间不属于 Agent 执行耗时，否则用户稍晚
        # 批准就会在工具返回后被 300 秒墙钟预算误杀。
        run_started_at = float(state.get("run_started_at") or 0.0)
        if wait_started and run_started_at:
            updates["run_started_at"] = run_started_at + max(
                0.0, time.time() - wait_started
            )
        return updates
    llm = get_provider(provider_name=provider_name, model_name=model_name)

    def initialize_run_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
        """在新一轮用户请求开始时重置运行状态。taskplan/retrycount等记录都清零"""
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], HumanMessage):
            return {}
        return {
            "current_task": "",
            "task_plan": [],
            "step_index": 0,
            "task_status": TASK_PENDING,
            "retry_count": 0,
            "replan_count": 0,
            "last_failure_class": "",
            "last_failure_note": "",
            "llm_calls": 0,
            "tool_rounds": 0,
            "run_started_at": 0.0,
            "budget_exceeded_reason": "",
            "last_policy_summary": "",
            "approval_interrupts": 0,
            "hard_deny_count": 0,
            "context_trim_count": 0,
            "context_turns_before_trim": 0,
            "context_turns_kept": 0,
            "context_turns_discarded": 0,
            "context_chars_before_trim": 0,
            "context_chars_after_trim": 0,
            "summary_llm_calls": 0,
            "last_run_summary": {},
        }

    def _budget_fail_update(state: AgentState, reason: str) -> dict:
        """构造预算超限时的状态与当前计划更新。"""
        updates = apply_budget_fail(reason)
        task_plan = state.get("task_plan") or []
        if task_plan:
            plan, idx, _ = advance_plan(
                task_plan, state.get("step_index", 0), STEP_FAILED, reason
            )
            updates.update({"task_plan": plan, "step_index": idx})
        updates["last_run_summary"] = build_run_summary({**state, **updates})
        return updates

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        """调用 LLM 处理当前状态，并生成回复或工具调用。"""
        thread_id = config.get("configurable", {}).get(
            "thread_id", "system_default")
        state_updates: dict[str, Any] = {}
        state_updates.update(init_run_clock(state))
        merged = {**state, **state_updates}
        reason = budget_violation(merged, run_budget)  # 预算熔断，如果已经超了预算就不调LLM了
        if reason:
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="budget_exceeded",
                reason=reason,
            )
            state_updates.update(_budget_fail_update(merged, reason))
            return state_updates  # 不往下走了

        raw_messages = state["messages"]
        last_human = next(
            (m for m in reversed(raw_messages) if isinstance(m, HumanMessage)),
            None,
        )  # 获取最近的用户消息,避免工具调用或AI消息干扰
        scoped_tools, scope_label = _tool_scope_for_turn(
            str(last_human.content) if last_human else "", actual_tools
        )
        route_observation = _route_observation(
            str(last_human.content) if last_human else "", actual_tools, scoped_tools
        )
        audit_logger.log_event(
            thread_id=thread_id,
            event="tool_route_decision",
            explicit=bool(scope_label),
            scope_label=scope_label,
            **route_observation,
        )
        if raw_messages:
            recent_tool_msgs = []
            # 收集最近的工具调用消息
            for msg in reversed(raw_messages):
                if msg.type == "tool":
                    recent_tool_msgs.append(msg)
                else:
                    break

            # 审计日志：记录最近的工具调用结果
            for msg in reversed(recent_tool_msgs):
                audit_logger.log_event(
                    thread_id=thread_id,
                    event="tool_result",
                    tool=msg.name,
                    result_summary=str(msg.content)[:200],
                )

        # 对话达到高水位后裁到低水位；旧回合压成摘要并用 RemoveMessage 删除。
        current_summary = state.get("summary", "")
        if scope_label or len(scoped_tools) < len(actual_tools):
            # Explicit backend switches start a fresh task context. Keeping the old backend's
            # ToolMessages, or an inferred switch to a different capability,
            # encourages the model to repeat stale content or continue a failed
            # workflow. The current turn still retains its own ToolMessage.
            final_msgs, discarded_msgs = _latest_user_turn(raw_messages), []
        else:
            final_msgs, discarded_msgs = trim_context_messages(
                raw_messages,
                trigger_turns=context_trigger_turns,
                keep_turns=context_keep_turns,
            )
        # 兼容旧会话：工具异常中断后，补齐未配对的 tool_calls，避免下一次
        # 请求被 OpenAI 兼容接口以消息协议 400 拒绝。
        final_msgs = repair_unpaired_tool_calls(final_msgs)

        # Reserve every LLM call before making it. Context compression and the
        # main answer are two separate calls, so both must fit in the budget.
        reason = budget_violation(
            {**state, **state_updates},
            run_budget,
            pending_llm_calls=2 if discarded_msgs else 1,
        )
        if reason:
            state_updates.update(_budget_fail_update(
                {**state, **state_updates}, reason))
            return state_updates

        if discarded_msgs:
            before_metrics = measure_context(raw_messages)
            kept_metrics = measure_context(final_msgs)
            discarded_metrics = measure_context(discarded_msgs)
            state_updates.update({
                "context_trim_count": int(state.get("context_trim_count") or 0) + 1,  # 裁剪次数
                "context_turns_before_trim": before_metrics["turns"],
                "context_turns_kept": kept_metrics["turns"],
                "context_turns_discarded": discarded_metrics["turns"],
                "context_chars_before_trim": before_metrics["characters"],
                "context_chars_after_trim": kept_metrics["characters"],
                "summary_llm_calls": int(state.get("summary_llm_calls") or 0) + 1,
            })
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="context_trimmed",
                trigger_turns=context_trigger_turns,
                keep_turns=context_keep_turns,
                turns_before=before_metrics["turns"],
                turns_kept=kept_metrics["turns"],
                turns_discarded=discarded_metrics["turns"],
                chars_before=before_metrics["characters"],
                chars_after=kept_metrics["characters"],
            )
            summary_prompt = (
                f"[existing summary]\n{current_summary if current_summary else 'none'}\n\n"
                "[discarded turns]\n"
                + "\n".join(
                    f"- {getattr(m, 'type', '?')}: {str(getattr(m, 'content', ''))[:200]}"
                    for m in discarded_msgs
                )
                + "\n\nCompress into a concise recent summary. Keep facts, preferences, open items."
            )  # 压缩上下文的prompt,把旧对话压缩成摘要,减少上下文长度,保留事实、偏好、未完成事项
            state_updates.update(bump_llm_calls({**state, **state_updates}, 1))
            new_summary_response = llm.invoke(
                [HumanMessage(content=summary_prompt)], config={"callbacks": []}
            )
            active_summary = new_summary_response.content
            state_updates["summary"] = active_summary
            state_updates.setdefault("messages", [])

            for old in discarded_msgs:
                if getattr(old, "id", None):
                    # 用 RemoveMessage 删除旧消息，减少上下文长度
                    state_updates["messages"].append(RemoveMessage(id=old.id))
        else:
            active_summary = current_summary

        # 读取用户画像（user_profile.md），如果不存在则默认 "none"，所有会话共享、长期存在、低频修改
        profile_path = os.path.join(MEMORY_DIR, "user_profile.md")
        profile_content = "none"
        if os.path.exists(profile_path):
            try:
                with open(profile_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                    if content:
                        profile_content = content
            except OSError:
                pass

        # 构建系统提示，包含角色设定是fluxagent、用户画像（从user_profile.md读取）、摘要、当前任务、计划步骤、上次失败信息
        sys_prompt = (
            "你是FluxAgent——一个透明且可控制的本地智能体。\n"
            "Rules:\n"
            "1. Stay inside the office sandbox for files/shell.\n"
            "2. For unfamiliar skills: mode=help first, then mode=run.\n"
            "3. When user states long-term preferences, call save_user_profile.\n"
            "4. Installed MCP tools use names beginning with mcp_. Prefer a matching MCP tool for capabilities not covered by built-ins. "
            "After installing a new MCP server, tell the user that FluxAgent must restart to discover it.\n"
            "5. To install an MCP, call install_mcp_server with complete arguments. The installer synchronously downloads/copies and registers the source only;"
            " it automatically detects Node/Python, prepares dependencies, builds, and "
            "health-checks before registry commit; it runs synchronously, not in the background.\n"
            "6. Never claim background progress, monitoring, time estimates, completion, or a tool call unless structured tool_calls and ToolMessage results exist.\n"
            "7. Never claim a file was created, written, saved, deleted, or a tool was called unless the corresponding structured tool_call and ToolMessage result are present"
            "8. User profile:\n"
            f"{profile_content}\n"
        )
        if scope_label:
            names = ", ".join(str(tool.name) for tool in scoped_tools)
            sys_prompt += (
                f"\n\n[Explicit tool routing]\n"
                f"用户明确指定使用 {scope_label}。"
                + (f"本轮只能调用以下工具：{names}。" if names else "该服务当前未加载。")
                + "不要调用其他 MCP、Skill 或内置工具；如果指定服务不可用，直接说明原因。\n"
            )
        if active_summary:
            # 添加最近的上下文摘要
            sys_prompt += f"\n\n[Recent context]\n{active_summary}\n"

        if state.get("last_failure_class") == "model_protocol":
            sys_prompt += (
                "\n\n[Protocol correction]\n"
                "上一条回复声称执行了操作，但没有结构化 tool_calls，实际并未执行。"
                "本轮必须通过真实工具调用完成操作；在收到 ToolMessage 结果前，"
                "不得声称文件已创建、已写入或工具已执行。\n"
            )

        task_plan = state.get("task_plan") or []
        if task_plan:
            step_index = state.get("step_index", 0)
            sys_prompt += (
                f"\n\n[Current task]\n{state.get('current_task', '')}\n"
                f"[Plan]\n{render_plan(task_plan, step_index)}\n"
                "(Only advance the current step.)"
            )  # 构建系统提示，包含当前任务和计划步骤

            fail_note = state.get("last_failure_note") or ""
            if fail_note:
                sys_prompt += (
                    f"\n[Last failure] class={state.get('last_failure_class', '')}; "
                    f"note={fail_note}\n"
                    "Use an alternative approach; do not repeat the same bad args."
                )  # 包含上次失败信息，提醒不要重复同样的错误参数

        msgs_for_llm = [SystemMessage(content=sys_prompt)] + [
            m for m in final_msgs if not isinstance(m, SystemMessage)
        ]
        for m in msgs_for_llm:
            if isinstance(m.content, str):
                m.content = m.content.encode("utf-8", "ignore").decode("utf-8")

        audit_logger.log_event(
            thread_id=thread_id, event="llm_input", message_count=len(msgs_for_llm)
        )

        # 把拼好的 prompt + 裁好的 messages 喂给 LLM
        state_updates.update(bump_llm_calls({**state, **state_updates}, 1))
        # Bind the narrowed set for this turn.  This is a hard capability
        # boundary for providers that honor tool schemas, rather than a soft
        # "prefer" instruction competing with unrelated MCP descriptions.
        response = llm.bind_tools(scoped_tools).invoke(msgs_for_llm)
        unverified_claim = _has_unverified_action_claim(response.content)
        if not response.tool_calls:
            recovered = _install_call_from_user_request(
                raw_messages, str(last_human.content) if last_human else ""
            )
            if recovered:
                response = AIMessage(content="", tool_calls=[recovered])
        if not response.tool_calls and unverified_claim:
            response = AIMessage(
                content=(
                    "未检测到真实的 MCP 工具调用，因此没有启动后台安装。"
                    "请提供完整安装参数，并等待 install_mcp_server 返回结果。"
                )
            )
        # LLM 的回复 append 回 state（messages）

        if response.tool_calls:
            for tool_call in response.tool_calls:
                audit_logger.log_event(
                    thread_id=thread_id,
                    event="tool_call",
                    tool=tool_call["name"],
                    args=tool_call["args"],
                )
        elif response.content:
            audit_logger.log_event(
                thread_id=thread_id, event="ai_message", content=response.content
            )

        state_updates.setdefault("messages", [])
        state_updates["messages"].append(response)

        if task_plan and response.tool_calls and state.get("task_status") == TASK_DONE:
            # 如果 LLM 调用工具后，任务状态仍然是已完成，说明 LLM 还在思考中，任务状态保持运行中
            # 解决了一个问题：LLM 在计划步骤 1/N 时调用工具，返回结果后继续思考，任务状态不应该被标记为已完成。
            state_updates["task_status"] = TASK_RUNNING

        if not response.tool_calls:
            if task_plan:
                # 如果有计划步骤但没有工具调用，说明 LLM 还在思考中，任务状态保持运行中
                # 文本可完成仅包含推理步骤的计划步骤。将该路线纳入进度管理，以防止未完成的计划在第 1/N 个步骤时悄然终止。
                state_updates["task_status"] = TASK_RUNNING
            else:
                # 如果没有计划步骤也没有工具调用，说明 LLM 已经完成任务，标记为已完成
                state_updates["task_status"] = TASK_DONE
                state_updates["last_run_summary"] = build_run_summary(
                    {**state, **state_updates, "task_status": TASK_DONE}
                )

        # A model can narrate a tool call instead of returning structured
        # tool_calls. Never treat that narration as a completed side effect.
        # Once this protocol error is detected, a later plain-text response is
        # also not allowed to clear it; only a real ToolMessage can do that.
        protocol_retry = state.get("last_failure_class") == "model_protocol"
        if not response.tool_calls and (unverified_claim or protocol_retry):
            state_updates["task_status"] = TASK_RUNNING
            state_updates["last_failure_class"] = "model_protocol"
            state_updates["last_failure_note"] = (
                "模型声称已执行文件或工具操作，但响应中没有结构化 tool_calls，"
                "操作尚未发生。请直接发起所需工具调用，不要用文字代替。"
            )
            next_retry = int(state.get("retry_count") or 0) + 1
            state_updates["retry_count"] = next_retry
            if next_retry >= 2:
                state_updates["task_status"] = TASK_FAILED
                state_updates["last_run_summary"] = build_run_summary(
                    {**state, **state_updates, "task_status": TASK_FAILED}
                )
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="unverified_action_claim",
                failure_class="model_protocol",
            )

        merged_after = {**state, **state_updates}
        busted_after = budget_violation(
            merged_after,
            run_budget,
            pending_tool_rounds=1 if response.tool_calls else 0,
        )
        if busted_after:
            state_updates.update(_budget_fail_update(
                merged_after, busted_after))
            if not response.tool_calls:
                state_updates["last_run_summary"] = build_run_summary(
                    {**state, **state_updates}
                )
        return state_updates

    def planner_node(state: AgentState, config: RunnableConfig) -> dict:
        """将需要规划的用户目标拆分为可执行步骤。"""
        thread_id = config.get("configurable", {}).get(
            "thread_id", "system_default")
        updates: dict[str, Any] = {}
        updates.update(init_run_clock(state))
        reason = budget_violation(
            {**state, **updates}, run_budget, pending_llm_calls=1
        )
        if reason:
            return {**updates, **_budget_fail_update({**state, **updates}, reason)}
        last_human = next(
            (m for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)), None
        )
        goal = last_human.content if last_human else ""
        prompt = _build_planner_prompt(str(goal))
        updates.update(bump_llm_calls({**state, **updates}, 1))
        try:
            resp = llm.invoke([HumanMessage(content=prompt)],
                              config={"callbacks": []})
            plan = _parse_plan(resp.content)
        except Exception as exc:
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="plan_failed",
                error=str(exc)[:200],
            )
            updates.update(
                {
                    "current_task": goal[:200],
                    "task_plan": [],
                    "task_status": TASK_FAILED,
                    "last_failure_class": "planner",
                    "last_failure_note": f"planner_model_error:{exc}"[:300],
                }
            )
            updates["last_run_summary"] = build_run_summary(
                {**state, **updates})
            return updates

        if not plan:
            plan = [make_plan_step(1, goal[:80] or "handle request")]
        plan[0]["status"] = STEP_RUNNING
        audit_logger.log_event(
            thread_id=thread_id,
            event="system_action",
            action="plan_created",
            task=goal[:100],
            steps=[s["action"] for s in plan],
        )
        updates.update(
            {
                "current_task": goal[:200],
                "task_plan": plan,
                "step_index": 0,
                "task_status": TASK_RUNNING,
                "retry_count": 0,
                "replan_count": 0,
                "last_failure_class": "",
                "last_failure_note": "",
                "budget_exceeded_reason": "",
            }
        )
        return updates

    def replan_node(state: AgentState, config: RunnableConfig) -> dict:
        """根据最近失败生成替代计划并替换原计划。"""
        thread_id = config.get("configurable", {}).get(
            "thread_id", "system_default")
        updates: dict[str, Any] = {}
        updates.update(init_run_clock(state))
        reason = budget_violation(
            {**state, **updates}, run_budget, pending_llm_calls=1
        )
        if reason:
            return {**updates, **_budget_fail_update({**state, **updates}, reason)}
        goal = state.get("current_task") or ""
        if not goal:
            last_human = next(
                (m for m in reversed(state["messages"])
                 if isinstance(m, HumanMessage)),
                None,
            )
            goal = last_human.content if last_human else "continue task"
        failure_note = state.get("last_failure_note") or "previous step failed"
        old_plan = state.get("task_plan") or []
        prompt = build_replan_prompt(goal, old_plan, failure_note, failure_class=state.get(
            "last_failure_class", ""), step_index=state.get("step_index", 0))
        updates.update(bump_llm_calls({**state, **updates}, 1))
        try:
            resp = llm.invoke([HumanMessage(content=prompt)],
                              config={"callbacks": []})
            plan_actions = [s.get("action", "")
                            for s in _parse_plan(resp.content)]
            forbidden = failed_step_action(
                old_plan, state.get("step_index", 0))
            ok, plan_actions = validate_replan_steps(
                plan_actions, forbidden_action=forbidden)
            plan = [make_plan_step(i + 1, a)
                    for i, a in enumerate(plan_actions)] if ok else []
        except Exception as exc:
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="replan_failed",
                error=str(exc)[:200],
            )
            updates.update(
                {
                    "task_status": TASK_FAILED,
                    "last_failure_class": "replan",
                    "last_failure_note": f"replan_model_error:{exc}"[:300],
                }
            )
            updates["last_run_summary"] = build_run_summary(
                {**state, **updates})
            return updates
        if not plan:
            plan = [
                make_plan_step(i + 1, step)
                for i, step in enumerate(fallback_replan_steps(goal, failure_note, failed_step_action(old_plan, state.get("step_index", 0))))
            ]
        plan[0]["status"] = STEP_RUNNING
        audit_logger.log_event(
            thread_id=thread_id,
            event="system_action",
            action="plan_replanned",
            replan_count=state.get("replan_count", 0),
            old_steps=[s.get("action") for s in (old_plan or [])],
            new_steps=[s["action"] for s in plan],
            steps=[s["action"] for s in plan],
            failure_class=state.get("last_failure_class", ""),
        )
        updates.update(
            {
                "current_task": goal[:200],
                "task_plan": plan,
                "step_index": 0,
                "task_status": TASK_RUNNING,
                "retry_count": 0,
            }
        )
        merged = {**state, **updates}
        reason = budget_violation(merged, run_budget)
        if reason:
            updates.update(_budget_fail_update(merged, reason))
        return updates

    def progress_node(state: AgentState, config: RunnableConfig) -> dict:
        """检查工具执行结果并更新任务进度或恢复动作。"""
        thread_id = config.get("configurable", {}).get(
            "thread_id", "system_default")
        updates: dict[str, Any] = {}
        if state.get("task_status") == TASK_CANCELLED:
            updates["last_run_summary"] = build_run_summary(state)
            return updates

        task_plan = state.get("task_plan") or []
        texts = extract_tool_failure_texts(state.get("messages") or [])
        if not texts:
            last_message = (state.get("messages") or [None])[-1]
            if (
                task_plan
                and getattr(last_message, "type", None) == "ai"
                and not (getattr(last_message, "tool_calls", None) or [])
            ):
                plan, idx, status = advance_plan(
                    task_plan, state.get("step_index", 0), STEP_DONE
                )
                updates.update(
                    {
                        "task_plan": plan,
                        "step_index": idx,
                        "task_status": status,
                        "retry_count": 0,
                        "last_failure_class": "",
                        "last_failure_note": "",
                    }
                )
                audit_logger.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    action="step_done",
                    step_index=idx,
                    task_status=status,
                    source="agent_text",
                )
                if status == TASK_DONE:
                    updates["last_run_summary"] = build_run_summary({**state, **updates})
                return updates
            install_result = str(getattr(last_message, "content", "") or "")
            install_success = (
                getattr(last_message, "type", None) == "tool"
                and getattr(last_message, "name", None) in TERMINAL_INSTALL_TOOLS
                and ("安装成功" in install_result or ("已安装" in install_result and "登记" in install_result))
                and not install_result.startswith("❌")
            )
            if not install_success:
                return updates

        updates.update(bump_tool_rounds(state, 1))
        merged = {**state, **updates}
        reason = budget_violation(merged, run_budget)
        if reason:
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="budget_exceeded",
                reason=reason,
            )
            updates.update(_budget_fail_update(merged, reason))
            return updates

        task_plan = state.get("task_plan") or []
        # 安装类工具是一个完整的“安装步骤”事务：下载/复制、校验、登记由受控安装器完成。只有当当前计划没有后续步骤时，安装成功才等价于整个任务完成
        # 计划中仍有验证、配置或回答步骤时必须继续推进。
        last_tool = next(
            (
                message
                for message in reversed(state.get("messages") or [])
                if isinstance(message, ToolMessage)
            ),
            None,
        )
        if last_tool:
            last_human = next(
                (
                    message
                    for message in reversed(state.get("messages") or [])
                    if isinstance(message, HumanMessage)
                ),
                None,
            )
            routed_tools, routed_label = _tool_scope_for_turn(
                str(last_human.content) if last_human else "", actual_tools
            )
            allowed_names = {str(tool.name) for tool in routed_tools}
            if str(last_tool.name) not in allowed_names:
                audit_logger.log_event(
                    thread_id=thread_id,
                    event="tool_selection_mismatch",
                    tool=str(last_tool.name),
                    scope_label=routed_label,
                    selected_tools=sorted(allowed_names),
                    result_status="success" if not str(last_tool.content or "").startswith("❌") else "error",
                )
        if last_tool and last_tool.name in TERMINAL_INSTALL_TOOLS:
            result_text = str(last_tool.content or "")
            if (
                ("安装成功" in result_text or ("已安装" in result_text and "登记" in result_text))
                and not result_text.startswith("❌")
                and not task_plan
            ):
                audit_logger.log_event(
                    thread_id=thread_id,
                    event="system_action",
                    action="installation_completed",
                    tool=last_tool.name,
                )
                updates.update(
                    {
                        "task_plan": [],
                        "step_index": 0,
                        "task_status": TASK_DONE,
                        "retry_count": 0,
                        "last_failure_class": "",
                        "last_failure_note": "",
                    }
                )
                updates["last_run_summary"] = build_run_summary(
                    {**state, **updates}
                )
                return updates

        if not any_tool_failure(texts):
            if not task_plan:
                # A successful tool call still needs a user-facing synthesis
                # pass. Without this, no-plan Skill/MCP requests terminate
                # immediately after ToolMessage and can leave the next turn
                # looking like a delayed or mixed response. Installation is
                # handled above as a true terminal transaction when no plan
                # remains.
                if isinstance(last_tool, ToolMessage):
                    updates.update(
                        {
                            "task_status": TASK_RUNNING,
                            "retry_count": 0,
                            "last_failure_class": "",
                            "last_failure_note": "",
                        }
                    )
                    return updates
                updates.update(
                    {
                        "retry_count": 0,
                        "last_failure_class": "",
                        "last_failure_note": "",
                    }
                )
                return updates

            last_message = (state.get("messages") or [None])[-1]
            tool_result_needs_answer = isinstance(last_message, ToolMessage)
            plan, idx, status = advance_plan(
                task_plan, state.get("step_index", 0), STEP_DONE
            )
            if tool_result_needs_answer and status == TASK_DONE:
                status = TASK_RUNNING
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="step_done",
                step_index=idx,
                task_status=status,
            )
            updates.update(
                {
                    "task_plan": plan,
                    "step_index": idx,
                    "task_status": status,
                    "retry_count": 0,
                    "last_failure_class": "",
                    "last_failure_note": "",
                }
            )
            if status == TASK_DONE:
                updates["last_run_summary"] = build_run_summary(
                    {**state, **updates})
            return updates

        decision = classify_failure(
            texts,
            retry_count=int(state.get("retry_count") or 0),
            max_retries=run_budget.max_retries_per_step,
            replan_count=int(state.get("replan_count") or 0),
            max_replans=run_budget.max_replans,
            enable_replan=enable_replan,
        )
        updates.update(
            {
                "retry_count": decision.retry_count,
                "replan_count": decision.replan_count,
                "last_failure_class": decision.failure_class,
                "last_failure_note": decision.note[:300],
            }
        )

        if decision.action == ACTION_RETRY:
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="step_retry",
                retry_count=decision.retry_count,
                failure_class=decision.failure_class,
            )
            updates["task_status"] = TASK_RUNNING
            return updates

        if decision.action == ACTION_REPLAN:
            audit_logger.log_event(
                thread_id=thread_id,
                event="system_action",
                action="replan_scheduled",
                replan_count=decision.replan_count,
                failure_class=decision.failure_class,
            )
            if task_plan:
                plan, idx, _ = advance_plan(
                    task_plan,
                    state.get("step_index", 0),
                    STEP_FAILED,
                    decision.note[:120],
                )
                updates.update({"task_plan": plan, "step_index": idx})
            updates["task_status"] = TASK_REPLANNING
            return updates

        if task_plan:
            plan, idx, _ = advance_plan(
                task_plan,
                state.get("step_index", 0),
                STEP_FAILED,
                decision.note[:120],
            )
            updates.update({"task_plan": plan, "step_index": idx})
        audit_logger.log_event(
            thread_id=thread_id,
            event="system_action",
            action="task_aborted",
            retry_count=decision.retry_count,
            replan_count=decision.replan_count,
            failure_class=decision.failure_class,
        )
        updates["task_status"] = TASK_FAILED
        updates["last_run_summary"] = build_run_summary({**state, **updates})
        return updates

    def policy_block_node(state: AgentState, config: RunnableConfig) -> dict:
        """拦截策略判定为高风险的工具调用并记录失败。"""
        thread_id = config.get("configurable", {}).get( "thread_id", "system_default")
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        decision = assess_tool_calls(
            tool_calls, config=policy_cfg, require_approval=require_approval
        )
        note = POLICY_DENY_NOTE
        last_human = next(
            (m for m in reversed(state.get("messages") or []) if isinstance(m, HumanMessage)),
            None,
        )
        scoped_tools, scope_label = _tool_scope_for_turn(
            str(last_human.content) if last_human else "", actual_tools
        )
        allowed_names = {str(tool.name) for tool in scoped_tools}
        if any(tc.get("name") not in allowed_names for tc in tool_calls):
            note = (
                f"工具路由拒绝：本轮已根据用户请求选择相关能力{scope_label or '集合'}，"
                "不允许调用未入选的工具。请改用相关工具或直接回答用户。"
            )
        if decision.reasons:
            note = note + " reasons: " + "; ".join(decision.reasons[:4])
        tool_msgs = [
            ToolMessage(content=note, tool_call_id=tc["id"], name=tc.get("name", "")) for tc in tool_calls
        ]  # 替换掉原来的 tool_call 消息，避免被执行，原本是高风险的工具调用，现在替换为策略拦截消息， "策略拦截: 该工具调用被策略引擎判定为高风险，已被阻止执行。"
        audit_logger.log_event(
            thread_id=thread_id,
            event="system_action",
            action="policy_hard_deny",
            tools=[tc.get("name") for tc in tool_calls],
            level=decision.max_level_name,
            reasons=list(decision.reasons)[:6],
        )  # 记录策略拦截事件，包含工具名、最高风险等级、判定理由

        return {
            "messages": tool_msgs,# 替换掉原来的 tool_call 消息，避免被执行，原本是高风险的工具调用，现在替换为策略拦截消息， "策略拦截: 该工具调用被策略引擎判定为高风险，已被阻止执行。"
            "last_policy_summary": render_policy_decision(decision),  # 记录策略判定摘要
            "last_failure_class": "policy",  # 标记为策略拦截
            "last_failure_note": note[:300],  # 截断判定理由,防止过长
            "task_status": TASK_RUNNING,  # 标记为运行中,避免被 route_after_agent 直接 END
            "hard_deny_count": int(state.get("hard_deny_count") or 0) + 1  # 记录硬拒次数
        }

    # 如果启用规划器且没有任务计划，则检查最近的 HumanMessage 是否需要规划
    def route_entry(state: AgentState, config: RunnableConfig | None = None) -> str:
        """根据规划配置和请求复杂度选择首个处理节点。"""
        if not enable_planner or state.get("task_plan"):
            # 如果禁用规划器或已经有任务计划，则直接进入智能体节点
            target = "agent"
            _record_transition(state, config, "initialize", target)
            return target

        # 如果启用规划器且没有任务计划，则检查最近的 HumanMessage 是否需要规划
        last_human = next(
            (m for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)), None
        )  # 获取最近的 HumanMessage
        # 如果需要规划，则进入plannner节点，否则直接进入agent节点
        target = "planner" if last_human and _needs_plan(last_human.content) else "agent"
        _record_transition(state, config, "initialize", target)
        return target

    # 如果任务失败或预算超标，则直接结束；否则根据工具调用和策略引擎的判定，决定进入 approval/tools/policy_block 节点
    def route_after_agent(state: AgentState, config: RunnableConfig | None = None) -> str:
        """根据任务、预算及工具策略选择 Agent 后续节点。"""
        if state.get("task_status") in (TASK_FAILED, TASK_CANCELLED):
            target = END
            _record_transition(state, config, "agent", target)
            return target
        if budget_violation(state, run_budget):
            target = END
            _record_transition(state, config, "agent", target)
            return target
        
        decision = tools_condition(state)
        if decision != "tools":
            if (
                state.get("last_failure_class") == "model_protocol"
                and state.get("task_status") == TASK_RUNNING
                and int(state.get("retry_count") or 0) < 2
            ):
                # 如果策略引擎判定不需要审批，则进入 tools 节点
                target = "agent"
                _record_transition(state, config, "agent", target)
                return target
            if state.get("task_plan") and state.get("task_status") == TASK_RUNNING:
                target = "progress"
                _record_transition(state, config, "agent", target)
                return target
            target = END
            _record_transition(state, config, "agent", target)
            return target
        last_msg = state["messages"][-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        last_human = next(
            (m for m in reversed(state.get("messages") or []) if isinstance(m, HumanMessage)),
            None,
        )
        scoped_tools, scope_label = _tool_scope_for_turn(
            str(last_human.content) if last_human else "", actual_tools
        )
        allowed_names = {str(tool.name) for tool in scoped_tools}
        if any(tc.get("name") not in allowed_names for tc in tool_calls):
            target = "policy_block"
            _record_transition(state, config, "agent", target)
            return target
        policy_decision = assess_tool_calls(
            tool_calls,
            config=policy_cfg,
            require_approval=require_approval,
            approval_checker=(
                permission_manager.is_allowed if permission_manager else None),
        )
        if policy_decision.hard_deny:
            # 如果策略引擎判定有硬拒绝，则进入 policy_block 节点
            target = "policy_block"
            _record_transition(state, config, "agent", target)
            return target
        if policy_decision.needs_approval:
            # 如果策略引擎判定需要审批，则进入 approval 节点
            target = "approval"
            _record_transition(state, config, "agent", target)
            return target
    
        # 如果策略引擎判定不需要审批，则进入 tools 节点
        target = "tools"
        _record_transition(state, config, "agent", target)
        return target

    def route_after_planning(state: AgentState, config: RunnableConfig | None = None) -> str:
        """如果规划失败或被取消，则结束；否则继续到智能体节点"""
        target = END if state.get("task_status") in (TASK_FAILED, TASK_CANCELLED) else "agent"
        _record_transition(state, config, "planner", target)
        return target

    def route_after_replan(state: AgentState, config: RunnableConfig | None = None) -> str:
        """如果重规划失败或被取消，则结束；否则继续到智能体节点"""
        target = END if state.get("task_status") in (TASK_FAILED, TASK_CANCELLED) else "agent"
        _record_transition(state, config, "replan", target)
        return target

    def route_after_progress(state: AgentState, config: RunnableConfig | None = None) -> str:
        """根据任务进度选择结束、重规划或继续推理。"""
        if state.get("task_status") in (TASK_FAILED, TASK_CANCELLED):  # 任务失败，直接结束
            target = END
            _record_transition(state, config, "progress", target)
            return target
        if state.get("task_status") == TASK_DONE:  # 仅在整个计划完成（包括安装后的后续步骤）时结束。
            target = END
            _record_transition(state, config, "progress", target)
            return target
        if state.get("task_status") == TASK_REPLANNING:  # 如果任务状态是重规划，则进入 replan 节点
            target = "replan"
            _record_transition(state, config, "progress", target)
            return target
        if budget_violation(state, run_budget):  # 预算超标，直接结束
            target = END
            _record_transition(state, config, "progress", target)
            return target

        # 如果任务仍在运行中，则继续进入智能体节点
        target = "agent"
        _record_transition(state, config, "progress", target)
        return target

    # ================构建agent工作流的状态图==================
    # 添加8个节点
    workflow = StateGraph(AgentState)
    workflow.add_node("initialize", _instrument_node("initialize", initialize_run_node))  # 初始化
    workflow.add_node("planner", _instrument_node("planner", planner_node))  # 规划器节点
    workflow.add_node("agent", _instrument_node("agent", agent_node))  # 智能体节点
    workflow.add_node("approval", _instrument_node("approval", approval_node))  # 审批节点
    workflow.add_node("policy_block", _instrument_node("policy_block", policy_block_node))  # 策略拦截节点
    workflow.add_node("tools", _instrument_node("tools", tools_node))  # 工具调用节点
    workflow.add_node("progress", _instrument_node("progress", progress_node))  # 进度更新节点
    workflow.add_node("replan", _instrument_node("replan", replan_node))  # 重规划节点

    # 连边——先连固定边（无条件跳转）\一定转移
    workflow.add_edge(START, "initialize")       # 入口 → 初始化
    workflow.add_edge("approval", "tools")       # 批准 → 执行
    workflow.add_edge("policy_block", "progress")  # 硬拒 → 检查结果
    workflow.add_edge("tools", "progress")       # 执行完 → 检查结果

    # 连条件边（有路由函数决定走哪）
    workflow.add_conditional_edges(
        "initialize", route_entry,
        {
            "planner": "planner",
            "agent": "agent"
        }
    )
    workflow.add_conditional_edges(
        "agent", route_after_agent,
        {
            "approval": "approval",
            "policy_block": "policy_block",
            "tools": "tools",
            "progress": "progress",
            "agent": "agent",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "planner", route_after_planning, {"agent": "agent", END: END}
    )
    workflow.add_conditional_edges(
        "replan", route_after_replan, {"agent": "agent", END: END}
    )
    workflow.add_conditional_edges(
        "progress", route_after_progress,
        {
            "agent": "agent",
            "replan": "replan",
            END: END
        },
    )

    # 编译工作流为可执行应用
    interrupt_nodes = ["approval"] if require_approval else []
    app = workflow.compile(checkpointer=checkpointer,
                           interrupt_before=interrupt_nodes)
    return app
