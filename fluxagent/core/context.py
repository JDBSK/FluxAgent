# -*- coding: utf-8 -*-
"""Agent 状态定义与上下文管理。

本模块定义 LangGraph 状态图使用的 AgentState（对话历史 + 摘要 + 任务计划 +
运行计数），以及上下文裁剪策略 trim_context_messages（按完整用户回合裁剪，
保证 tool_call 与 ToolMessage 成对不被拆散）。

核心概念：
  - AgentState：LangGraph 的状态对象，messages 用 add_messages reducer 自动累加/删除
  - 双水位记忆：高水位=用户画像 md 文件（静态长期），低水位=摘要（动态短期）
  - 按回合裁剪：一个回合=HumanMessage + 后续所有 AI/Tool 消息，防止 ToolNode 报错
"""

from typing import Annotated, Any, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages

# 任务生命周期。waiting_approval 表示图已在 tools 前中断，等人批。
# replanning 表示 progress 判定需要换计划，下一跳走 replan 节点。
TASK_PENDING = "pending"           # 初始，还没开始跑
TASK_RUNNING = "running"           # 正在执行
TASK_WAITING_APPROVAL = "waiting_approval"  # 图停在 approval 节点等人批
TASK_REPLANNING = "replanning"        # progress 判定要换计划
TASK_DONE = "done"              # 全部步骤完成
TASK_FAILED = "failed"            # 失败终止
TASK_CANCELLED = "cancelled"         # 用户拒绝审批

STEP_PENDING = "pending"  # 计划步骤初始状态
STEP_RUNNING = "running"  # 计划步骤正在执行
STEP_DONE = "done"  # 计划步骤完成
STEP_FAILED = "failed"  # 计划步骤失败


def repair_unpaired_tool_calls(messages: list[BaseMessage]) -> list[BaseMessage]:
    """为历史中未完成的工具调用补齐协议所需的 ToolMessage。

    工具执行异常时，checkpoint 可能只保存了带 ``tool_calls`` 的 AIMessage，
    没有保存对应的 ToolMessage。下一次把这段历史交给 OpenAI 兼容接口会收到
    400。本函数只修复送入 LLM 的消息副本，不篡改原始 checkpoint。

    Args:
        messages: 准备交给模型的消息序列。

    Returns:
        list[BaseMessage]: 每个 AI tool_call 都有协议结果的消息序列。
    """
    repaired: list[BaseMessage] = []
    for index, message in enumerate(messages):
        repaired.append(message)
        if not isinstance(message, AIMessage):
            continue
        calls = getattr(message, "tool_calls", None) or []
        if not calls:
            continue

        existing_ids: set[str] = set()
        for next_message in messages[index + 1:]:
            if not isinstance(next_message, ToolMessage):
                break
            if next_message.tool_call_id:
                existing_ids.add(next_message.tool_call_id)

        for call in calls:
            call_id = call.get("id")
            if call_id and call_id not in existing_ids:
                repaired.append(
                    ToolMessage(
                        content=(
                            "工具调用未完成：上一轮在工具执行前异常中断。"
                            "请根据当前用户请求重新判断，不要盲目重试原参数。"
                        ),
                        tool_call_id=call_id,
                        name=call.get("name", ""),
                    )
                )
    return repaired

# 使用状态对象、不使用对话历史
# 因为有的东西从message中廉价重推不出来，必须跨节点存活，比如重试多少次、预算用多少等，显式放到状态中节点之间直接读取


class AgentState(TypedDict, total=False):
    """构建agent状态"""
    messages: Annotated[list[BaseMessage], add_messages] # 对话历史（自动累加/删除）
    summary: str # 摘要压缩,不存储在md,而是跟随状态被sql持久化

    # --- 工作状态：无法从 messages 廉价重推的东西才放进来 ---
    current_task: str  # 当前任务描述
    # 计划步骤 [{id, action, status, note}]。整体覆盖语义：重规划时替换全表，
    # 若挂 operator.add 追加，重规划会残留旧步骤，无法表达"计划被推翻"
    task_plan: list[dict[str, Any]]  # 计划步骤表
    step_index: int  # 游标：指向 task_plan 中正在执行的下标
    task_status: str
    retry_count: int  # 重试次数

    # --- 恢复 / 预算：同样必须跨节点存活 ---
    replan_count: int  # 重规划次数
    last_failure_class: str
    last_failure_note: str
    llm_calls: int
    tool_rounds: int
    run_started_at: float
    budget_exceeded_reason: str
    # 最近一次策略评估摘要（审批 UI / 审计用）
    last_policy_summary: str
    approval_interrupts: int
    hard_deny_count: int
    context_trim_count: int
    context_turns_before_trim: int
    context_turns_kept: int
    context_turns_discarded: int
    context_chars_before_trim: int
    context_chars_after_trim: int
    summary_llm_calls: int
    last_run_summary: dict


def make_plan_step(step_id: int, action: str, status: str = STEP_PENDING, note: str = "") -> dict[str, Any]:
    """构造一个计划步骤字典。

    Args:
        step_id: 步骤序号（从 1 开始）。
        action: 步骤要执行的动作描述。
        status: 步骤状态，默认 STEP_PENDING。
        note: 步骤备注（如失败原因），默认空。

    Returns:
        dict[str, Any]: 包含 id/action/status/note 的步骤字典。
    """
    return {"id": step_id, "action": action, "status": status, "note": note}


def advance_plan(
    task_plan: list[dict[str, Any]],
    step_index: int,
    outcome: str = STEP_DONE,
    note: str = "",
) -> tuple[list[dict[str, Any]], int, str]:
    """把当前步标记完成/失败并推进游标，返回 (新计划, 新游标, 任务状态)。

    纯函数，不碰 LLM，方便单测；节点只负责把结果写回 state。
    """
    if not task_plan:
        return [], step_index, TASK_RUNNING

    plan = [dict(s) for s in task_plan]
    idx = max(0, min(step_index, len(plan) - 1))
    plan[idx]["status"] = outcome
    if note:
        plan[idx]["note"] = note

    if outcome == STEP_FAILED:
        return plan, idx, TASK_FAILED

    next_idx = idx + 1
    if next_idx >= len(plan):
        return plan, len(plan) - 1, TASK_DONE
    plan[next_idx]["status"] = STEP_RUNNING
    return plan, next_idx, TASK_RUNNING


def render_plan(task_plan: list[dict[str, Any]], step_index: int) -> str:
    """把计划渲染成给 LLM 看的进度块，让它知道自己在第几步，而不是从历史里猜。"""
    if not task_plan:
        return ""
    marks = {STEP_DONE: "[x]", STEP_FAILED: "[!]",
             STEP_RUNNING: "[>]", STEP_PENDING: "[ ]"}
    lines = []
    for i, step in enumerate(task_plan):
        mark = marks.get(step.get("status", STEP_PENDING), "[ ]")
        cursor = " <= 当前" if i == step_index else ""
        lines.append(f"{mark} {i + 1}. {step.get('action', '')}{cursor}")
    return "\n".join(lines)


def measure_context(messages: list[BaseMessage]) -> dict[str, int]:
    """Return deterministic context-size proxies without a model tokenizer."""
    return {
        "turns": sum(isinstance(message, HumanMessage) for message in messages),
        "messages": len(messages),
        "characters": sum(len(str(getattr(message, "content", ""))) for message in messages),
    }


def trim_context_messages(messages: list[BaseMessage], trigger_turns: int = 20, keep_turns: int = 10) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """按完整用户回合裁剪上下文，保证 tool_call 与 ToolMessage 成对不被拆散。

    一个回合 = 一条 HumanMessage 开头 + 后续所有 AI/Tool 消息，直到下一条
    HumanMessage。回合数未达 trigger_turns 阈值时直接全量返回，不浪费 LLM 调用。

    Args:
        messages: 当前对话历史消息列表。
        trigger_turns: 触发裁剪的回合数阈值，默认 20。
        keep_turns: 裁剪后保留最近几个回合，默认 10。

    Returns:
        tuple[list, list]: (裁剪后保留的消息列表, 被裁掉待压缩的消息列表)。
    """
    if trigger_turns <= 0 or keep_turns <= 0:
        raise ValueError("trigger_turns and keep_turns must be positive")
    if keep_turns > trigger_turns:
        raise ValueError("keep_turns must not exceed trigger_turns")

    # 裁剪上下文
    # 按照完整用户回合来裁剪上下文：即 一个会从从HumanMessage开始，直到下一个HumanMessage结束，会把AIMessage、tool_calls、ToolMessage一并保留
    first_system = next(
        (m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    # 遍历非系统信息，按回合进行分组
    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            # 一个回合== HumanMessage + 之后的所有消息，直到下一个HumanMessage
            # 保证ToolMessage、AIMessage、ToolCall都在同一个回合中,不被拆散
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)

    # 保存最后一个回合
    if current_turn:
        turns.append(current_turn)

    total_turns = len(turns)

    # 双阈值压缩，回合数达到阈值，才开始裁剪，裁剪后保留最近的回合数；否则直接返回
    if total_turns < trigger_turns:
        final_messages = (
            [first_system] if first_system else []) + non_system_msgs
        return final_messages, []

    # 如果回合数超过触发阈值，则进行裁剪
    recent_turns = turns[-keep_turns:]  # 保留最近的回合
    discarded_turns = turns[:-keep_turns]  # 前面的拿去压缩

    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    for turn in recent_turns:
        final_messages.extend(turn)

    discarded_messages: list[BaseMessage] = []
    for turn in discarded_turns:
        discarded_messages.extend(turn)

    return final_messages, discarded_messages
