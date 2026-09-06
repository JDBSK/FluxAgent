"""审批闸门的控制面。

图停在 approval 节点后，调用方需要三件事：看清要批什么、批准、拒绝。
这里把它们收成函数，CLI 和评测共用同一套语义，避免两边各写一遍走偏。

拒绝为什么不是"直接跳到 END"：
tool_calls 已经在历史里了，OpenAI 协议要求每个 tool_call 必须有对应的
ToolMessage 回应，否则下一轮请求会 400。所以拒绝的正确做法是伪造一条
"用户拒绝"的 ToolMessage 喂回去，让 LLM 知道被拒并自己组织说辞。
"""
from typing import Any, Optional

from langchain_core.messages import ToolMessage

from .context import TASK_CANCELLED
from .logger import audit_logger
from .policy import assess_tool_calls, render_policy_decision

REJECT_NOTE = "用户拒绝执行该操作。请不要重试，向用户说明已取消。"


def pending_approval(app, config: dict) -> Optional[list[dict[str, Any]]]:
    """图是否停在审批点。返回待批的 tool_calls，没停就返回 None。"""
    snapshot = app.get_state(config)
    if not snapshot.next or "approval" not in snapshot.next:
        return None
    messages = snapshot.values.get("messages") or []
    if not messages:
        return None
    calls = getattr(messages[-1], "tool_calls", None)
    return list(calls) if calls else None


async def apending_approval(app, config: dict) -> Optional[list[dict[str, Any]]]:
    """Async variant for graphs compiled with an async checkpointer."""
    snapshot = await app.aget_state(config)
    if not snapshot.next or "approval" not in snapshot.next:
        return None
    messages = snapshot.values.get("messages") or []
    if not messages:
        return None
    calls = getattr(messages[-1], "tool_calls", None)
    return list(calls) if calls else None


def describe_calls(tool_calls: list[dict[str, Any]]) -> str:
    """把待批调用渲染成人能读的一段话——带风险定级。"""
    decision = assess_tool_calls(tool_calls, require_approval=True)
    lines = [render_policy_decision(decision), "待批准调用:"]
    for tc in tool_calls:
        args = tc.get("args") or {}
        rendered = ", ".join(f"{k}={str(v)[:60]}" for k, v in args.items()) or "无参数"
        lines.append(f"  - {tc.get('name')}({rendered})")
    return "\n".join(lines)


def approve_prepare(app, config: dict, tool_calls: Optional[list] = None) -> None:
    """批准前的准备：只记审计，不推进图。

    prepare 与 resume 分离，是因为调用方可能用 astream 流式续跑（CLI 要显示
    过程），如果 prepare 里就 invoke 掉，外层再 astream 会重复推进一次。
    """
    thread_id = config.get("configurable", {}).get("thread_id", "system_default")
    audit_logger.log_event(
        thread_id=thread_id, event="system_action", action="approval_granted",
        tools=[tc.get("name") for tc in (tool_calls or [])],
    )


def reject_prepare(
    app, config: dict, tool_calls: list[dict[str, Any]], reason: str = REJECT_NOTE
) -> None:
    """拒绝前的准备：伪造 ToolMessage 补齐协议并标记 cancelled，不推进图。

    as_node="tools" 是关键——告诉 LangGraph"这些消息当作 tools 节点的产出"，
    图才会走 tools 之后的边（progress→agent），而不是真去执行工具。
    """
    thread_id = config.get("configurable", {}).get("thread_id", "system_default")
    tool_msgs = [
        ToolMessage(content=reason, tool_call_id=tc["id"], name=tc.get("name", ""))
        for tc in tool_calls
    ]
    app.update_state(
        config, {"messages": tool_msgs, "task_status": TASK_CANCELLED}, as_node="tools"
    )
    audit_logger.log_event(
        thread_id=thread_id, event="system_action", action="approval_denied",
        tools=[tc.get("name") for tc in tool_calls],
    )


async def areject_prepare(
    app, config: dict, tool_calls: list[dict[str, Any]], reason: str = REJECT_NOTE
) -> None:
    """Async rejection preparation for graphs using an async checkpointer."""
    thread_id = config.get("configurable", {}).get("thread_id", "system_default")
    tool_msgs = [
        ToolMessage(content=reason, tool_call_id=tc["id"], name=tc.get("name", ""))
        for tc in tool_calls
    ]
    await app.aupdate_state(
        config, {"messages": tool_msgs, "task_status": TASK_CANCELLED}, as_node="tools"
    )
    audit_logger.log_event(
        thread_id=thread_id, event="system_action", action="approval_denied",
        tools=[tc.get("name") for tc in tool_calls],
    )


def approve(app, config: dict, tool_calls: Optional[list] = None):
    """批准并同步续跑到下一个停点。评测/脚本用这个一步到位。"""
    approve_prepare(app, config, tool_calls)
    return app.invoke(None, config)


def reject(app, config: dict, tool_calls: list[dict[str, Any]], reason: str = REJECT_NOTE):
    """拒绝并同步续跑。"""
    reject_prepare(app, config, tool_calls, reason)
    return app.invoke(None, config)


async def aapprove(app, config: dict, tool_calls: Optional[list] = None):
    """Approve and resume a graph using async execution APIs."""
    approve_prepare(app, config, tool_calls)
    return await app.ainvoke(None, config)


async def areject(
    app, config: dict, tool_calls: list[dict[str, Any]], reason: str = REJECT_NOTE
):
    """Reject and resume a graph using async execution APIs."""
    await areject_prepare(app, config, tool_calls, reason)
    return await app.ainvoke(None, config)
