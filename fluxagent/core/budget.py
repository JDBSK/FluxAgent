from __future__ import annotations

# -*- coding: utf-8 -*-
"""运行时预算：防止 Agent 回环烧钱、拖死、无限制重规划。

设计取舍：
- 不用真实 tokenizer 算 token：评测/单测要确定性，且不同模型计价不同。
  用 llm_calls / tool_rounds 作为成本代理，足够做熔断，也足够面试讲清。
- 墙钟超时与步数上限分开：有的调用慢但不一定步数多。
- 纯函数 check：节点里只负责读 state 填计数，是否超限可单测。
"""


import time
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RunBudget:
    """定义单次 Agent 运行可消耗的资源上限。"""

    max_tool_rounds: int = 24
    max_llm_calls: int = 40
    max_replans: int = 2
    max_retries_per_step: int = 3
    max_wall_seconds: float = 300.0

    def to_dict(self) -> dict[str, Any]:
        """将预算配置转换为普通字典。"""
        return asdict(self)


DEFAULT_BUDGET = RunBudget()


def init_run_clock(state: dict[str, Any], now: Optional[float] = None) -> dict[str, Any]:
    """任务首次进入时打上开始时间；已有则不改。"""
    if state.get("run_started_at"):
        return {}
    return {"run_started_at": float(now if now is not None else time.time())}


def bump_llm_calls(state: dict[str, Any], n: int = 1) -> dict[str, Any]:
    """LLM 调用计数 +n，返回写回 state 的增量字典。"""
    return {"llm_calls": int(state.get("llm_calls") or 0) + n}


def bump_tool_rounds(state: dict[str, Any], n: int = 1) -> dict[str, Any]:
    """工具轮次计数 +n，返回写回 state 的增量字典。"""
    return {"tool_rounds": int(state.get("tool_rounds") or 0) + n}


def budget_violation(
    state: dict[str, Any],
    budget: Optional[RunBudget] = None,
    now: Optional[float] = None,
    *,
    pending_llm_calls: int = 0,
    pending_tool_rounds: int = 0,
    pending_replans: int = 0,
) -> Optional[str]:
    """Return a reason when current or reserved work would exceed the budget.

    ``pending_*`` is used before an expensive action. Without reservation, a
    state exactly at the limit would be allowed to perform one extra call and
    only fail afterwards.
    """
    b = budget or DEFAULT_BUDGET
    llm_calls = int(state.get("llm_calls") or 0) + max(0, int(pending_llm_calls))
    tool_rounds = int(state.get("tool_rounds") or 0) + max(0, int(pending_tool_rounds))
    replan_count = int(state.get("replan_count") or 0) + max(0, int(pending_replans))
    started = state.get("run_started_at")
    now_ts = float(now if now is not None else time.time())

    if llm_calls > b.max_llm_calls:
        return f"llm_calls_exceeded:{llm_calls}>{b.max_llm_calls}"
    if tool_rounds > b.max_tool_rounds:
        return f"tool_rounds_exceeded:{tool_rounds}>{b.max_tool_rounds}"
    if replan_count > b.max_replans:
        return f"replans_exceeded:{replan_count}>{b.max_replans}"
    if started is not None:
        elapsed = now_ts - float(started)
        if elapsed > b.max_wall_seconds:
            return f"wall_timeout:{elapsed:.2f}>{b.max_wall_seconds}"
    return None


def apply_budget_fail(reason: str) -> dict[str, Any]:
    """超限时写回 state 的标准字段。"""
    return {
        "task_status": "failed",
        "budget_exceeded_reason": reason,
        "last_failure_class": "budget",
        "last_failure_note": reason,
    }

def build_run_summary(state: dict[str, Any], now: Optional[float] = None) -> dict[str, Any]:
    """One-run structured metrics for CLI/eval/interview evidence."""
    now_ts = float(now if now is not None else time.time())
    started = state.get("run_started_at")
    elapsed = None
    if started is not None:
        elapsed = round(now_ts - float(started), 3)
    return {
        "task_status": state.get("task_status"),
        "llm_calls": int(state.get("llm_calls") or 0),
        "tool_rounds": int(state.get("tool_rounds") or 0),
        "replan_count": int(state.get("replan_count") or 0),
        "retry_count": int(state.get("retry_count") or 0),
        "approval_interrupts": int(state.get("approval_interrupts") or 0),
        "hard_deny_count": int(state.get("hard_deny_count") or 0),
        "context_trim_count": int(state.get("context_trim_count") or 0),
        "context_turns_before_trim": int(state.get("context_turns_before_trim") or 0),
        "context_turns_kept": int(state.get("context_turns_kept") or 0),
        "context_turns_discarded": int(state.get("context_turns_discarded") or 0),
        "context_chars_before_trim": int(state.get("context_chars_before_trim") or 0),
        "context_chars_after_trim": int(state.get("context_chars_after_trim") or 0),
        "summary_llm_calls": int(state.get("summary_llm_calls") or 0),
        "budget_exceeded_reason": state.get("budget_exceeded_reason") or "",
        "last_failure_class": state.get("last_failure_class") or "",
        "elapsed_seconds": elapsed,
    }
