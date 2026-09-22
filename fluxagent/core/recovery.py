"""失败分类与重规划辅助。

旧逻辑：任何失败都 retry_count+1，到 MAX_RETRY 就整任务 failed。
问题：
1. 越权/策略拒绝再试 N 次纯属烧调用，计划也不会自己变合法。
2. 瞬时工具错误与「计划本身不合理」被同等对待。
3. 没有重规划，失败即死，面试官会追问恢复路径。

新逻辑（显式状态机，不靠 LLM 自己从历史里悟）：
- transient: 同一步重试
- policy: 权限/越权/策略硬拒 -> 不盲重试，直接尝试 replan 或 abort
- exhausted: 重试用尽 -> replan（若还有 replan 预算）否则 abort
- budget: 运行时预算用尽 -> abort
- fatal: 不可恢复 -> abort

replan 是整表替换 task_plan（与创建计划同一语义），不是 append。
"""
from __future__ import annotations

import json
import re

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence


# 动作：progress / route 读这个决定走哪条边
ACTION_RETRY = "retry"
ACTION_REPLAN = "replan"
ACTION_ABORT = "abort"

# 失败类：写入 state 供审计与 prompt 使用
CLASS_TRANSIENT = "transient"
CLASS_POLICY = "policy"
CLASS_EXHAUSTED = "exhausted"
CLASS_BUDGET = "budget"
CLASS_FATAL = "fatal"
CLASS_PERMANENT = "permanent"
CLASS_NONE = "none"

_POLICY_MARKERS = (
    "越权拦截",
    "权限拒绝",
    "策略拒绝",
    "hard_deny",
    "policy_deny",
    "route_mismatch",
    "stale_contract",
    "Path traversal",
    "PermissionError",
)
_TRANSIENT_MARKERS = (
    "❌",
    "Error",
    "超时",
    "Timeout",
    "暂时",
    "失败",
    "不可用",
)
_PERMANENT_MARKERS = (
    "文件不存在",
    "文件未找到",
    "找不到文件",
    "路径不存在",
    "目录不存在",
    "no such file",
    "file not found",
    "filenotfounderror",
    "参数错误",
    "参数无效",
    "无效参数",
    "invalid parameter",
    "invalid argument",
    "missing required",
    "缺少必要参数",
    "schema_error",
)


@dataclass(frozen=True)
class FailureDecision:
    """描述工具失败后的恢复动作及更新后计数。"""

    action: str
    failure_class: str
    note: str
    retry_count: int
    replan_count: int


def extract_tool_failure_texts(messages: Sequence[Any]) -> list[str]:
    """按最近调用批次的 ID 对账，缺失结果也视为失败。"""
    batch = extract_tool_call_batch(messages)
    if not batch:
        return []
    results = {
        str(getattr(message, "tool_call_id", "")): message
        for message in messages
        if getattr(message, "type", None) == "tool"
    }
    texts = []
    for call in batch:
        call_id = str(call.get("id", ""))
        result = results.get(call_id)
        if result is None:
            # 有调用没结果：按失败处理，绝不当作成功
            texts.append("❌ missing_tool_result:" + call_id)
        elif tool_result_failed(result):
            texts.append("❌ " + str(result.content))
        else:
            texts.append(json.dumps({"ok": True}, ensure_ascii=False))
    return texts


def extract_tool_call_batch(messages: Sequence[Any]) -> list[dict]:
    """返回"结果已回填但模型还没消化"的那个 tool_calls 批次。

    只有序列末尾是 ToolMessage 时才存在待对账批次；模型已经用文本回答过
    的批次视为已消化，返回空列表。
    """
    if not messages or getattr(messages[-1], "type", None) != "tool":
        return []
    for message in reversed(list(messages)[:-1]):
        if getattr(message, "type", None) == "human":
            return []
        calls = getattr(message, "tool_calls", None) or []
        if calls:
            return list(calls)
    return []


def tool_result_failed(message: Any) -> bool:
    """优先读取执行边界的结构化状态，再兼容旧工具文本。"""
    if getattr(message, "status", None) == "error":
        return True
    metadata = getattr(message, "additional_kwargs", {}) or {}
    outcome = metadata.get("fluxagent_result", {})
    if isinstance(outcome, dict):
        flag = _ok_flag(outcome.get("ok"))
        if flag is not None:
            return not flag
    return is_tool_failure_text(str(getattr(message, "content", "")))


def _ok_flag(value: Any) -> Optional[bool]:
    """把工具自报的 ``ok`` 标记转成布尔；认不出就返回 None，绝不猜。

    工具是手写代码，``ok`` 常常是字符串或 0/1 而不是 JSON 布尔值；
    只认 ``is True/False`` 会把 ``{"ok":"false"}`` 当成成功（漏报失败）。
    """
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "ok", "success", "成功"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "error", "失败"}:
            return False
    return None


def normalize_tool_result(message: Any) -> Any:
    """保留原始内容，给执行结果附上统一状态。"""
    failed = tool_result_failed(message)
    metadata = dict(getattr(message, "additional_kwargs", {}) or {})
    metadata["fluxagent_result"] = {"ok": not failed}
    return message.model_copy(update={
        "status": "error" if failed else "success",
        "additional_kwargs": metadata,
    })


def tool_call_key(call: dict) -> str:
    """重试去重只认相同工具和完全相同的参数。"""
    return json.dumps([call.get("name"), call.get("args") or {}],
                      ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_tool_failure_text(text: str) -> bool:
    """Recognize legacy text errors and structured ``ok=false`` tool results."""
    s = str(text or "")
    head = s.lstrip()
    if head.startswith("{"):
        try:
            payload = json.loads(head)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict):
            flag = _ok_flag(payload.get("ok"))
            if flag is not None:
                return not flag
    if head.startswith(("❌", "权限拒绝", "越权拦截", "策略拒绝", "工具路由拒绝")):
        return True
    if re.match(r"(?:[\w.]*Error|error|exception|timeout)\s*[:：\n]", head, re.I):
        return True
    if re.match(r"(?:保存|写入|读取|执行|安装|请求|调用|操作|连接|验证|删除|创建|下载)?失败(?:[：:，,。\s]|$)", head):
        return True
    if re.match(r"(?:policy_deny|hard_deny|route_mismatch|stale_contract)\b", head):
        return True
    if any(head.casefold().startswith(marker.casefold()) for marker in _PERMANENT_MARKERS):
        return True
    return False


def any_tool_failure(texts: Iterable[str]) -> bool:
    """判断给定工具输出中是否存在失败。"""
    return any(is_tool_failure_text(t) for t in texts)


def is_policy_failure(texts: Iterable[str]) -> bool:
    """判断工具输出是否包含权限或策略拒绝。"""
    for t in texts:
        s = str(t)
        if any(m in s for m in _POLICY_MARKERS):
            return True
    return False


def is_permanent_failure(texts: Iterable[str]) -> bool:
    """识别重试无法修复的输入或资源问题。"""
    for t in texts:
        s = str(t or "").casefold()
        if any(marker.casefold() in s for marker in _PERMANENT_MARKERS):
            return True
    return False


def classify_failure(
    texts: Sequence[str],
    *,
    retry_count: int,
    max_retries: int,
    replan_count: int,
    max_replans: int,
    enable_replan: bool = True,
) -> FailureDecision:
    """根据工具输出与计数，决定 retry / replan / abort。

    计数语义：
    - 传入的 retry_count / replan_count 是「当前已发生次数」
    - 本函数返回的是「决策后应写回的新计数」
    """
    joined = " | ".join(str(t)[:120] for t in texts)[:240] or "unknown_failure"

    if is_policy_failure(texts):
        # 策略/越权：盲重试无意义。有 replan 预算就换计划，否则终止。
        if enable_replan and replan_count < max_replans:
            return FailureDecision(
                action=ACTION_REPLAN,
                failure_class=CLASS_POLICY,
                note=f"policy_failure:{joined}",
                retry_count=0,
                replan_count=replan_count + 1,
            )
        return FailureDecision(
            action=ACTION_ABORT,
            failure_class=CLASS_POLICY,
            note=f"policy_failure_no_replan:{joined}",
            retry_count=retry_count,
            replan_count=replan_count,
        )

    if is_permanent_failure(texts):
        if enable_replan and replan_count < max_replans:
            return FailureDecision(
                action=ACTION_REPLAN,
                failure_class=CLASS_PERMANENT,
                note=f"permanent_failure:{joined}",
                retry_count=0,
                replan_count=replan_count + 1,
            )
        return FailureDecision(
            action=ACTION_ABORT,
            failure_class=CLASS_PERMANENT,
            note=f"permanent_failure_no_replan:{joined}",
            retry_count=retry_count,
            replan_count=replan_count,
        )

    # max_retries 表示当前步骤允许的总失败尝试数；首次失败后，最多再重试
    # max_retries - 1 次。例如 3 表示“首次 + 两次重试”。
    new_retry = retry_count + 1
    if new_retry < max_retries:
        return FailureDecision(
            action=ACTION_RETRY,
            failure_class=CLASS_TRANSIENT,
            note=f"transient:{joined}",
            retry_count=new_retry,
            replan_count=replan_count,
        )

    # 重试用尽
    if enable_replan and replan_count < max_replans:
        return FailureDecision(
            action=ACTION_REPLAN,
            failure_class=CLASS_EXHAUSTED,
            note=f"retries_exhausted:{joined}",
            retry_count=0,
            replan_count=replan_count + 1,
        )

    return FailureDecision(
        action=ACTION_ABORT,
        failure_class=CLASS_EXHAUSTED,
        note=f"retries_and_replans_exhausted:{joined}",
        retry_count=new_retry,
        replan_count=replan_count,
    )


def normalize_action(text: str) -> str:
    """规范化动作文本以便进行等价性比较。"""
    s = str(text or "").strip().lower()
    for ch in ["\t", "\r", "\n"]:
        s = s.replace(ch, " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def failed_step_action(old_plan, step_index: int = 0) -> str:
    """从旧计划中取得失败步骤的动作文本。"""
    if not old_plan:
        return ""
    idx = max(0, min(int(step_index or 0), len(old_plan) - 1))
    return str((old_plan[idx] or {}).get("action") or "")


def is_isomorphic_action(a: str, b: str) -> bool:
    """判断两个动作文本是否表达近似的执行步骤。"""
    na, nb = normalize_action(a), normalize_action(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return shorter in longer and len(shorter) >= max(4, int(len(longer) * 0.6))


def validate_replan_steps(steps, forbidden_action: str = "", min_steps: int = 2, max_steps: int = 6):
    """清理并校验重规划步骤的数量与动作差异。"""
    cleaned = []
    for s in steps or []:
        text = str(s).strip()
        if not text:
            continue
        if forbidden_action and is_isomorphic_action(text, forbidden_action):
            continue
        cleaned.append(text)
    dedup = []
    for s in cleaned:
        if dedup and is_isomorphic_action(dedup[-1], s):
            continue
        dedup.append(s)
    if len(dedup) > max_steps:
        dedup = dedup[:max_steps]
    ok = min_steps <= len(dedup) <= max_steps
    return ok, dedup


def build_replan_prompt(goal: str, old_plan, failure_note: str, failure_class: str = "", step_index: int = 0) -> str:
    """构造要求避开失败动作的重规划提示词。"""
    forbidden = failed_step_action(old_plan, step_index)
    rendered = []
    for i, step in enumerate(old_plan or []):
        rendered.append(str(i + 1) + ". " + str(step.get("action", "")
                                                ) + " [" + str(step.get("status", "")) + "]")
    plan_txt = chr(10).join(rendered) if rendered else "(no old plan)"
    parts = [
        "You are the replan module. Previous execution failed. Output a REPLACEMENT plan.",
        "Goal: " + str(goal),
        "Failure class: " + str(failure_class or "unknown"),
        "Failure note: " + str(failure_note),
        "Failed step action (DO NOT repeat isomorphic step): " + forbidden,
        "Old plan:",
        plan_txt,
        "",
        "Rules: 1) ONLY JSON string array 2) 2-6 steps 3) Avoid failed approach",
    ]
    return chr(10).join(parts)


def fallback_replan_steps(goal: str, failure_note: str, forbidden_action: str = ""):
    """在模型重规划不可用时生成基础替代步骤。"""
    g = str(goal or "continue user task").strip()[:80]
    note = str(failure_note or "").lower()[:40]
    steps = ["diagnose failure and change approach:" +
             note, "use alternate safe path for:" + g]
    if forbidden_action:
        steps[0] = "avoid failed action and reassess:" + \
            normalize_action(forbidden_action)[:40]
    return steps
