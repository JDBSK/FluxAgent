from __future__ import annotations
from urllib.parse import urlparse
from typing import Any, Callable, Iterable, Optional
from enum import IntEnum
from dataclasses import dataclass, field
import re

# -*- coding: utf-8 -*-
"""风险策略引擎：按「工具类型 x 参数信号」定级，而不是只看工具名集合。

分级：safe（放行）/ approval（人工审批）/ critical（硬拒绝）。
判定维度：工具名是否在副作用名单 + 参数是否含危险信号（路径穿越/危险命令/大范围删除）。
高危调用在工具执行前硬拒绝，副作用操作进入 HITL 审批，以沙盒为第二层防线。
"""

"""风险策略引擎：按「工具类型 × 参数信号」定级，而不是只看工具名集合。

为什么要独立成纯函数模块：
1. 路由/CLI/评测/MCP 都能复用同一套判定，避免各写一份漂移。
2. 不依赖 LLM、不依赖图运行时，可组合穷举测。
3. 静态 SIDE_EFFECT_TOOLS 能防审批疲劳，但分不出 `echo hello` 和 `rm -rf`；
   策略引擎补的是参数维度。

级别（低 -> 高）：read / write / mutate / exec / critical

硬拒绝 vs 沙盒：
- 策略层：执行前挡住明显恶意参数
- 沙盒层：最后兜底
"""


class RiskLevel(IntEnum):
    """表示工具调用风险等级。"""

    READ = 10
    WRITE = 20
    MUTATE = 30
    EXEC = 40
    CRITICAL = 50


LEVEL_NAME = {
    RiskLevel.READ: "read",
    RiskLevel.WRITE: "write",
    RiskLevel.MUTATE: "mutate",
    RiskLevel.EXEC: "exec",
    RiskLevel.CRITICAL: "critical",
}


SIDE_EFFECT_TOOLS = {
    "write_office_file",
    "execute_office_shell",
    "propose_memory_change",
    "schedule_task",
    "delete_scheduled_task",
    "modify_scheduled_task",
    "install_skill",
    "install_mcp_server",
}


def is_side_effect_tool(name: str) -> bool:
    """判断工具是否属于已知副作用工具。"""

    return name in SIDE_EFFECT_TOOLS


TOOL_BASE_LEVEL: dict[str, RiskLevel] = {
    "get_current_time": RiskLevel.READ,
    "calculator": RiskLevel.READ,
    "list_office_files": RiskLevel.READ,
    "read_office_file": RiskLevel.READ,
    "list_scheduled_tasks": RiskLevel.READ,
    "get_system_model_info": RiskLevel.READ,
    # 委派是编排/分析类动作：子 Agent 默认只读，委派本身不产生副作用。
    # 子 Agent 作用域内的具体工具调用另由 subagents 的策略闸门单独评估。
    "delegate_task": RiskLevel.READ,
    "propose_memory_change": RiskLevel.MUTATE,
    "write_office_file": RiskLevel.WRITE,
    "schedule_task": RiskLevel.MUTATE,
    "delete_scheduled_task": RiskLevel.MUTATE,
    "modify_scheduled_task": RiskLevel.MUTATE,
    "execute_office_shell": RiskLevel.EXEC,
    "install_skill": RiskLevel.EXEC,
    "install_mcp_server": RiskLevel.EXEC,
    "list_installed_mcp_servers": RiskLevel.READ,
}

_INTERPRETER_NAMES = frozenset({
    "bash", "cmd", "cmd.exe", "node", "node.exe", "perl", "perl.exe",
    "powershell", "powershell.exe", "pwsh", "pwsh.exe", "python",
    "python.exe", "python3", "ruby", "ruby.exe",
})
_NETWORK_CLIENT_NAMES = frozenset({"curl", "curl.exe", "wget", "wget.exe"})
_DESTRUCTIVE_COMMAND_NAMES = frozenset({
    "dd", "del", "erase", "format", "mkfs", "rd", "reboot",
    "remove-item", "rm", "rmdir", "shutdown",
})
_TRAVERSAL_MARKERS = ("..", "/etc/", "~/", "%userprofile%")
_WINDOWS_DRIVE_PATH = re.compile(r"(?i)(?:^|[\s\"'(<>=,;|&])([a-z]:[\\/])")
_WINDOWS_UNC_PATH = re.compile(
    r"(?:^|[\s\"'(<>=,;|&])(\\\\[^\\/\s\"']+[\\/][^\s\"']+)"
)
_COMMAND_NAME = re.compile(r"^[\s\"']*([a-z0-9_.-]+)", re.IGNORECASE)

ApprovalChecker = Callable[[str, dict[str, Any]], bool]


@dataclass(frozen=True)
class PolicyConfig:
    """定义风险评估和审批门槛配置。"""

    gate_level: RiskLevel = RiskLevel.WRITE
    hard_deny_critical: bool = True
    legacy_gate_tools: Optional[frozenset[str]] = None
    tool_base_overrides: dict[str, RiskLevel] = field(default_factory=dict)


@dataclass(frozen=True)
class CallAssessment:
    """记录单次工具调用的风险评估结果。"""

    name: str
    args: dict[str, Any]
    level: RiskLevel
    reasons: tuple[str, ...]
    hard_deny: bool

    @property
    def level_name(self) -> str:
        """返回风险等级的可读名称。"""

        return LEVEL_NAME[self.level]


@dataclass(frozen=True)
class PolicyDecision:
    """汇总一批工具调用的策略决策。"""

    assessments: tuple[CallAssessment, ...]
    max_level: RiskLevel
    needs_approval: bool
    hard_deny: bool
    reasons: tuple[str, ...]

    @property
    def max_level_name(self) -> str:
        """返回最高风险等级的可读名称。"""

        return LEVEL_NAME[self.max_level]


def _as_args(raw: Any) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _path_signals(*values: Any) -> list[str]:
    reasons: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if not text:
            continue
        if any(m in text for m in _TRAVERSAL_MARKERS):
            reasons.append(f"path_suspicious:{text[:40]}")
        if any(
            scheme in text
            for scheme in ("file://", "ftp://", "http://", "https://", "smb://")
        ):
            reasons.append(f"path_url_scheme:{text[:40]}")
        if (
            _WINDOWS_DRIVE_PATH.search(text)
            or _WINDOWS_UNC_PATH.search(text)
            or text.startswith("~")
        ):
            reasons.append(f"path_escape:{text[:40]}")
    return reasons


def _normalize_cmd(command: str) -> str:
    """Collapse spacing / chain variants so simple bypasses still match rules."""
    text = (command or "").strip().lower()
    # normal space, NBSP, zero-width / ideographic spaces
    text = re.sub(r"[\s\u00a0\u2000-\u200b\u202f\u205f\u3000]+", " ", text)
    return text


def _command_name(command: str) -> str:
    """Return the executable token without matching words inside arguments."""
    match = _COMMAND_NAME.match(command or "")
    return (match.group(1) if match else "").casefold()


def _command_segments(command: str) -> list[str]:
    """Split common command chains while preserving quoted operators."""
    text = command or ""
    segments: list[str] = []
    start = 0
    quote = ""
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = ""
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            index += 1
            continue
        size = 1 if char == ";" else 2 if text.startswith(
            ("&&", "||"), index) else 0
        if size:
            segment = text[start:index].strip()
            if segment:
                segments.append(segment)
            index += size
            start = index
            continue
        index += 1
    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def _nested_shell_command(segment: str, executable: str) -> str:
    """Extract a visible cmd or PowerShell command payload."""
    if executable in {"cmd", "cmd.exe"}:
        match = re.search(r"(?i)(?:^|\s)/(?:c|k)\s+(.+)$", segment)
    elif executable in {"powershell", "powershell.exe", "pwsh", "pwsh.exe"}:
        match = re.search(r"(?i)(?:^|\s)-(?:command|c)\s+(.+)$", segment)
    else:
        return ""
    return match.group(1).strip().strip("\"'") if match else ""


def _shell_path_signals(text: str) -> list[str]:
    """Detect traversal, URL, quoted drive, and UNC paths in commands."""
    reasons: list[str] = []
    if _WINDOWS_DRIVE_PATH.search(text):
        reasons.append("shell_absolute_drive_path")
    if _WINDOWS_UNC_PATH.search(text):
        reasons.append("shell_unc_path")
    if any(marker in text for marker in _TRAVERSAL_MARKERS):
        reasons.append("shell_traversal")
    if re.search(r"(?i)\b(?:file|ftp|http|https|smb)://", text):
        reasons.append("shell_url_scheme")
    return reasons


def _installer_signals(args: dict[str, Any]) -> tuple[RiskLevel, list[str]]:
    """Validate installer source, subdirectory, command, and launch arguments."""
    level = RiskLevel.EXEC
    reasons: list[str] = []
    source_type = str(args.get("source_type", "")).strip().casefold()
    source = str(args.get("source", "")).strip()
    subdirectory = str(args.get("subdirectory", "")).strip()
    if not source_type:
        reasons.append("installer_source_type_missing")
    elif source_type == "git":
        parsed = urlparse(source)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            reasons.append("installer_invalid_git_source")
            level = RiskLevel.CRITICAL
        else:
            reasons.append(f"installer_git_host:{parsed.hostname.casefold()}")
    elif source_type == "local":
        if re.match(r"(?i)^[a-z][a-z0-9+.-]*://", source):
            reasons.append("installer_local_source_url")
            level = RiskLevel.CRITICAL
        else:
            reasons.append("installer_local_source")
    else:
        reasons.append("installer_invalid_source_type")
        level = RiskLevel.CRITICAL
    if subdirectory and _path_signals(subdirectory):
        reasons.append("installer_unsafe_subdirectory")
        level = RiskLevel.CRITICAL
    command = str(args.get("command", "")).strip()
    if command:
        command_name = _command_name(command)
        if not command_name or re.search(r"[;&|<>]", command):
            reasons.append("installer_unsafe_launch_command")
            level = RiskLevel.CRITICAL
        elif command_name in _INTERPRETER_NAMES:
            reasons.append(f"installer_launch_interpreter:{command_name}")
    launch_args = args.get("launch_args", args.get("args", []))
    if launch_args is not None and not isinstance(launch_args, (list, tuple)):
        reasons.append("installer_invalid_launch_args")
        level = RiskLevel.CRITICAL
    else:
        for value in launch_args or []:
            if _path_signals(value):
                reasons.append("installer_unsafe_launch_arg")
                level = RiskLevel.CRITICAL
                break
    return level, reasons


def _shell_signals(command: str) -> tuple[RiskLevel, list[str]]:
    text = _normalize_cmd(command)
    reasons: list[str] = []
    level = RiskLevel.EXEC
    if not text:
        return level, ["shell_empty"]
    path_reasons = _shell_path_signals(text)
    if path_reasons:
        reasons.extend(path_reasons)
        level = RiskLevel.CRITICAL
    for segment in _command_segments(text):
        executable = _command_name(segment)
        if executable in _DESTRUCTIVE_COMMAND_NAMES:
            reasons.append(f"shell_destructive:{executable}")
            level = RiskLevel.CRITICAL
        if executable in _INTERPRETER_NAMES:
            reasons.append(f"shell_interpreter:{executable}")
            nested = _nested_shell_command(segment, executable)
            nested_name = _command_name(nested)
            if nested_name in _DESTRUCTIVE_COMMAND_NAMES:
                reasons.append(f"shell_destructive:{nested_name}")
                level = RiskLevel.CRITICAL
        elif executable in _NETWORK_CLIENT_NAMES:
            reasons.append(f"shell_network_client:{executable}")
    if not reasons:
        reasons.append("shell_exec")
    return level, reasons


def assess_call(
    name: str,
    args: Optional[dict[str, Any]] = None,
    config: Optional[PolicyConfig] = None,
) -> CallAssessment:
    """评估单次工具调用的风险等级和拒绝状态。"""

    cfg = config or PolicyConfig()
    args = _as_args(args)
    reasons: list[str] = []

    base = cfg.tool_base_overrides.get(name) or TOOL_BASE_LEVEL.get(name)
    if base is None:
        base = RiskLevel.WRITE
        reasons.extend(("unknown_tool", "unknown_tool_default_write"))
    else:
        reasons.append(f"base:{LEVEL_NAME[base]}")

    level = base

    if name == "execute_office_shell":
        shell_level, shell_reasons = _shell_signals(
            str(args.get("command", "")))
        reasons.extend(shell_reasons)
        level = max(level, shell_level)
    elif name == "write_office_file":
        path_reasons = _path_signals(args.get("filepath"), args.get("path"))
        if path_reasons:
            reasons.extend(path_reasons)
            level = RiskLevel.CRITICAL
        content = str(args.get("content", ""))
        mode = str(args.get("mode", "w") or "w").strip().casefold()
        if "content" in args and mode in {"w", "write", "overwrite"} and not content:
            reasons.append("write_empty_overwrite")
            level = RiskLevel.CRITICAL
        if len(content) >= 8000:
            reasons.append("write_large_payload")
            level = max(level, RiskLevel.MUTATE)
    elif name in {"read_office_file", "list_office_files"}:
        path_reasons = _path_signals(
            args.get("filepath"), args.get("path"), args.get("sub_dir")
        )
        if path_reasons:
            reasons.extend(path_reasons)
            level = RiskLevel.CRITICAL
    elif name in {"install_skill", "install_mcp_server"}:
        install_level, install_reasons = _installer_signals(args)
        reasons.extend(install_reasons)
        level = max(level, install_level)

    hard_deny = bool(cfg.hard_deny_critical and level >= RiskLevel.CRITICAL)
    if hard_deny:
        reasons.append("hard_deny_critical")

    return CallAssessment(
        name=name or "",
        args=args,
        level=level,
        reasons=tuple(reasons),
        hard_deny=hard_deny,
    )


def assess_tool_calls(
    tool_calls: Iterable[dict[str, Any]],  # 工具调用列表
    config: Optional[PolicyConfig] = None,  # 策略配置
    require_approval: bool = True,  # 是否需要审批
    approved_tools: Optional[set[str]] = None,  # 已授权工具集合
    approval_checker: Optional[ApprovalChecker] = None,  # 审批检查器
) -> PolicyDecision:
    """汇总工具调用，检查风险和审批需求。
    判断风险、写操作、人工审批、路径穿越、危险命令、网络请求等信号，返回策略决策。

    审批门：
    - hard_deny 优先：不审批、不执行
    - legacy_gate_tools 非空：只对名单内工具审批（兼容旧 approval_tools 替换语义）
    - approved_tools：已授权工具自动放行，但不改变 critical 硬拒绝
    - 否则按 max_level >= gate_level
    """
    cfg = config or PolicyConfig()
    collected: list[CallAssessment] = []

    # 遍历每个工具调用，评估风险等级和拒绝状态
    for tc in tool_calls:
        if not isinstance(tc, dict):
            collected.append(
                CallAssessment(
                    name="",
                    args={},
                    level=RiskLevel.WRITE,
                    reasons=("invalid_tool_call_type", "unknown_tool"),
                    hard_deny=False,
                )
            )
            continue
        collected.append(
            assess_call(tc.get("name", ""), _as_args(tc.get("args")), cfg)
        )

    # 汇总评估结果，计算最高风险等级、是否需要审批、是否硬拒绝，以及原因列表
    assessments = tuple(collected)
    if not assessments:
        return PolicyDecision(
            assessments=(),
            max_level=RiskLevel.READ,
            needs_approval=False,
            hard_deny=False,
            reasons=("empty_tool_calls",),
        )# 如果没有评估结果，则返回默认的策略决策

    max_level = max(a.level for a in assessments)  # 计算最高风险等级
    hard_deny = any(a.hard_deny for a in assessments)  # 是否有硬拒绝的调用
    reasons: list[str] = []  # 原因列表
    for a in assessments:
        reasons.extend(f"{a.name}:{r}" for r in a.reasons)

    approval_errors: list[str] = [] 

    def is_approved(assessment: CallAssessment) -> bool:
        """判断单次调用是否已有授权。"""

        if assessment.name in (approved_tools or set()):
            return True
        if approval_checker is None:
            return False
        try:
            return bool(approval_checker(assessment.name, assessment.args))
        except Exception as exc:
            approval_errors.append(type(exc).__name__)
            return False

    needs_approval = False
    # 硬拒绝时跳过审批
    if hard_deny:
        reasons.append("skip_approval_due_to_hard_deny")
    # 进入审批环节
    elif require_approval:
        if cfg.legacy_gate_tools is not None:
            pending = [
                assessment for assessment in assessments if assessment.name in cfg.legacy_gate_tools
                and not is_approved(assessment)
            ]
            needs_approval = bool(pending)
            if pending:
                reasons.append("legacy_gate:" + ",".join(a.name for a in pending))
            elif any(a.name in cfg.legacy_gate_tools for a in assessments):
                reasons.append("approved_call_grant")
        elif max_level >= cfg.gate_level:
            pending = [
                assessment for assessment in assessments
                if assessment.level >= cfg.gate_level and not is_approved(assessment)
            ]
            needs_approval = bool(pending)
            if pending:
                reasons.append(f"gate:{LEVEL_NAME[cfg.gate_level]}")
            else:
                reasons.append("approved_call_grant")

    if approval_errors:
        reasons.append("approval_checker_error:" + ",".join(approval_errors[:3]))

    return PolicyDecision(
        assessments=assessments,
        max_level=max_level,
        needs_approval=needs_approval,
        hard_deny=hard_deny,
        reasons=tuple(reasons),
    )


def render_policy_decision(decision: PolicyDecision) -> str:
    """将策略决策渲染为便于展示的文本。"""

    lines = [f"风险级别: {decision.max_level_name}"]
    if decision.hard_deny:
        lines.append("策略决策: 硬拒绝（不执行、不进入人工审批）")
    elif decision.needs_approval:
        lines.append("策略决策: 需要人工审批")
    else:
        lines.append("策略决策: 放行")
    for a in decision.assessments:
        arg_preview = ", ".join(
            f"{k}={str(v)[:40]}" for k, v in a.args.items()) or "无参数"
        why = ", ".join(a.reasons[:3])
        lines.append(f"  - {a.name}({arg_preview}) [{a.level_name}] {why}")
    return "\n".join(lines)


def build_policy_config(
    approval_tools: Optional[set[str]] = None,
    hard_deny_critical: bool = True,
    gate_level: RiskLevel = RiskLevel.WRITE,
    tool_base_overrides: Optional[dict[str, RiskLevel]] = None,
) -> PolicyConfig:
    """根据审批工具集合构建兼容旧配置的策略配置。"""

    legacy = frozenset(approval_tools) if approval_tools is not None else None
    return PolicyConfig(
        gate_level=gate_level,
        hard_deny_critical=hard_deny_critical,
        legacy_gate_tools=legacy,
        tool_base_overrides=dict(tool_base_overrides or {}),
    )
