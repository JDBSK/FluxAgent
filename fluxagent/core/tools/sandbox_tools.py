# -*- coding: utf-8 -*-
"""Office 沙箱工具，具有文件系统和子进程硬限制。

文件系统边界通过词法和解析路径共同强制执行，以防止前缀冲突、符号链接和 Windows 连接逃逸 OFFICE_DIR。
命令被解析为 argv 并使用 ``shell=False`` 执行；shell 语法、绝对路径、URL 和内联解释器代码在启动前被拒绝。

聚合指令 (``&&``, ``||``, ``|``, ``>``, ``>>``) 通过一个小型内置 mini-shell 进行解析和执行：
仅允许内置命令 (echo/dir/ls/cat/type/grep/findstr/mkdir/md) 参与，
因此外部解释器永远无法获得管道/重定向组合。外部允许的可执行文件仍然是单命令执行。
"""

from __future__ import annotations

import json
import ntpath
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import BinaryIO

from .base import fluxagent_tool
from ..config import OFFICE_DIR


SYS_OS                 = platform.system()
MAX_READ_BYTES         = 1 * 1024 * 1024
MAX_WRITE_BYTES        = 1 * 1024 * 1024
MAX_SHELL_OUTPUT_BYTES = 1 * 1024 * 1024
SHELL_TIMEOUT_SECONDS  = 60
MAX_DIRECTORY_ENTRIES  = 1000

_PATH_LOCKS_GUARD = threading.Lock()   # 互斥锁
_PATH_LOCKS: dict[str, threading.RLock] = {}  # 路径锁字典
_WORKSPACE_SHELL_LOCK = threading.RLock()  # 全局 shell 执行锁，避免并发执行导致的输出混乱


def _path_lock(path: str) -> threading.RLock:
    """创建或获取路径锁，确保同一文件/目录的并发访问安全。"""
    key = os.path.normcase(os.path.realpath(path))
    # 使用全局锁保护路径锁字典，避免并发创建冲突
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())

_ALLOWED_EXECUTABLES = frozenset(
    {
        "dir", "echo", "git", "ls", "node", "node.exe", "npm", "npm.cmd", "npx", "npx.cmd",
        "pytest", "pytest.exe",
        "python", "python.exe", "python3", "python3.exe", "uv", "uv.exe",
    }
)

# 允许的内置命令：语义完全由本模块实现，外部二进制不参与管道/重定向
_INLINE_CODE_FLAGS = {
    "python": {"-c", "-m"},
    "python.exe": {"-c", "-m"},
    "python3": {"-c", "-m"},
    "python3.exe": {"-c", "-m"},
    "node": {"-e", "--eval", "-p", "--print"},
    "node.exe": {"-e", "--eval", "-p", "--print"},
}

# 正则表达式预编译
_SHELL_META = re.compile(r"[\r\n|&;<>`]|\$\(|\$\{|%[^%]+%|\$env:|\^|!", re.IGNORECASE)
_URL_SCHEME = re.compile(r"(?i)\b(?:file|smb|ftp|https?)://")
_WINDOWS_DRIVE = re.compile(r"(?i)^[a-z]:[\\/]")


def _tool_error(code: str, message: str, *, retryable: bool = False) -> str:
    """返回一个稳定的 JSON 错误包，用于 ToolNode 和恢复逻辑."""
    return json.dumps(
        {
            "ok": False,
            "display": "❌ 权限拒绝" if code == "sandbox_violation" else "❌ 工具错误",
            "error": {"code": code, "message": str(message), "retryable": bool(retryable)},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _inside(base: str, candidate: str) -> bool:
    """返回是否候选路径在基础路径内，使用路径组件边界."""
    try:
        return os.path.commonpath([base, candidate]) == base
    except (ValueError, OSError):
        return False


def _get_safe_path(relative_path: str) -> str:
    """解析用户路径并拒绝遍历、符号链接或连接逃逸"""
    if not isinstance(relative_path, str):
        raise TypeError("路径必须是字符串。")
    if "\x00" in relative_path:
        raise PermissionError("路径包含 NUL 字符。")
    base_lexical = os.path.abspath(OFFICE_DIR)
    target_lexical = os.path.abspath(os.path.join(base_lexical, relative_path))
    if not _inside(os.path.normcase(base_lexical), os.path.normcase(target_lexical)):
        raise PermissionError(f"越权拦截：路径不在 office 内：{relative_path}")
    base_resolved = os.path.realpath(base_lexical)
    target_resolved = os.path.realpath(target_lexical)
    if not _inside(os.path.normcase(base_resolved), os.path.normcase(target_resolved)):
        raise PermissionError(f"越权拦截：路径通过链接指向 office 外：{relative_path}")
    return target_resolved


def _safe_argument_path(value: str) -> bool:
    # 如果参数是空的，或者是 URL，或者是以 - 开头的选项（但不包含路径穿越），或者是相对路径且不包含 .. 或 / 或 \，则认为是安全的；否则认为是不安全的。
    token = value.strip().strip("\"'")
    if not token:
        return True
    if _URL_SCHEME.search(token):
        return False
    if token.startswith("-"):
        # 选项可通过 --key=value 形式或附加形式（如 git -C../outside）来指定路径。
        # 验证数值，而非依赖于 shell 解析，避免绕过检查。
        if "=" in token:
            return _safe_argument_path(token.split("=", 1)[1])
        if ".." in token or "/" in token or "\\" in token:
            return False
        return True
    if (os.path.isabs(token) or ntpath.isabs(token) or _WINDOWS_DRIVE.match(token)
            or token.startswith(("\\\\", "//", "~"))):
        return False
    parts = token.replace("\\", "/").split("/")
    if ".." in parts:
        return False
    if "/" in token or "\\" in token:
        try:
            _get_safe_path(token)
        except (OSError, TypeError, ValueError):
            return False
    return True


def _parse_command(command: str) -> list[str]:
    # 大白话: 解析受限制的命令为 argv，而不调用 shell。
    text = str(command or "").strip()
    if not text:
        raise ValueError("命令不能为空。")
    if _SHELL_META.search(text):
        raise PermissionError("禁止 Shell 链式、重定向、变量展开或命令替换语法。")
    if _URL_SCHEME.search(text):
        raise PermissionError("Shell 工具不允许直接访问 URL。")
    try:
        raw_args = shlex.split(text, posix=(os.name != "nt"))
    except ValueError as exc:
        raise ValueError(f"命令引号不完整：{exc}") from exc
    args = [
        arg[1:-1] if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in "\"'" else arg
        for arg in raw_args
    ]
    if not args:
        raise ValueError("命令不能为空。")
    # 命令名不得携带路径前缀（./python、/usr/bin/ls、bin\\ls 等）：
    # basename 归一化会把它们伪装成白名单命令，必须原样拒绝。
    if any(separator in args[0] for separator in ("/", "\\")) or ":" in args[0]:
        raise PermissionError("命令名不得携带路径前缀")
    executable = ntpath.basename(args[0]).casefold()
    if executable not in _ALLOWED_EXECUTABLES:
        raise PermissionError(f"命令不在允许列表中：{args[0]}")
    forbidden_flags = _INLINE_CODE_FLAGS.get(executable, set())
    if any(arg.casefold() in forbidden_flags for arg in args[1:]):
        raise PermissionError("禁止通过解释器参数执行内联代码或任意模块。")
    if any(not _safe_argument_path(arg) for arg in args[1:]):
        raise PermissionError("命令包含 office 外路径、URL 或路径穿越。")
    if executable in {"echo", "dir", "ls"}:
        args[0] = f"__fluxagent_builtin_{executable}__"
        return args
    resolved = shutil.which(args[0])
    if not resolved:
        raise FileNotFoundError(f"找不到允许的可执行文件：{args[0]}")
    args[0] = resolved
    return args


# ============ 复合命令 mini-shell（仅内置命令可参与） ============

# 复合命令允许的内置命令：语义完全由本模块实现，外部二进制不参与管道/重定向
_COMPOUND_BUILTINS = frozenset(
    {"echo", "dir", "ls", "cat", "type", "grep", "findstr", "mkdir", "md"}
)

# 这些字符在 token 层一律封死：`;` `<` 反引号 `$` `^` `!` 与换行。
# 解释权在本模块而非系统 shell，直接拒绝可避免"展开后检查"窗口。
_TOKEN_FORBIDDEN = re.compile(r"[;<`$^!\r\n]")
# cmd 风格 %VAR% 成对展开（孤立 % 不算变量）
_CMDVAR_PATTERN = re.compile(r"%[^%\s]{1,}%")


def _tokenize_compound(text: str) -> list[tuple[str, str]]:
    """把命令文本切成 word/op token；引号内不识别运算符。

    Returns:
        token 列表，元素为 (kind, value)，kind 为 "word" 或 "op"，
        op 取值 "&&" "||" "|" ">" ">>"。

    Raises:
        PermissionError / ValueError: 出现被封死的字符、引号不完整或非法运算符。
    """
    if _CMDVAR_PATTERN.search(text):
        raise PermissionError("检测到 cmd 变量展开语法（%VAR%），已封禁。")
    tokens: list[tuple[str, str]] = []
    i, length = 0, len(text)
    while i < length:
        char = text[i]
        if char.isspace():
            i += 1
            continue
        if _TOKEN_FORBIDDEN.match(char):
            raise PermissionError(f"禁止的字符或语法：{char!r}。")
        if char in "&|>":
            if text.startswith("&&", i):
                op = "&&"
            elif text.startswith("||", i):
                op = "||"
            elif text.startswith(">>", i):
                op = ">>"
            elif char in "|>":
                op = char
            else:
                raise PermissionError("不允许单个 & 字符。")
            end = i + len(op)
            if end < length and text[end] in "&|>":
                raise PermissionError("无效的运算符组合。")
            tokens.append(("op", op))
            i = end
            continue
        parts: list[str] = []
        while i < length and not text[i].isspace() and text[i] not in "&|>":
            char = text[i]
            if char in "\"'":
                close = text.find(char, i + 1)
                if close == -1:
                    raise ValueError(f"命令引号不完整：{text[:40]}")
                parts.append(text[i + 1:close])
                i = close + 1
                continue
            if _TOKEN_FORBIDDEN.match(char):
                raise PermissionError(f"禁止的字符或语法：{char!r}。")
            parts.append(char)
            i += 1
        tokens.append(("word", "".join(parts)))
    return tokens


@dataclass
class _Stage:
    name: str
    args: list[str]


@dataclass
class _CompoundCommand:
    connector: str            # 与上一条命令的关系："", "&&", "||"
    stages: list[_Stage]      # 用 | 串起来的管道段
    redirect: tuple[str, str] | None = None  # (">" 或 ">>", 目标文件)


def _plan_compound(tokens: list[tuple[str, str]]) -> list[_CompoundCommand]:
    """把 token 流组装成复合命令计划，并完成白名单与参数校验。"""
    commands: list[_CompoundCommand] = []
    connector = ""
    stages: list[_Stage] = []
    current: list[str] = []
    redirect: tuple[str, str] | None = None
    pending_redirect: str | None = None

    def close_stage() -> None:
        nonlocal current
        if not current:
            raise ValueError("复合命令中存在空的命令段。")
        head = current[0].casefold()
        if head not in _COMPOUND_BUILTINS:
            raise PermissionError(
                f"复合命令仅允许内置命令 {sorted(_COMPOUND_BUILTINS)}，"
                f"不允许：{current[0]}。外部命令必须单独执行。"
            )
        for arg in current[1:]:
            if not _safe_argument_path(arg):
                raise PermissionError(f"参数越界：{arg}。所有路径必须限制在 office 工位内。")
        stages.append(_Stage(head, list(current[1:])))
        current = []

    def close_command() -> None:
        nonlocal stages, redirect, connector
        close_stage()
        commands.append(_CompoundCommand(connector, stages, redirect))
        stages = []
        redirect = None
        connector = ""

    for kind, value in tokens:
        if kind == "word":
            if pending_redirect is not None:
                if redirect is not None:
                    raise ValueError("一条命令只允许一个输出重定向。")
                redirect = (pending_redirect, value)
                pending_redirect = None
                continue
            current.append(value)
            continue
        if pending_redirect is not None:
            raise ValueError("重定向缺少目标文件。")
        if value in ("&&", "||"):
            close_command()
            connector = value
        elif value == "|":
            if redirect is not None:
                raise ValueError("重定向只能出现在管道末段之后。")
            close_stage()
        else:  # ">" 或 ">>"
            if redirect is not None:
                raise ValueError("一条命令只允许一个输出重定向。")
            pending_redirect = value
    if pending_redirect is not None:
        raise ValueError("重定向缺少目标文件。")
    close_command()

    if commands and commands[0].connector:
        raise ValueError("复合命令不能以运算符开头。")
    for command in commands:
        if command.redirect is not None and not _safe_argument_path(command.redirect[1]):
            raise PermissionError(
                f"重定向目标越界：{command.redirect[1]}。输出必须落在 office 工位内。"
            )
    return commands


def _read_stage_files(paths: list[str]) -> tuple[list[str], list[str]]:
    """批量读取 office 内文件行，返回 (lines, errors)；单文件失败不中断其余。"""
    lines: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            resolved = _get_safe_path(path)
            if not os.path.isfile(resolved):
                errors.append(f"文件不存在：{path}")
                continue
            if os.path.getsize(resolved) > MAX_READ_BYTES:
                errors.append(f"文件超过 {MAX_READ_BYTES} 字节读取限制：{path}")
                continue
            with open(resolved, "r", encoding="utf-8", errors="replace") as handle:
                lines.extend(handle.read(MAX_READ_BYTES).splitlines())
        except (PermissionError, OSError) as exc:
            errors.append(str(exc))
    return lines, errors


def _run_builtin_stage(stage: _Stage, stdin_text: str | None) -> tuple[int, str, str]:
    """执行一个内置命令段，返回 (returncode, stdout, stderr)；不抛异常。"""
    name, args = stage.name, stage.args
    try:
        if name == "echo":
            return 0, " ".join(args) + "\n", ""
        if name in ("dir", "ls"):
            relative = args[0] if args else "."
            target = _get_safe_path(relative)
            if not os.path.isdir(target):
                return 1, "", f"不是目录：{relative}"
            return 0, "\n".join(sorted(os.listdir(target))) + "\n", ""
        if name in ("cat", "type"):
            if not args:
                if stdin_text is None:
                    return 1, "", "cat: 缺少文件参数且无管道输入。"
                return 0, stdin_text, ""
            lines, errors = _read_stage_files(args)
            body = "".join(line + "\n" for line in lines)
            return (1 if errors and not lines else 0), body, "\n".join(errors)
        if name in ("grep", "findstr"):
            if not args:
                return 2, "", "grep: 缺少 pattern 参数。"
            pattern, files = args[0], args[1:]
            if files:
                lines, errors = _read_stage_files(files)
                if errors and not lines:
                    return 1, "", "\n".join(errors)
            else:
                if stdin_text is None:
                    return 2, "", "grep: 缺少文件参数且无管道输入。"
                lines, errors = stdin_text.splitlines(), []
            matched = [line for line in lines if pattern in line]
            body = "".join(line + "\n" for line in matched)
            return (0 if matched else 1), body, "\n".join(errors)
        if name in ("mkdir", "md"):
            if len(args) != 1:
                return 1, "", "mkdir: 需要恰好一个目录参数。"
            target = _get_safe_path(args[0])
            os.mkdir(target)
            return 0, "", ""
        return 127, "", f"未知内置命令：{name}"
    except FileExistsError:
        return 1, "", f"mkdir: 目录已存在：{args[0] if args else ''}"
    except (PermissionError, OSError, ValueError) as exc:
        return 1, "", str(exc)


def _run_compound(commands: list[_CompoundCommand]) -> tuple[int, str, str]:
    """按 && / || / 管道 / 重定向语义执行内置命令计划。"""
    stderr_parts: list[str] = []
    last_rc = 0
    final_stdout = ""
    for command in commands:
        if command.connector == "&&" and last_rc != 0:
            continue
        if command.connector == "||" and last_rc == 0:
            continue
        stdin_text: str | None = None
        for stage in command.stages:
            returncode, stdout, stderr = _run_builtin_stage(stage, stdin_text)
            # 管道内 stdin 始终是字符串：上一段输出为空时是空管道（EOF），不是"无管道"
            stdin_text = (stdout or "")[:MAX_SHELL_OUTPUT_BYTES]
            if stderr:
                stderr_parts.append(stderr)
            last_rc = returncode
        if command.redirect is not None:
            mode, target = command.redirect
            resolved = _get_safe_path(target)
            os.makedirs(os.path.dirname(resolved), exist_ok=True)
            with open(resolved, "w" if mode == ">" else "a", encoding="utf-8") as handle:
                handle.write(stdin_text or "")
            final_stdout = f"(stdout 已重定向{'追加' if mode == '>>' else '写入'} {target})"
        else:
            final_stdout = stdin_text or ""
    return last_rc, final_stdout, "\n".join(stderr_parts)


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """终止给定的子进程及其所有后代进程，适用于 Windows 或 POSIX 系统."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                shell=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.wait(timeout=2)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _read_limited(
    stream: BinaryIO | None,
    chunks: list[bytes],
    overflow: threading.Event,
    total: list[int],
    lock: threading.Lock,
) -> None:
    """读取一个进程流直到 EOF 或共享输出上限被超过."""
    if stream is None:
        return
    try:
        while not overflow.is_set():
            chunk = stream.read(4096)
            if not chunk:
                break
            with lock:
                remaining = MAX_SHELL_OUTPUT_BYTES - total[0]
                if remaining <= 0:
                    overflow.set()
                    break
                accepted = chunk[:remaining]
                chunks.append(accepted)
                total[0] += len(accepted)
                if len(chunk) > remaining:
                    overflow.set()
                    break
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _run_command(args: list[str]) -> tuple[int, str, str]:
    """运行一个允许的 argv 命令，但不使用命令行 shell."""
    if args and args[0].startswith("__fluxagent_builtin_"):
        name = args[0].removeprefix("__fluxagent_builtin_").removesuffix("__")
        if name == "echo":
            return 0, " ".join(args[1:]) + "\n", ""
        relative = args[1] if len(args) > 1 else "."
        target = _get_safe_path(relative)
        if not os.path.isdir(target):
            return 1, "", f"不是目录：{relative}"
        return 0, "\n".join(sorted(os.listdir(target))) + "\n", ""
    popen_kwargs: dict[str, object] = {
        "cwd": OFFICE_DIR, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        "stdin": subprocess.DEVNULL, "shell": False,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(args, **popen_kwargs)
    stdout_parts: list[bytes] = []
    stderr_parts: list[bytes] = []
    overflow = threading.Event()
    output_total = [0]
    output_lock = threading.Lock()
    readers = [
        threading.Thread(
            target=_read_limited,
            args=(process.stdout, stdout_parts,
                  overflow, output_total, output_lock),
            daemon=True,
        ),
        threading.Thread(
            target=_read_limited,
            args=(process.stderr, stderr_parts,
                  overflow, output_total, output_lock),
            daemon=True,
        ),
    ]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + SHELL_TIMEOUT_SECONDS
    while process.poll() is None:
        if overflow.is_set():
            _terminate_process_tree(process)
            raise OverflowError(f"命令输出超过 {MAX_SHELL_OUTPUT_BYTES} 字节限制。")
        if time.monotonic() >= deadline:
            _terminate_process_tree(process)
            raise subprocess.TimeoutExpired(args, SHELL_TIMEOUT_SECONDS)
        time.sleep(0.02)
    for reader in readers:
        reader.join(timeout=1)
    stdout = b"".join(stdout_parts).decode("utf-8", errors="replace")
    stderr = b"".join(stderr_parts).decode("utf-8", errors="replace")
    return int(process.returncode or 0), stdout, stderr


@fluxagent_tool
def list_office_files(sub_dir: str = "") -> str:
    """列出office根目录下的文件，不跟随逃逸链接."""
    try:
        target_dir = _get_safe_path(sub_dir)
        if not os.path.exists(target_dir):
            return _tool_error("not_found", f"目录不存在：{sub_dir}")
        if not os.path.isdir(target_dir):
            return _tool_error("not_directory", f"不是目录：{sub_dir}")
        items = os.listdir(target_dir)
        if not items:
            return f"[{sub_dir if sub_dir else 'office 根目录'}] 是空的。"
        visible = items[:MAX_DIRECTORY_ENTRIES]
        result = []
        for item in visible:
            item_path = _get_safe_path(os.path.join(sub_dir, item))
            item_type = "[DIR]" if os.path.isdir(item_path) else "[FILE]"
            result.append(f"{item_type} {item}")
        if len(items) > len(visible):
            result.append(f"...[仅显示前 {MAX_DIRECTORY_ENTRIES} 项]")
        return "\n".join(result)
    except PermissionError as exc:
        return _tool_error("sandbox_violation", str(exc))
    except OSError as exc:
        return _tool_error("filesystem_error", str(exc), retryable=True)


@fluxagent_tool
def read_office_file(filepath: str) -> str:
    """从office根目录内的普通文件中读取最多1 MiB的数据."""
    try:
        target_path = _get_safe_path(filepath)
        if not os.path.exists(target_path):
            return _tool_error("not_found", f"文件不存在：{filepath}")
        if not os.path.isfile(target_path):
            return _tool_error("not_file", f"不是普通文件：{filepath}")
        if os.path.getsize(target_path) > MAX_READ_BYTES:
            return _tool_error("file_too_large", f"文件超过 {MAX_READ_BYTES} 字节读取限制。")
        with open(target_path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(MAX_READ_BYTES + 1)[:MAX_READ_BYTES]
    except PermissionError as exc:
        return _tool_error("sandbox_violation", str(exc))
    except OSError as exc:
        return _tool_error("filesystem_error", str(exc), retryable=True)


@fluxagent_tool
def write_office_file(filepath: str, content: str, mode: str = "w") -> str:
    """创建、替换或追加一个受限制的 UTF-8 文件在office根目录内."""
    try:
        if mode not in {"w", "a"}:
            return _tool_error("invalid_mode", "mode 参数必须是 'w' 或 'a'。")
        encoded = str(content).encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            return _tool_error("payload_too_large", f"单次写入超过 {MAX_WRITE_BYTES} 字节限制。")

        # --------- 使用获取锁保证路径安全，避免并发写入冲突 ---------
        target_path = _get_safe_path(filepath)
        lock = _path_lock(target_path)  # 获取路径锁，确保同一文件的并发写入安全
        # with clock = lock.acquire(blocking=True)阻塞等待,退出with块自动释放锁
        with lock:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            target_path = _get_safe_path(filepath)
            # 检查文件大小限制
            if mode == "a" and os.path.exists(target_path):
                if os.path.getsize(target_path) + len(encoded) > MAX_WRITE_BYTES:
                    return _tool_error("file_too_large", f"追加后文件将超过 {MAX_WRITE_BYTES} 字节限制。")
            # 写入文件
            if mode == "a":
                with open(target_path, "a", encoding="utf-8") as handle:
                    if content and not content.startswith("\n"):
                        handle.write("\n")
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
            # 使用原子替换写入新文件，避免部分写入
            else:
                temp_name = ""
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=os.path.dirname(target_path),
                        prefix=f".{os.path.basename(target_path)}.",
                        suffix=".tmp",
                        delete=False,
                    ) as handle:
                        temp_name = handle.name
                        handle.write(content)
                        handle.flush()
                        os.fsync(handle.fileno())  # 确保数据写入磁盘

                    # 原子替换目标文件
                    os.replace(temp_name, target_path)
                    temp_name = ""
                finally:
                    if temp_name:
                        try:
                            os.unlink(temp_name)  # 确保临时文件被删除
                        except FileNotFoundError:
                            pass

        # --------- 写入成功，返回提示信息 ---------
        action = "覆盖/新建" if mode == "w" else "追加"
        return f" ● 成功以 {action} 模式写入文件：{filepath} (共 {len(content)} 字符)"
    except PermissionError as exc:
        return _tool_error("sandbox_violation", str(exc))
    except OSError as exc:
        return _tool_error("filesystem_error", str(exc), retryable=True)


def _has_unquoted_operator(text: str) -> bool:
    """返回是否 & | > 出现在引号外的命令文本中."""
    quote: str | None = None
    for char in text:
        if quote:
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
        elif char in "&|>":
            return True
    return False


def _uses_compound_path(command: str) -> bool:
    """复合路径判定：含未加引号的运算符，或首词是 mini-shell 内置命令。

    内置命令（echo/dir/ls/cat/type/grep/findstr/mkdir/md）的语义完全由本模块
    实现，统一走复合执行器；其余命令保持原 argv 白名单单命令路径。
    """
    stripped = str(command or "").strip()
    if not stripped:
        return False
    head = stripped.split(None, 1)[0].strip("\"'").casefold()
    return head in _COMPOUND_BUILTINS or _has_unquoted_operator(stripped)


@fluxagent_tool
def execute_office_shell(command: str) -> str:
    """运行一个允许的 argv 命令，但不使用命令行 shell."""
    try:
        with _WORKSPACE_SHELL_LOCK:
            if _uses_compound_path(command):
                plan = _plan_compound(_tokenize_compound(command))
                returncode, stdout, stderr = _run_compound(plan)
            else:
                args = _parse_command(command)
                returncode, stdout, stderr = _run_command(args)
        if returncode != 0:
            return _tool_error("command_failed", f"命令退出码 {returncode}：{(stderr or stdout)[-1000:]}")
        output = (
            f" ● 当前系统: {SYS_OS}\n ● 执行命令: `{command}`\n"
            f" ● 退出码 (Exit Code): {returncode}"
        )
        if stdout:
            output += f"\n[STDOUT]\n{stdout[-4000:]}"
        if stderr:
            output += f"\n[STDERR]\n{stderr[-4000:]}"
        if not stdout and not stderr:
            output += "\n(静默执行完毕：无终端输出)"
        return output
    except PermissionError as exc:
        return _tool_error("sandbox_violation", str(exc))
    except FileNotFoundError as exc:
        return _tool_error("command_not_found", str(exc))
    except ValueError as exc:
        return _tool_error("invalid_command", str(exc))
    except subprocess.TimeoutExpired:
        return _tool_error("command_timeout", f"命令执行超过 {SHELL_TIMEOUT_SECONDS} 秒，已终止进程树。", retryable=True)
    except OverflowError as exc:
        return _tool_error("output_limit", str(exc))
    except (OSError, subprocess.SubprocessError) as exc:
        return _tool_error("command_error", str(exc), retryable=True)
