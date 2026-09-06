# -*- coding: utf-8 -*-
"""Office sandbox tools with filesystem and subprocess hard limits.

The filesystem boundary is enforced with both lexical and resolved paths so
prefix collisions, symlinks, and Windows junctions cannot escape OFFICE_DIR.
Commands are parsed into argv and executed with ``shell=False``; shell syntax,
absolute paths, URLs, and inline interpreter code are rejected before launch.
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
import threading
import time
from typing import BinaryIO

from .base import fluxagent_tool
from ..config import OFFICE_DIR


SYS_OS = platform.system()
MAX_READ_BYTES = 1 * 1024 * 1024
MAX_WRITE_BYTES = 1 * 1024 * 1024
MAX_SHELL_OUTPUT_BYTES = 1 * 1024 * 1024
SHELL_TIMEOUT_SECONDS = 60
MAX_DIRECTORY_ENTRIES = 1000

_ALLOWED_EXECUTABLES = frozenset(
    {
        "dir", "echo", "git", "ls", "node", "node.exe", "npm", "npm.cmd", "npx", "npx.cmd",
        "pip", "pip.exe", "pip3", "pip3.exe", "pytest", "pytest.exe",
        "python", "python.exe", "python3", "python3.exe", "uv", "uv.exe",
    }
)
_INLINE_CODE_FLAGS = {
    "python": {"-c", "-m"},
    "python.exe": {"-c", "-m"},
    "python3": {"-c", "-m"},
    "python3.exe": {"-c", "-m"},
    "node": {"-e", "--eval", "-p", "--print"},
    "node.exe": {"-e", "--eval", "-p", "--print"},
}
_SHELL_META = re.compile(
    r"[\r\n|&;<>`]|\$\(|\$\{|%[^%]+%|\$env:|\^|!", re.IGNORECASE)
_URL_SCHEME = re.compile(r"(?i)\b(?:file|smb|ftp|https?)://")
_WINDOWS_DRIVE = re.compile(r"(?i)^[a-z]:[\\/]")


def _tool_error(code: str, message: str, *, retryable: bool = False) -> str:
    """Return a stable JSON error envelope for ToolNode and recovery logic."""
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
    """Return whether candidate is inside base using path-component boundaries."""
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
        # Options can carry paths in --key=value or attached forms such as
        # git -C../outside. Validate the value instead of trusting the dash.
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


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Terminate the subprocess and all descendants on Windows or POSIX."""
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
    """Read one process stream until EOF or the shared output cap is exceeded."""
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
    """Run argv with timeout, bounded output, and process-tree cleanup."""
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
    """List files below the office root without following an escaping link."""
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
    """Read at most one MiB from a regular file inside the office root."""
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
    """Create, replace, or append a bounded UTF-8 file inside the office root."""
    try:
        if mode not in {"w", "a"}:
            return _tool_error("invalid_mode", "mode 参数必须是 'w' 或 'a'。")
        encoded = str(content).encode("utf-8")
        if len(encoded) > MAX_WRITE_BYTES:
            return _tool_error("payload_too_large", f"单次写入超过 {MAX_WRITE_BYTES} 字节限制。")
        target_path = _get_safe_path(filepath)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        target_path = _get_safe_path(filepath)
        if mode == "a" and os.path.exists(target_path):
            if os.path.getsize(target_path) + len(encoded) > MAX_WRITE_BYTES:
                return _tool_error("file_too_large", f"追加后文件将超过 {MAX_WRITE_BYTES} 字节限制。")
        with open(target_path, mode, encoding="utf-8") as handle:
            if mode == "a" and content and not content.startswith("\n"):
                handle.write("\n")
            handle.write(content)
        action = "覆盖/新建" if mode == "w" else "追加"
        return f" ● 成功以 {action} 模式写入文件：{filepath} (共 {len(content)} 字符)"
    except PermissionError as exc:
        return _tool_error("sandbox_violation", str(exc))
    except OSError as exc:
        return _tool_error("filesystem_error", str(exc), retryable=True)


@fluxagent_tool
def execute_office_shell(command: str) -> str:
    """Run one allow-listed argv command inside office without a command shell."""
    try:
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
