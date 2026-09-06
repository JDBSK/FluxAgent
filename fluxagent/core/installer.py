from __future__ import annotations

# -*- coding: utf-8 -*-
"""受控的 Skill/MCP 安装器。

安装器专门处理远程 GitHub Clone 与本地目录/ZIP 导入，不能通过通用
execute_office_shell 绕过网络与路径策略。远程来源默认只接受 GitHub HTTPS
仓库；本地导入会拒绝符号链接和压缩包路径穿越。

MCP 安装会识别常见 Node/Python 项目、准备隔离依赖、执行必要构建并完成
协议握手，验证成功后才登记。第三方安装和构建脚本仍受 EXEC 审批保护。
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .config import MCP_SERVERS_DIR, MCP_SERVERS_INDEX, SKILLS_DIR
from .logger import audit_logger


MAX_FILES = 2_000
MAX_TOTAL_BYTES = 100 * 1024 * 1024
CLONE_TIMEOUT_SECONDS = 120
SETUP_TIMEOUT_SECONDS = 300
MCP_PROBE_TIMEOUT_SECONDS = 20
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_MCP_INDEX_LOCK = threading.RLock()
_INSTALL_TREE_LOCK = threading.RLock()


class InstallError(ValueError):
    """安装请求不满足来源、路径或内容校验时抛出。"""


def _run_setup(command: list[str], cwd: Path, label: str) -> None:
    """Run one fixed setup command and preserve its useful failure output."""
    executable = shutil.which(command[0])
    if executable is None:
        raise InstallError(f"{label}需要命令 {command[0]}，但当前环境未找到。")
    resolved_command = [executable, *command[1:]]
    try:
        result = subprocess.run(
            resolved_command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SETUP_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise InstallError(f"{label}无法启动命令 {executable}。") from exc
    except subprocess.TimeoutExpired as exc:
        raise InstallError(f"{label}超时（{SETUP_TIMEOUT_SECONDS} 秒）。") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "未知错误").strip()[-1_200:]
        raise InstallError(f"{label}失败: {detail}")


def _node_entry(root: Path, package: dict) -> str:
    candidates: list[str] = []
    main = package.get("main")
    if isinstance(main, str):
        candidates.append(main)
    binary = package.get("bin")
    if isinstance(binary, str):
        candidates.append(binary)
    elif isinstance(binary, dict):
        candidates.extend(value for value in binary.values() if isinstance(value, str))
    candidates.extend(("dist/index.js", "build/index.js", "index.js"))
    for candidate in dict.fromkeys(candidates):
        normalized = candidate.removeprefix("./").replace("\\", "/")
        if (root / normalized).is_file():
            return normalized
    return candidates[0].removeprefix("./").replace("\\", "/") if candidates else "dist/index.js"


def _prepare_node_mcp(root: Path) -> tuple[str, list[str], str]:
    # 自动识别 Node MCP 项目，安装依赖并构建，返回可执行命令和参数。
    try:
        package = json.loads((root / "package.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError("Node MCP 的 package.json 无法解析。") from exc
    if not isinstance(package, dict):
        raise InstallError("Node MCP 的 package.json 必须是 JSON 对象。")

    if (root / "package-lock.json").is_file():
        # Respect the repository lockfile. Floating caret dependencies can become
        # source-incompatible with older MCP server code.
        install_command = ["npm", "ci", "--no-audit", "--no-fund"]
    else:
        install_command = [
            "npm",
            "install",
            "--no-audit",
            "--no-fund",
            "--no-package-lock",
        ]
    try:
        _run_setup(install_command, root, "Node 依赖安装")
    except InstallError as exc:
        detail = str(exc)
        if "edgesOut" not in detail and "ERESOLVE" not in detail:
            raise
        _run_setup(
            [*install_command, "--legacy-peer-deps"],
            root,
            "Node 依赖兼容模式安装",
        )
    entry = _node_entry(root, package)
    scripts = package.get("scripts")
    if not (root / entry).is_file() and isinstance(scripts, dict) and scripts.get("build"):
        _run_setup(["npm", "run", "build"], root, "Node MCP 构建")
        entry = _node_entry(root, package)
    if not (root / entry).is_file():
        raise InstallError(
            f"Node MCP 启动入口不存在: {entry}。请检查 package.json 的 main/bin/build。"
        )
    return "node", [entry], "Node/npm"


def _prepare_python_mcp(root: Path) -> tuple[str, list[str], str]:
    pyproject_path = root / "pyproject.toml"
    required_python = ""
    if pyproject_path.is_file():
        try:
            project_text = pyproject_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            project_text = ""
        match = re.search(r"requires-python\s*=\s*[\"']>=\s*(\d+)\.(\d+)", project_text)
        if match:
            required_python = f"{match.group(1)}.{match.group(2)}"
            current = (sys.version_info.major, sys.version_info.minor)
            minimum = (int(match.group(1)), int(match.group(2)))
            if current < minimum and os.name == "nt" and shutil.which("py"):
                python_launcher = ["py", f"-{required_python}"]
            elif current < minimum:
                raise InstallError(
                    f"MCP 要求 Python >= {required_python}，当前环境为 "
                    f"{sys.version_info.major}.{sys.version_info.minor}，且未找到可用解释器。"
                )
            else:
                python_launcher = [sys.executable]
        else:
            python_launcher = [sys.executable]
        if os.name == "nt" and re.search(r"manylinux|musllinux", project_text, re.I):
            raise InstallError(
                "该 Python MCP 的依赖清单引用 Linux 专用 wheel（manylinux/musllinux），"
                "当前 Windows 环境无法直接安装；请使用 Linux/WSL，或提供 Windows 兼容依赖。"
            )
    else:
        python_launcher = [sys.executable]

    venv_dir = root / ".venv"
    _run_setup([*python_launcher, "-m", "venv", str(venv_dir)], root, "Python 虚拟环境创建")
    python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if (root / "pyproject.toml").is_file() or (root / "setup.py").is_file():
        _run_setup([str(python), "-m", "pip", "install", "."], root, "Python 项目依赖安装")
    elif (root / "requirements.txt").is_file():
        _run_setup(
            [str(python), "-m", "pip", "install", "-r", "requirements.txt"],
            root,
            "Python 项目依赖安装",
        )

    entry = next((name for name in ("server.py", "main.py") if (root / name).is_file()), "")
    if not entry:
        raise InstallError("已识别为 Python 项目，但未找到 server.py 或 main.py 启动入口。")
    args = [entry]
    try:
        entry_text = (root / entry).read_text(encoding="utf-8", errors="replace")
    except OSError:
        entry_text = ""
    if "--transport" in entry_text and "stdio" in entry_text:
        args.extend(["--transport", "stdio"])
    return str(python), args, "Python/venv"


def _prepare_mcp(root: Path, command: str, args: list[str]) -> tuple[str, list[str], str]:
    """Prefer repository manifests over an LLM-supplied launch guess."""
    if (root / "package.json").is_file():
        return _prepare_node_mcp(root)
    python_markers = (
        "pyproject.toml",
        "setup.py",
        "requirements.txt",
        "server.py",
        "main.py",
    )
    if any((root / name).is_file() for name in python_markers):
        return _prepare_python_mcp(root)
    if command:
        return command, args, "用户配置"
    raise InstallError(
        "无法识别 MCP 运行时：未找到 package.json 或 Python 项目清单/入口。"
    )


def _probe_mcp(record: dict) -> list[str]:
    """Complete an MCP initialize/list_tools handshake before registry commit."""
    from ..mcp.client import _discover_server, _exception_detail

    try:
        tools = asyncio.run(
            asyncio.wait_for(
                _discover_server(record), timeout=MCP_PROBE_TIMEOUT_SECONDS
            )
        )
    except Exception as exc:
        raise InstallError(f"MCP 启动检查失败: {_exception_detail(exc)}") from exc
    return [str(tool.name) for tool in tools]


@dataclass(frozen=True)
class InstalledMCPServer:
    """登记到本地索引的外部 MCP Server 信息。"""

    name: str
    source_type: str
    source: str
    installed_path: str
    command: str
    args: list[str]
    read_only_tools: list[str]
    installed_at: str


def _allowed_git_hosts() -> set[str]:
    raw = os.getenv("FLUXAGENT_ALLOWED_GIT_HOSTS", "github.com")
    return {host.strip().lower() for host in raw.split(",") if host.strip()}


def _safe_name(name: str, field_name: str = "name") -> str:
    value = (name or "").strip()
    if not _NAME_PATTERN.fullmatch(value):
        raise InstallError(
            f"{field_name} 只能由字母、数字、下划线和连字符组成，且长度为 1-64。"
        )
    return value


def _inside(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise InstallError("检测到路径越界。") from exc
    return candidate_resolved


def _safe_subdirectory(root: Path, subdirectory: str) -> Path:
    if not subdirectory:
        return root
    return _inside(root, root / subdirectory)


def _validate_git_source(source: str) -> str:
    parsed = urlparse((source or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _allowed_git_hosts():
        allowed = ", ".join(sorted(_allowed_git_hosts())) or "无"
        raise InstallError(f"远程安装仅允许 HTTPS Git 来源。允许主机: {allowed}。")
    if parsed.query or parsed.fragment:
        raise InstallError("Git 仓库地址不能携带 query 或 fragment。")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise InstallError("Git 地址必须包含组织名和仓库名。")
    return source.strip()


def _safe_extract_zip(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES:
            raise InstallError(f"压缩包文件数超过限制 ({MAX_FILES})。")
        total_size = sum(info.file_size for info in infos)
        if total_size > MAX_TOTAL_BYTES:
            raise InstallError("压缩包解压后的总大小超过 100 MB 限制。")

        for info in infos:
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise InstallError("压缩包包含路径穿越条目。")
            target = _inside(target_dir, target_dir / relative)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source_file, open(target, "wb") as target_file:
                shutil.copyfileobj(source_file, target_file)


def _copy_tree(source_dir: Path, target_dir: Path) -> None:
    file_count = 0
    total_bytes = 0
    source_dir = source_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    for current_root, directory_names, file_names in os.walk(source_dir, followlinks=False):
        current_path = Path(current_root)
        directory_names[:] = [
            name
            for name in directory_names
            if name != ".git" and not (current_path / name).is_symlink()
        ]
        relative_root = current_path.relative_to(source_dir)
        destination_root = _inside(target_dir, target_dir / relative_root)
        destination_root.mkdir(parents=True, exist_ok=True)

        for file_name in file_names:
            source_file = current_path / file_name
            if source_file.is_symlink():
                raise InstallError("安装来源包含符号链接，已拒绝导入。")
            file_count += 1
            total_bytes += source_file.stat().st_size
            if file_count > MAX_FILES:
                raise InstallError(f"文件数超过限制 ({MAX_FILES})。")
            if total_bytes > MAX_TOTAL_BYTES:
                raise InstallError("安装来源总大小超过 100 MB 限制。")
            shutil.copy2(source_file, destination_root / file_name)


def _materialize_source(source_type: str, source: str, work_dir: Path) -> Path:
    kind = (source_type or "").strip().lower()
    if kind == "git":
        git_source = _validate_git_source(source)
        checkout_dir = work_dir / "checkout"
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", git_source, str(checkout_dir)],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CLONE_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            raise InstallError("未找到 git 命令，请先安装 Git 或改用本地导入。") from exc
        except subprocess.TimeoutExpired as exc:
            raise InstallError("Git Clone 超时（120 秒），请检查网络后重试。") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "未知 Git 错误").strip()[-500:]
            raise InstallError(f"Git Clone 失败: {detail}")
        return checkout_dir

    if kind == "local":
        local_source = Path(os.path.expandvars(os.path.expanduser(source or ""))).resolve()
        if local_source.is_dir():
            return local_source
        if local_source.is_file() and local_source.suffix.lower() == ".zip":
            extract_dir = work_dir / "archive"
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                _safe_extract_zip(local_source, extract_dir)
            except zipfile.BadZipFile as exc:
                raise InstallError("本地 ZIP 文件损坏或不是有效压缩包。") from exc
            return extract_dir
        raise InstallError("本地来源必须是存在的目录或 .zip 文件。")

    raise InstallError("source_type 只能是 'git' 或 'local'。")


def _find_skill_root(source_root: Path, subdirectory: str) -> Path:
    if subdirectory:
        candidate = _safe_subdirectory(source_root, subdirectory)
        if not (candidate / "SKILL.md").is_file():
            raise InstallError("指定 subdirectory 中没有 SKILL.md。")
        return candidate

    candidates = sorted(path.parent for path in source_root.rglob("SKILL.md"))
    if not candidates:
        raise InstallError("安装来源中未找到 SKILL.md。")
    if len(candidates) > 1:
        options = ", ".join(str(path.relative_to(source_root)) for path in candidates[:8])
        raise InstallError(f"发现多个 SKILL.md，请指定 subdirectory。候选: {options}")
    return candidates[0]


def _install_tree(source_dir: Path, target_root: Path, name: str) -> Path:
    # The existence check and rename must be one critical section. Without it,
    # two model/tool retries can both pass the check and one failed retry can
    # delete the directory created by the successful retry.
    with _INSTALL_TREE_LOCK:
        target_root.mkdir(parents=True, exist_ok=True)
        destination = _inside(target_root, target_root / name)
        if destination.exists():
            raise InstallError(
                f"目标已存在或仍被占用: {destination.name}。"
                "请先停止正在运行的 MCP 后重试。"
            )

        staging = _inside(target_root, target_root / f".staging-{uuid.uuid4().hex}")
        try:
            _copy_tree(source_dir, staging)
            os.replace(staging, destination)
        except PermissionError as exc:
            raise InstallError(
                f"无法切换安装目录 {destination.name}：目标可能被正在运行的 MCP、"
                "文件索引或杀毒软件占用。请停止 FluxAgent 后重试。"
            ) from exc
        except Exception:
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
        return destination


def install_skill_source(
    source_type: str,
    source: str,
    name: str = "",
    subdirectory: str = "",
    target_root: Optional[Path] = None,
) -> Path:
    # 完成 Skill 安装流程：获取来源、解压/克隆、查找 SKILL.md、复制到目标目录。
    
    """安装一个 Skill，支持受控 Git Clone 或本地目录/ZIP 导入。"""
    with tempfile.TemporaryDirectory(prefix="fluxagent-skill-") as tmp:
        source_root = _materialize_source(source_type, source, Path(tmp))
        skill_root = _find_skill_root(source_root, subdirectory)
        target_name = _safe_name(name or skill_root.name, "name")
        destination = _install_tree(skill_root, target_root or Path(SKILLS_DIR), target_name)
    audit_logger.log_event(
        thread_id="system_default",
        event="system_action",
        action="skill_installed",
        source_type=source_type,
        source=source,
        skill_name=destination.name,
    )
    return destination


def _read_mcp_index(index_path: Path) -> list[dict]:
    """Read the MCP registry and fail closed when it is corrupt."""
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        servers = payload.get("servers", []) if isinstance(payload, dict) else []
        return servers if isinstance(servers, list) else []
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"MCP 索引损坏或不可读: {index_path}") from exc


def _write_mcp_index(index_path: Path, servers: list[dict]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_suffix(".tmp")
    temp_path.write_text(
        json.dumps({"servers": servers}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, index_path)


def install_mcp_source(
    source_type: str,
    source: str,
    name: str,
    command: str = "",
    args: Optional[list[str]] = None,
    read_only_tools: Optional[list[str]] = None,
    subdirectory: str = "",
    target_root: Optional[Path] = None,
    index_path: Optional[Path] = None,
) -> InstalledMCPServer:
    """识别、准备、验证并登记一个 stdio MCP Server。"""
    server_name = _safe_name(name, "name")
    launch_command = (command or "").strip()
    launch_args = [str(item) for item in (args or [])]
    declared_read_only = [
        str(item).strip() for item in (read_only_tools or []) if str(item).strip()
    ]
    root = target_root or Path(MCP_SERVERS_DIR)
    registry_path = index_path or Path(MCP_SERVERS_INDEX)

    with _MCP_INDEX_LOCK:
        registered = _read_mcp_index(registry_path)
        existing = next(
            (item for item in registered if item.get("name") == server_name), None
        )

    if existing is not None:
        if str(existing.get("source") or "") != source:
            raise InstallError(
                f"MCP Server 名称已被其他来源占用: {server_name}。"
            )
        destination = Path(str(existing.get("installed_path") or "")).resolve()
        if not destination.is_dir():
            raise InstallError(
                f"已登记的 MCP 源码目录不存在: {destination}。请换一个名称重新安装。"
            )
        launch_command, launch_args, runtime = _prepare_mcp(
            destination, launch_command, launch_args
        )
        record = InstalledMCPServer(
            name=server_name,
            source_type=source_type,
            source=source,
            installed_path=str(destination),
            command=launch_command,
            args=launch_args,
            read_only_tools=declared_read_only or list(existing.get("read_only_tools") or []),
            installed_at=str(
                existing.get("installed_at")
                or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            ),
        )
        discovered_tools = _probe_mcp(asdict(record))
        with _MCP_INDEX_LOCK:
            servers = _read_mcp_index(registry_path)
            servers = [
                asdict(record) if item.get("name") == server_name else item
                for item in servers
            ]
            _write_mcp_index(registry_path, servers)
        audit_logger.log_event(
            thread_id="system_default",
            event="system_action",
            action="mcp_server_repaired",
            source_type=source_type,
            source=source,
            mcp_name=server_name,
            runtime=runtime,
            tools=discovered_tools,
        )
        return record

    destination: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="fluxagent-mcp-") as tmp:
            source_root = _materialize_source(source_type, source, Path(tmp))
            server_root = _safe_subdirectory(source_root, subdirectory)
            if not server_root.is_dir():
                raise InstallError("指定的 MCP subdirectory 不是目录。")
            destination = _install_tree(server_root, root, server_name)

        launch_command, launch_args, runtime = _prepare_mcp(
            destination, launch_command, launch_args
        )
        record = InstalledMCPServer(
            name=server_name,
            source_type=source_type,
            source=source,
            installed_path=str(destination),
            command=launch_command,
            args=launch_args,
            read_only_tools=declared_read_only,
            installed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        discovered_tools = _probe_mcp(asdict(record))

        with _MCP_INDEX_LOCK:
            servers = _read_mcp_index(registry_path)
            if any(item.get("name") == server_name for item in servers):
                raise InstallError(f"MCP Server 名称已登记: {server_name}。")
            servers.append(asdict(record))
            _write_mcp_index(registry_path, servers)
    except Exception:
        # Download, setup, health check and registry commit form one transaction.
        if destination is not None and destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        raise

    audit_logger.log_event(
        thread_id="system_default",
        event="system_action",
        action="mcp_server_installed",
        source_type=source_type,
        source=source,
        mcp_name=server_name,
        runtime=runtime,
        tools=discovered_tools,
    )
    return record


def list_installed_mcp_servers(index_path: Optional[Path] = None) -> list[dict]:
    """读取已安装 MCP Server 的本地登记信息。"""
    with _MCP_INDEX_LOCK:
        return _read_mcp_index(index_path or Path(MCP_SERVERS_INDEX))
