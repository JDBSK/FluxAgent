# -*- coding: utf-8 -*-
"""懒加载技能加载器 + help→run 一次性凭据门禁。

设计要点：
  1. 启动只扫描元数据（name/description/脚本清单），首次调用才读完整说明；
  2. run 是"先看说明书、再凭据执行"的两阶段协议：help 返回一次性
     help_token，run 必须携带。凭据绑定 会话 + 技能 + 说明版本（SHA-256），
     300 秒有效、锁内单次消费；没拿凭据、重复使用、跨会话/技能、说明变更，
     执行器（execute_office_shell）都不会被调用；
  3. 凭据只证明"看过当前说明"，不代表人工授权，也不是脚本隔离；
  4. 纯文档技能（无可执行脚本）没有执行器，run 在凭据校验前就直接
     返回 skill_not_executable，引导模型依据说明直接回答。
"""

import hashlib
import json
import os
import re
import secrets
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from .config import SKILLS_DIR
from .tools.sandbox_tools import execute_office_shell


MAX_SKILL_BYTES = 64 * 1024  # 技能说明书大小上限：不截断授权，超限即拒绝

_SCRIPT_SUFFIXES = (".py", ".js", ".cmd", ".bat", ".ps1")


def _parse_metadata_list(value: str) -> list[str]:
    """Parse simple aliases/triggers fields without adding a YAML dependency."""
    text = str(value or "").strip().strip("[]")
    quoted = re.findall(r"[「『\"']([^」』\"']+)[」』\"']", text)
    if quoted:
        return [item.strip() for item in quoted if item.strip()]
    return [
        item.strip().strip("\"'`「」『』（）()")
        for item in re.split(r"[,，、;；|]", text)
        if item.strip().strip("\"'")
    ]


def _skill_scripts(skill_dir: str) -> list[str]:
    """Return executable-looking files shipped with a skill.

    A number of Skills are prompt/manual-only packages.  Treating every
    package as a shell program makes the model guess script names and yields
    opaque Python ``can't open file`` errors.  Keep this discovery local to
    the skill directory and return paths relative to the Skill for diagnostics.
    """
    scripts: list[str] = []
    if not os.path.isdir(skill_dir):
        return scripts
    for root, _dirs, files in os.walk(skill_dir):
        for filename in files:
            if filename.casefold().endswith(_SCRIPT_SUFFIXES):
                relative = os.path.relpath(os.path.join(root, filename), skill_dir)
                scripts.append(relative.replace(os.sep, "/"))
    return sorted(scripts)


def _skill_error(code: str, message: str) -> str:
    """Return the same stable JSON envelope used by sandbox tools."""
    return json.dumps(
        {
            "ok": False,
            "display": "❌ 工具错误",
            "error": {"code": code, "message": message, "retryable": False},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class DynamicSkillInput(BaseModel):
    """动态技能工具的调用参数：先 help 拿凭据，再 run 执行。"""

    mode: Literal["help", "run"] = Field(
        description="必须是 'help' 或 'run'。help 阅读完整说明书并签发一次性 help_token。"
    )
    command: Optional[str] = Field(
        default="",
        description="仅在 mode='run' 时需要。你要执行的完整命令，保留 {baseDir} 占位符。"
    )
    help_token: str = Field(
        default="",
        description="mode='run' 必填：本会话 help 返回的一次性 help_token。"
    )


class LazySkillLoader:
    """元数据 TTL 60s；说明书字节每次 help/run 都重新校验。

    Grants 绑定会话/技能/说明版本，派发前锁内单次消费；重启后需重新 help。
    """

    def __init__(self, cache_size: int = 50, grant_ttl: int = 300, max_grants: int = 1024):
        if cache_size < 1 or grant_ttl <= 0 or max_grants < 1:
            raise ValueError("cache_size、grant_ttl 和 max_grants 必须大于 0")
        self._cache_size = cache_size
        self._grant_ttl = grant_ttl
        self._max_grants = max_grants
        self._skill_registry: Optional[List[Dict[str, Any]]] = None
        self._last_scan_time = 0.0
        self._scan_interval = 60
        self._lock = threading.RLock()
        self._content_cache: "OrderedDict[tuple, str]" = OrderedDict()
        self._grants: "OrderedDict[str, tuple]" = OrderedDict()

    # ---------- 说明书快照：路径边界 + SHA-256 版本 + LRU 内容缓存 ----------

    def _snapshot(self, md_path: str) -> tuple[str, str, str]:
        """读取技能说明，返回 (规范路径, 版本哈希, 内容)。

        说明必须位于 skills 目录内且不超过 64 KiB；版本是内容的 SHA-256，
        凭据绑定该版本——文件一变，旧凭据立即失效。
        """
        root = os.path.realpath(SKILLS_DIR)
        path = os.path.realpath(md_path)
        if os.path.commonpath([root, path]) != root:
            raise PermissionError("技能说明必须位于 skills 目录内")
        with open(path, "rb") as stream:
            raw = stream.read(MAX_SKILL_BYTES + 1)
        if len(raw) > MAX_SKILL_BYTES:
            raise ValueError(f"技能说明超过 {MAX_SKILL_BYTES} 字节，请精简；不会截断后授权")
        version = hashlib.sha256(raw).hexdigest()
        key = (path, version)
        with self._lock:
            content = self._content_cache.get(key)
            if content is None:
                content = raw.decode("utf-8")
                self._content_cache[key] = content
            self._content_cache.move_to_end(key)
            while len(self._content_cache) > self._cache_size:
                self._content_cache.popitem(last=False)
        return path, version, content

    def _load_skill_content(self, md_path: str, mtime: float | None = None) -> str:
        # 兼容旧签名；捕获到的 mtime 绝不能作为执行授权依据。
        return self._snapshot(md_path)[2]

    # ---------- 会话凭据：签发 / 消费 ----------

    @staticmethod
    def _session_id(config: RunnableConfig | None) -> str | None:
        value = (config or {}).get("configurable", {}).get("thread_id")
        return str(value) if value is not None and str(value).strip() else None

    def _issue_grant(self, session: str, path: str, version: str) -> str:
        now = time.monotonic()
        with self._lock:
            # 同会话同技能重复 help：旧凭据作废，防止凭据囤积。
            for token, grant in list(self._grants.items()):
                if grant[3] <= now or grant[:2] == (session, path):
                    del self._grants[token]
            token = secrets.token_urlsafe(32)
            self._grants[token] = (session, path, version, now + self._grant_ttl)
            while len(self._grants) > self._max_grants:
                self._grants.popitem(last=False)
            return token

    def _consume_grant(self, token: str, session: str, path: str, version: str) -> bool:
        with self._lock:
            grant = self._grants.get(token)
            if grant is None:
                return False
            if grant[3] <= time.monotonic():
                del self._grants[token]
                return False
            if grant[:3] != (session, path, version):
                return False
            del self._grants[token]
            return True

    # ---------- 扫描与元数据（保留 FluxAgent 的别名/脚本发现） ----------

    def _scan_skills(self, force_rescan: bool = False) -> List[Dict[str, Any]]:
        current_time = time.monotonic()
        with self._lock:
            if (not force_rescan and self._skill_registry is not None
                    and current_time - self._last_scan_time < self._scan_interval):
                return self._skill_registry

        skills: list[dict[str, Any]] = []
        seen_capabilities: set[str] = set()

        if not os.path.exists(SKILLS_DIR):
            with self._lock:
                self._skill_registry = []
                self._last_scan_time = current_time
            return []

        for item in sorted(os.listdir(SKILLS_DIR)):
            folder_path = os.path.join(SKILLS_DIR, item)
            if not os.path.isdir(folder_path):
                continue

            md_path = os.path.join(folder_path, "SKILL.md")
            if not os.path.exists(md_path):
                md_path = os.path.join(folder_path, "README.md")

            if not os.path.exists(md_path):
                continue

            try:
                metadata = self._extract_metadata(md_path)
            except Exception as e:
                # 走 stderr：MCP stdio 把 stdout 当协议通道，print 会打挂握手
                print(f" [警告] 扫描技能 {item} 失败: {e}", file=sys.stderr)
                continue

            if metadata:
                capability_key = str(metadata.get("name") or item).casefold()
                if capability_key in seen_capabilities:
                    # 能力名冲突必须 fail closed：重名会让模型无法区分要调用
                    # 的工具，静默跳过会隐藏真实部署错误。（在 try 之外抛出，
                    # 不被单技能的扫描容错吞掉。）
                    raise ValueError(f"技能名称冲突: {metadata.get('name') or item}")
                seen_capabilities.add(capability_key)
                skills.append({
                    "folder": item,
                    "md_path": md_path,
                    "scripts": _skill_scripts(folder_path),
                    **metadata
                })

        with self._lock:
            self._skill_registry = skills
            self._last_scan_time = current_time

        if skills:
            print(f" [OK] 扫描到 {len(skills)} 个技能（懒加载 + 凭据门禁模式）", file=sys.stderr)

        return skills

    def _extract_metadata(self, md_path: str) -> Optional[Dict[str, Any]]:
        """只读取前 120 行提取 name/description/触发词，避免加载全文。"""
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= 120:
                        break
                    lines.append(line)
                content = "\n".join(lines)

            name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
            desc_match = re.search(r"^description:\s*(.*)$", content, re.MULTILINE)

            raw_name = name_match.group(1).strip() if name_match else os.path.basename(os.path.dirname(md_path))
            tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)
            if not tool_name:
                raise ValueError("技能名称不能为空")

            raw_desc = desc_match.group(1).strip() if desc_match else f"提供 {raw_name} 相关功能"
            if raw_desc in {"|", ">"}:
                desc_lines = []
                collecting = False
                for line in content.splitlines():
                    if re.match(r"^description:\s*[|>]", line):
                        collecting = True
                        continue
                    if collecting:
                        if line.startswith((" ", "\t")):
                            desc_lines.append(line.strip())
                        elif line.strip() == "---":
                            break
                raw_desc = " ".join(desc_lines).strip() or f"提供 {raw_name} 相关功能"
            if (raw_desc.startswith('"') and raw_desc.endswith('"')) or (raw_desc.startswith("'") and raw_desc.endswith("'")):
                raw_desc = raw_desc[1:-1]

            aliases: list[str] = []
            for match in re.finditer(
                r"(?:^|\n)\s*(?:aliases|triggers|keywords|触发词)\s*[:：]\s*(.+)",
                content,
                re.IGNORECASE,
            ):
                aliases.extend(_parse_metadata_list(match.group(1)))
            aliases.extend([raw_name, os.path.basename(os.path.dirname(md_path))])
            aliases = list(dict.fromkeys(alias for alias in aliases if alias))

            return {
                "raw_name": raw_name,
                "name": tool_name,
                "description": raw_desc,
                "aliases": aliases,
            }
        except Exception as e:
            print(f" [警告] 提取元数据失败 {md_path}: {e}", file=sys.stderr)
            return None

    # ---------- 工具构造：help→run 两阶段协议 ----------

    def _create_lazy_tool(self, skill_info: Dict[str, Any]) -> StructuredTool:
        def lazy_runner(
            mode: str, command: str = "", help_token: str = "",
            config: RunnableConfig = None,
        ) -> str:
            """懒加载执行器：help 阅读说明并签发凭据，run 凭据执行。"""
            session = self._session_id(config)
            mode = str(mode or "").strip()
            if mode not in ("help", "run"):
                return "错误：mode 参数只能是 'help' 或 'run'。"
            if mode == "run" and not str(command or "").strip():
                return "错误：在 'run' 模式下，必须提供 command 参数！"

            scripts = skill_info.get("scripts", [])
            if mode == "run" and not scripts:
                # 纯文档技能没有执行器：无需凭据即可明确拒绝（不触发 shell）。
                return _skill_error(
                    "skill_not_executable",
                    f"Skill '{skill_info['raw_name']}' 只有说明文档，没有可执行脚本。"
                    " 请依据 mode='help' 返回的内容直接回答，不要调用 run。",
                )

            if mode == "run" and (not session or not str(help_token or "").strip()):
                return (
                    "权限拒绝：run 需要 thread_id、非空 command 和本会话 help 返回的"
                    " help_token。请先以 mode='help' 获取凭据。"
                )

            try:
                path, version, content = self._snapshot(skill_info["md_path"])
            except (OSError, ValueError) as exc:
                return f"权限拒绝：无法读取当前技能说明，请修复后重新 help。{exc}"

            if mode == "help":
                manual = (
                    f"========== 【{skill_info['raw_name']} 完整说明书】 ==========\n"
                    f"{content}\n"
                    f"====================================\n"
                )
                if not scripts:
                    return manual + "提示：这是说明文档型 Skill，没有本地脚本。请依据以上内容直接回答用户。"
                if not session:
                    return manual + "未提供 config.configurable.thread_id：可阅读说明，但不会签发执行凭据。"
                # 凭据绑定 (技能目录, 说明路径, 版本)：同一说明经符号链接出现在
                # 别名目录下时路径分量不同，凭据不互通，防止凭据跨技能复用。
                token = self._issue_grant(session, (skill_info["folder"], path), version)
                return (
                    manual
                    + f"help_token: {token}\n"
                    f"有效期 {self._grant_ttl:g} 秒，仅限本会话、本技能当前说明版本使用一次。\n"
                    "确认适用后以 mode='run'、command 和 help_token 调用；"
                    "执行失败或说明变化后需重新 help。"
                )

            # mode == "run"：锁内单次消费凭据，重放/跨会话/版本变更一律拒绝。
            if not self._consume_grant(help_token, session, (skill_info["folder"], path), version):
                return (
                    "权限拒绝：help_token 无效、已使用、过期、跨会话/技能或说明版本已变化。"
                    " 请重新 help。"
                )

            actual_cmd = command.replace("{baseDir}", f"skills/{skill_info['folder']}")
            # 拦截模型编造脚本名的常见错误；只检查显式指向本 Skill 的路径，
            # 其余命令参数校验全部留在 shell 沙盒里。
            skill_prefix = f"skills/{skill_info['folder']}/"
            valid_paths = {f"skills/{skill_info['folder']}/{script}" for script in scripts}
            referenced = re.findall(
                rf"{re.escape(skill_prefix)}[^\s\"']+", actual_cmd.replace("\\", "/")
            )
            for referenced_path in referenced:
                relative = referenced_path.replace("\\", "/")
                if relative not in valid_paths:
                    return _skill_error(
                        "skill_script_not_found",
                        f"Skill '{skill_info['raw_name']}' 中不存在脚本：{relative}。"
                        f" 可用脚本：{', '.join(scripts)}",
                    )

            return execute_office_shell.invoke({"command": actual_cmd}, config=config)

        if skill_info.get("scripts"):
            usage_hint = (
                "注意：这是一个外部扩展技能。首次使用请务必先传入 `mode='help'` 阅读完整说明书"
                "并获取一次性 help_token，之后再使用 `mode='run'` 配合 `command` 与 `help_token` 执行底层脚本。"
            )
        else:
            usage_hint = (
                "注意：这是说明文档型 Skill。请先传入 `mode='help'` 阅读说明书，"
                "再依据内容直接回答，不要传入 `mode='run'`。"
            )
        mini_description = f"{skill_info['description']}\n\n{usage_hint}"

        return StructuredTool.from_function(
            func=lazy_runner,
            name=skill_info["name"],
            description=mini_description,
            args_schema=DynamicSkillInput,
            metadata={
                # The declared Skill name is the stable capability ID. Folder
                # names are deployment details and may differ across copies.
                "capability": skill_info["name"],
                "aliases": skill_info.get("aliases", []),
                "kind": "skill",
            },
        )

    def get_all_tools(self, force_rescan: bool = False) -> List[StructuredTool]:
        skill_infos = self._scan_skills(force_rescan=force_rescan)
        return [self._create_lazy_tool(skill_info) for skill_info in skill_infos]

    def get_tool_count(self) -> int:
        """获取技能数量（不触发加载）"""
        return len(self._scan_skills())

    def clear_cache(self):
        """清除内容缓存并撤销全部未消费凭据。"""
        with self._lock:
            self._content_cache.clear()
            self._grants.clear()
            self._skill_registry = None
        print(" [OK] 技能缓存已清除，未消费凭据已撤销", file=sys.stderr)


# 全局懒加载器实例
_lazy_loader = LazySkillLoader(cache_size=50)


def load_dynamic_skills(force_rescan: bool = False) -> List[StructuredTool]:
    """加载动态技能（懒加载 + 凭据门禁）。"""
    return _lazy_loader.get_all_tools(force_rescan=force_rescan)


def reload_skills() -> List[StructuredTool]:
    """清缓存撤销凭据并强制重扫；既有图需要重新绑定工具。"""
    _lazy_loader.clear_cache()
    return _lazy_loader.get_all_tools(force_rescan=True)


def get_skill_count() -> int:
    """获取当前技能数量（不触发加载）"""
    return _lazy_loader.get_tool_count()


def clear_skill_cache():
    """清除技能内容缓存并撤销凭据"""
    _lazy_loader.clear_cache()
