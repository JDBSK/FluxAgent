"""Discover SKILL.md packages and expose them as lazily executed LangChain tools."""

import os
import re
import sys
import time
import json
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from functools import lru_cache

from .config import SKILLS_DIR
from .tools.sandbox_tools import execute_office_shell


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
    """定义动态技能工具的调用参数。"""

    mode: Literal["help", "run"] = Field(
        description="必须是 'help' 或 'run'。第一次使用时强烈建议先传入 'help' 阅读说明书。"
    )
    command: Optional[str] = Field(
        default="",
        description="仅在 mode='run' 时需要。你要执行的完整命令，保留 {baseDir} 占位符。"
    )


class LazySkillLoader:
    """
    渐进式技能加载器 + 缓存机制

    特性：
    1. 启动时只扫描元数据（name, description），不加载完整内容
    2. 首次调用技能时才加载完整内容并缓存
    3. 支持热更新（修改技能文件后自动重新加载）
    4. LRU缓存策略，自动清理不常用的技能
    """

    def __init__(self, cache_size: int = 50):
        self._skill_registry: Optional[List[Dict[str, Any]]] = None
        self._cache_size = cache_size
        self._last_scan_time = 0
        self._scan_interval = 60  # 缓存元数据扫描结果60秒

    @lru_cache(maxsize=50)
    def _load_skill_content(self, md_path: str, mtime: float) -> str:
        """
        加载技能完整内容（带缓存）

        Args:
            md_path: 技能文件路径
            mtime: 文件修改时间（用于缓存失效检测）

        Returns:
            技能的完整 Markdown 内容
        """
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    def _scan_skills(self, force_rescan: bool = False) -> List[Dict[str, Any]]:
        """
        扫描技能目录，只提取元数据（轻量级操作）

        Args:
            force_rescan: 是否强制重新扫描（忽略缓存）

        Returns:
            技能元数据列表
        """
        current_time = time.time()

        # 缓存检查：如果最近扫描过且不强制刷新，直接返回缓存
        if (not force_rescan and
            self._skill_registry is not None and
                current_time - self._last_scan_time < self._scan_interval):
            return self._skill_registry

        skills = []
        seen_capabilities: set[str] = set()

        if not os.path.exists(SKILLS_DIR):
            self._skill_registry = []
            self._last_scan_time = current_time
            return []

        for item in os.listdir(SKILLS_DIR):
            folder_path = os.path.join(SKILLS_DIR, item)
            if not os.path.isdir(folder_path):
                continue

            md_path = os.path.join(folder_path, "SKILL.md")
            if not os.path.exists(md_path):
                md_path = os.path.join(folder_path, "README.md")

            if not os.path.exists(md_path):
                continue

            try:
                # 只读取前几行（name, description）
                metadata = self._extract_metadata(md_path)

                if metadata:
                    capability_key = str(metadata.get("name") or item).casefold()
                    if capability_key in seen_capabilities:
                        print(
                            f" [警告] 跳过重复 Skill 能力 {metadata.get('name') or item}（目录 {item}）",
                            file=sys.stderr,
                        )
                        continue
                    seen_capabilities.add(capability_key)
                    skill_scripts = _skill_scripts(folder_path)
                    skills.append({
                        "folder": item,
                        "md_path": md_path,
                        "mtime": os.path.getmtime(md_path),
                        "scripts": skill_scripts,
                        **metadata
                    })
            except Exception as e:
                # 走 stderr：MCP stdio 把 stdout 当协议通道，print 会打挂握手
                print(f" [警告] 扫描技能 {item} 失败: {e}", file=sys.stderr)

        self._skill_registry = skills
        self._last_scan_time = current_time

        if skills:
            print(f" [OK] 扫描到 {len(skills)} 个技能（懒加载模式）", file=sys.stderr)

        return skills

    def _extract_metadata(self, md_path: str) -> Optional[Dict[str, Any]]:
        """
        从技能文件中提取元数据（只读取必要的部分）

        Args:
            md_path: 技能文件路径

        Returns:
            包含 name 和 description 的字典
        """
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                # 只读取前 120 行（元数据与 description 通常在文件开头）
                lines = []
                for i, line in enumerate(f):
                    if i >= 120:
                        break
                    lines.append(line)

                content = "\n".join(lines)

            name_match = re.search(r"^name:\s*(.+)$", content, re.MULTILINE)
            desc_match = re.search(r"^description:\s*(.*)$", content, re.MULTILINE)

            raw_name = name_match.group(1).strip(
            ) if name_match else os.path.basename(os.path.dirname(md_path))
            tool_name = re.sub(r'[^a-zA-Z0-9_-]', '_', raw_name)

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

    def _create_lazy_tool(self, skill_info: Dict[str, Any]) -> StructuredTool:
        """
        创建懒加载工具对象

        Args:
            skill_info: 技能元数据

        Returns:
            LangChain 工具对象
        """
        def lazy_runner(mode: str, command: str = "") -> str:
            """懒加载执行器：首次调用时才加载完整内容"""
            if mode == "help":
                # 懒加载：首次调用时才读取完整内容
                skill_content = self._load_skill_content(
                    skill_info["md_path"],
                    skill_info["mtime"]
                )

                return (
                    f"========== 【{skill_info['raw_name']} 完整说明书】 ==========\n"
                    f"{skill_content[:3000]}\n"
                    f"====================================\n"
                    + (
                        "提示：这是说明文档型 Skill，没有本地脚本。请依据以上内容直接回答用户。"
                        if not skill_info.get("scripts")
                        else "提示：请根据以上说明，如果觉得能解决问题，就将 mode 设为 'run'，"
                        "并将拼装好的执行命令填入 command 重新调用。"
                    )
                )
            elif mode == "run":
                if not command:
                    return "错误：在 'run' 模式下，必须提供 command 参数！"

                # Some Skills intentionally contain only a SKILL.md/README.md
                # and are meant to guide the model's response rather than run
                # a local program.  Fail before spawning Python so callers get
                # an actionable result instead of ``can't open file``.
                scripts = skill_info.get("scripts", [])
                if not scripts:
                    return _skill_error(
                        "skill_not_executable",
                        f"Skill '{skill_info['raw_name']}' 只有说明文档，没有可执行脚本。"
                        " 请依据 mode='help' 返回的内容直接回答，不要调用 run。",
                    )

                actual_cmd = command.replace(
                    "{baseDir}", f"skills/{skill_info['folder']}"
                )

                # Catch the common case where a model invents a script name.
                # Only inspect paths explicitly rooted at this Skill; command
                # policy and all other argument validation remain in the shell
                # sandbox.
                skill_prefix = f"skills/{skill_info['folder']}/"
                valid_paths = {
                    f"skills/{skill_info['folder']}/{script}" for script in scripts
                }
                referenced = re.findall(
                    rf"{re.escape(skill_prefix)}[^\s\"']+", actual_cmd.replace("\\", "/")
                )
                for path in referenced:
                    relative = path.replace("\\", "/")
                    if relative not in valid_paths:
                        return _skill_error(
                            "skill_script_not_found",
                            f"Skill '{skill_info['raw_name']}' 中不存在脚本：{relative}。"
                            f" 可用脚本：{', '.join(scripts)}",
                        )

                return execute_office_shell.invoke({"command": actual_cmd})
            else:
                return "错误：mode 参数只能是 'help' 或 'run'。"

        if skill_info.get("scripts"):
            usage_hint = (
                "注意：这是一个外部扩展技能。首次使用请务必先传入 `mode='help'` 来阅读完整说明书，"
                "之后再使用 `mode='run'` 配合 `command` 执行底层脚本。"
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
        """
        获取所有工具（懒加载占位符）

        Args:
            force_rescan: 是否强制重新扫描技能目录

        Returns:
            工具对象列表
        """
        skill_infos = self._scan_skills(force_rescan=force_rescan)

        tools = []
        for skill_info in skill_infos:
            tools.append(self._create_lazy_tool(skill_info))

        return tools

    def get_tool_count(self) -> int:
        """获取技能数量（不触发加载）"""
        return len(self._scan_skills())

    def clear_cache(self):
        """清除所有缓存"""
        self._load_skill_content.cache_clear()
        self._skill_registry = None
        print(f" [OK] 技能缓存已清除", file=sys.stderr)


# 全局懒加载器实例
_lazy_loader = LazySkillLoader(cache_size=50)


def load_dynamic_skills(force_rescan: bool = False) -> List[StructuredTool]:
    """
    加载动态技能（懒加载 + 缓存版本）

    Args:
        force_rescan: 是否强制重新扫描技能目录（默认 False）

    Returns:
        工具对象列表（懒加载占位符）

    Note:
        - 启动时只扫描元数据，不加载完整内容
        - 首次调用技能时才加载完整内容
        - 支持热更新（修改技能文件后自动重新加载）
        - 使用 LRU 缓存策略
    """
    return _lazy_loader.get_all_tools(force_rescan=force_rescan)


def reload_skills() -> List[StructuredTool]:
    """
    强制重新扫描技能目录并清除缓存

    Returns:
        更新后的工具列表
    """
    return _lazy_loader.get_all_tools(force_rescan=True)


def get_skill_count() -> int:
    """
    获取当前技能数量（不触发加载）

    Returns:
        技能总数
    """
    return _lazy_loader.get_tool_count()


def clear_skill_cache():
    """清除技能内容缓存"""
    _lazy_loader.clear_cache()
