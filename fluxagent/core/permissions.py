# -*- coding: utf-8 -*-
"""Parameter-scoped approval memory for FluxAgent tool calls.

Version 2 grants bind a tool name to a SHA-256 digest of canonical arguments.
The persisted file never stores complete arguments, and legacy version 1
tool-wide grants are deliberately not promoted to active permissions.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import PERMISSIONS_FILE


_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(args: Mapping[str, Any] | None) -> str:
    """Serialize arguments deterministically without relying on object identity."""
    value = dict(args or {})
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    except (TypeError, ValueError):
        normalized = {str(key): str(item) for key, item in value.items()}
        return json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def permission_call_key(name: str, args: Mapping[str, Any] | None = None) -> str:
    """Build the non-secret identity used for one exact tool-call grant."""
    tool = str(name or "").strip()
    digest = hashlib.sha256(_canonical_json(args).encode("utf-8")).hexdigest()
    return f"{tool}:{digest}"


def _entry(name: str, args: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Create a safe persisted grant entry without raw argument values."""
    tool = str(name or "").strip()
    key = permission_call_key(tool, args)
    return {"tool": tool, "args_sha256": key.rsplit(":", 1)[1]}


class PermissionManager:
    """Manage exact-call grants for the current process and future runs."""

    def __init__(self, path: str | Path = PERMISSIONS_FILE) -> None:
        self.path = Path(path)
        self.session_grants: set[str] = set()
        self.always_grants: set[str] = set()
        self._tool_names: dict[str, str] = {}
        self._legacy_session_tools: set[str] = set()
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        """Load v2 grants; malformed files and legacy broad grants fail closed."""
        self.always_grants.clear()
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        if not isinstance(payload, dict) or payload.get("version") != 2:
            return
        values = payload.get("always_allow_calls", [])
        if not isinstance(values, list):
            return
        for value in values:
            if not isinstance(value, dict):
                continue
            tool = str(value.get("tool") or "").strip()
            digest = str(value.get("args_sha256") or "").strip().lower()
            if not tool or not _DIGEST_RE.fullmatch(digest):
                continue
            key = f"{tool}:{digest}"
            self.always_grants.add(key)
            self._tool_names[key] = tool

    def is_allowed(self, name: str, args: Mapping[str, Any] | None = None) -> bool:
        """Return whether this exact tool name and argument set was granted."""
        return permission_call_key(name, args) in (self.session_grants | self.always_grants)

    def allowed_tools(self) -> set[str]:
        """Return granted tool names for display and deprecated callers only.

        Security decisions must use :meth:`is_allowed`; this method intentionally
        does not turn exact grants into permission for other argument values.
        """
        keys = self.session_grants | self.always_grants
        return {self._tool_names[key] for key in keys if key in self._tool_names} | self._legacy_session_tools

    def grant_calls(
        self,
        tool_calls: Iterable[Mapping[str, Any]],
        persistent: bool = False,
    ) -> None:
        """Grant exact calls; persist only hashes when ``persistent`` is true."""
        entries: list[dict[str, str]] = []
        for call in tool_calls:
            if not isinstance(call, Mapping):
                raise TypeError("tool_calls 中的每一项都必须是映射。")
            name = str(call.get("name") or "").strip()
            args = call.get("args") or {}
            if not name:
                raise ValueError("工具名不能为空。")
            if not isinstance(args, Mapping):
                raise TypeError(f"工具 {name} 的 args 必须是映射。")
            entries.append(_entry(name, args))
        with self._lock:
            target = self.always_grants if persistent else self.session_grants
            for item in entries:
                key = f"{item['tool']}:{item['args_sha256']}"
                target.add(key)
                self._tool_names[key] = item["tool"]
            if persistent:
                self._write_atomic()

    def grant(self, tools: Iterable[str], persistent: bool = False) -> None:
        """Compatibility helper for old callers; it never creates broad grants.

        Tool names are retained for current-process display only. Persistent or
        automatic approval requires :meth:`grant_calls` with concrete arguments.
        """
        names = {str(tool).strip() for tool in tools if str(tool).strip()}
        if persistent:
            raise ValueError("永久授权必须使用 grant_calls() 绑定具体参数。")
        self._legacy_session_tools.update(names)

    def _write_atomic(self) -> None:
        """Persist v2 grants with fsync and atomic replacement."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        values = []
        for key in sorted(self.always_grants):
            tool, digest = key.rsplit(":", 1)
            values.append({"tool": tool, "args_sha256": digest})
        payload = {"version": 2, "always_allow_calls": values}
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_name = handle.name
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.path)
            temp_name = None
        finally:
            if temp_name:
                try:
                    Path(temp_name).unlink()
                except FileNotFoundError:
                    pass

    def clear(self, persistent: bool = True) -> None:
        """Clear session grants and optionally remove the persistent grant file."""
        with self._lock:
            self.session_grants.clear()
            self._legacy_session_tools.clear()
            if persistent:
                self.always_grants.clear()
                self._tool_names.clear()
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass

    def describe(self) -> str:
        """Return a compact CLI summary without exposing argument values."""
        session = sorted({self._tool_names.get(key, "未知") for key in self.session_grants})
        session.extend(sorted(self._legacy_session_tools))
        always = sorted({self._tool_names.get(key, "未知") for key in self.always_grants})
        return (
            "权限状态:\n"
            f"  本次运行精确授权: {', '.join(session) or '无'}\n"
            f"  永久精确授权: {', '.join(always) or '无'}\n"
            "  授权只匹配相同工具和相同参数；critical 参数仍会被硬拒绝。"
        )
