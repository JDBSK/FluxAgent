# -*- coding: utf-8 -*-
"""Durable one-shot approval records for main and delegated tool calls."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Mapping, Optional
from uuid import uuid4

from .config import DB_PATH


class ActionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"
    EXPIRED = "expired"


TERMINAL_STATUSES = frozenset(
    {
        ActionStatus.REJECTED,
        ActionStatus.SUCCEEDED,
        ActionStatus.FAILED,
        ActionStatus.UNKNOWN,
        ActionStatus.EXPIRED,
    }
)


@dataclass(frozen=True)
class ActionRecord:
    action_id: str
    thread_id: str
    tool_call_id: str
    tool_name: str
    args: dict[str, Any]
    status: ActionStatus
    actor: str
    parent_run_id: str = ""
    child_run_id: str = ""
    child_name: str = ""
    risk_level: str = ""
    reason: str = ""
    workspace_scope: str = ""
    resource_version: Optional[dict[str, Any]] = None
    result: str = ""
    error_code: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0
    approved_at: Optional[float] = None
    executing_at: Optional[float] = None
    finished_at: Optional[float] = None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _safe_resource_version(tool_name: str, args: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    if tool_name != "write_office_file":
        return None
    relative = args.get("filepath")
    if not isinstance(relative, str) or not relative.strip():
        return None
    from .tools import sandbox_tools

    base = os.path.realpath(sandbox_tools.OFFICE_DIR)
    candidate = os.path.realpath(os.path.join(base, relative))
    try:
        if os.path.commonpath([os.path.normcase(base), os.path.normcase(candidate)]) != os.path.normcase(base):
            return None
    except (OSError, ValueError):
        return None
    try:
        stat = os.stat(candidate)
    except FileNotFoundError:
        return {"path": candidate, "exists": False, "size": 0, "mtime_ns": 0}
    except OSError:
        return None
    return {
        "path": candidate,
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class PermissionBroker:
    """基于 SQLite 数据库表实现的单次动作代理。

    授权即不透明的 ``action_id`` 及其状态转移。工具参数以不可变的动作载荷形式存储;
    不生成参数摘要,用作授权决定。
    """

    def __init__(self, path: str | Path = DB_PATH, *, ttl_seconds: float = 1800.0) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.path = str(path)
        self.ttl_seconds = float(ttl_seconds)
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=30000")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS permission_actions (
                    action_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    parent_run_id TEXT NOT NULL DEFAULT '',
                    child_run_id TEXT NOT NULL DEFAULT '',
                    child_name TEXT NOT NULL DEFAULT '',
                    risk_level TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    workspace_scope TEXT NOT NULL DEFAULT '',
                    resource_json TEXT,
                    result TEXT NOT NULL DEFAULT '',
                    error_code TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    approved_at REAL,
                    executing_at REAL,
                    finished_at REAL,
                    UNIQUE(thread_id, child_run_id, tool_call_id)
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_permission_actions_pending "
                "ON permission_actions(thread_id, status, created_at)"
            )
            now = time.time()
            self._conn.execute(
                "UPDATE permission_actions SET status=?, finished_at=?, error_code=? "
                "WHERE status=?",
                (
                    ActionStatus.UNKNOWN.value,
                    now,
                    "process_interrupted",
                    ActionStatus.EXECUTING.value,
                ),
            )
            self._expire_locked(now)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "PermissionBroker":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def register(
        self,
        *,
        thread_id: str,
        tool_call_id: str,
        tool_name: str,
        args: Mapping[str, Any] | None,
        actor: str,
        parent_run_id: str = "",
        child_run_id: str = "",
        child_name: str = "",
        risk_level: str = "",
        reason: str = "",
        workspace_scope: str = "",
    ) -> ActionRecord:
        if not thread_id or not tool_call_id or not tool_name:
            raise ValueError("thread_id, tool_call_id and tool_name are required")
        immutable_args = dict(args or {})
        now = time.time()
        action_id = uuid4().hex
        resource = _safe_resource_version(tool_name, immutable_args)
        with self._lock:
            self._expire_locked(now)
            try:
                self._conn.execute(
                    """
                    INSERT INTO permission_actions (
                        action_id, thread_id, tool_call_id, tool_name, args_json,
                        status, actor, parent_run_id, child_run_id, child_name,
                        risk_level, reason, workspace_scope, resource_json,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        action_id,
                        thread_id,
                        tool_call_id,
                        tool_name,
                        _json(immutable_args),
                        ActionStatus.PENDING.value,
                        actor,
                        parent_run_id,
                        child_run_id,
                        child_name,
                        risk_level,
                        reason,
                        workspace_scope,
                        _json(resource) if resource is not None else None,
                        now,
                        now + self.ttl_seconds,
                    ),
                )
            except sqlite3.IntegrityError:
                pass
            row = self._conn.execute(
                "SELECT * FROM permission_actions WHERE thread_id=? AND child_run_id=? AND tool_call_id=?",
                (thread_id, child_run_id, tool_call_id),
            ).fetchone()
        if row is None:
            raise RuntimeError("failed to register approval action")
        return self._from_row(row)

    def get(self, action_id: str) -> Optional[ActionRecord]:
        with self._lock:
            self._expire_locked(time.time())
            row = self._conn.execute(
                "SELECT * FROM permission_actions WHERE action_id=?", (action_id,)
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def find(self, *, thread_id: str, tool_call_id: str, child_run_id: str = "") -> Optional[ActionRecord]:
        with self._lock:
            self._expire_locked(time.time())
            row = self._conn.execute(
                "SELECT * FROM permission_actions WHERE thread_id=? AND child_run_id=? AND tool_call_id=?",
                (thread_id, child_run_id, tool_call_id),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def pending(self, thread_id: str) -> list[ActionRecord]:
        with self._lock:
            self._expire_locked(time.time())
            rows = self._conn.execute(
                "SELECT * FROM permission_actions WHERE thread_id=? AND status=? ORDER BY created_at, action_id",
                (thread_id, ActionStatus.PENDING.value),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def approve(self, action_id: str) -> bool:
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            cursor = self._conn.execute(
                "UPDATE permission_actions SET status=?, approved_at=? "
                "WHERE action_id=? AND status=? AND expires_at>?",
                (
                    ActionStatus.APPROVED.value,
                    now,
                    action_id,
                    ActionStatus.PENDING.value,
                    now,
                ),
            )
            return cursor.rowcount == 1

    def reject(self, action_id: str, reason: str = "user_rejected") -> bool:
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE permission_actions SET status=?, finished_at=?, error_code=? "
                "WHERE action_id=? AND status=?",
                (
                    ActionStatus.REJECTED.value,
                    now,
                    reason,
                    action_id,
                    ActionStatus.PENDING.value,
                ),
            )
            return cursor.rowcount == 1

    def claim(self, action_id: str) -> bool:
        now = time.time()
        with self._lock:
            self._expire_locked(now)
            cursor = self._conn.execute(
                "UPDATE permission_actions SET status=?, executing_at=? "
                "WHERE action_id=? AND status=?",
                (
                    ActionStatus.EXECUTING.value,
                    now,
                    action_id,
                    ActionStatus.APPROVED.value,
                ),
            )
            return cursor.rowcount == 1

    def succeed(self, action_id: str, result: str) -> bool:
        return self._finish(action_id, ActionStatus.SUCCEEDED, result=result)

    def fail(self, action_id: str, result: str, error_code: str = "tool_failed") -> bool:
        return self._finish(
            action_id, ActionStatus.FAILED, result=result, error_code=error_code
        )

    def mark_unknown(self, action_id: str, error_code: str = "outcome_unknown") -> bool:
        return self._finish(action_id, ActionStatus.UNKNOWN, error_code=error_code)

    def _finish(
        self,
        action_id: str,
        status: ActionStatus,
        *,
        result: str = "",
        error_code: str = "",
    ) -> bool:
        now = time.time()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE permission_actions SET status=?, result=?, error_code=?, finished_at=? "
                "WHERE action_id=? AND status=?",
                (
                    status.value,
                    str(result),
                    error_code,
                    now,
                    action_id,
                    ActionStatus.EXECUTING.value,
                ),
            )
            return cursor.rowcount == 1

    def resource_is_current(self, action: ActionRecord) -> bool:
        expected = action.resource_version
        if expected is None:
            return True
        current = _safe_resource_version(action.tool_name, action.args)
        return current == expected

    def purge_terminal(self, *, older_than_seconds: float = 86400.0) -> int:
        cutoff = time.time() - max(0.0, older_than_seconds)
        placeholders = ",".join("?" for _ in TERMINAL_STATUSES)
        values = [status.value for status in TERMINAL_STATUSES]
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM permission_actions WHERE status IN ({placeholders}) AND finished_at<?",
                (*values, cutoff),
            )
            return int(cursor.rowcount)

    def _expire_locked(self, now: float) -> None:
        self._conn.execute(
            "UPDATE permission_actions SET status=?, finished_at=?, error_code=? "
            "WHERE status=? AND expires_at<=?",
            (
                ActionStatus.EXPIRED.value,
                now,
                "approval_expired",
                ActionStatus.PENDING.value,
                now,
            ),
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> ActionRecord:
        return ActionRecord(
            action_id=str(row["action_id"]),
            thread_id=str(row["thread_id"]),
            tool_call_id=str(row["tool_call_id"]),
            tool_name=str(row["tool_name"]),
            args=dict(_loads(row["args_json"], {})),
            status=ActionStatus(str(row["status"])),
            actor=str(row["actor"]),
            parent_run_id=str(row["parent_run_id"] or ""),
            child_run_id=str(row["child_run_id"] or ""),
            child_name=str(row["child_name"] or ""),
            risk_level=str(row["risk_level"] or ""),
            reason=str(row["reason"] or ""),
            workspace_scope=str(row["workspace_scope"] or ""),
            resource_version=_loads(row["resource_json"], None),
            result=str(row["result"] or ""),
            error_code=str(row["error_code"] or ""),
            created_at=float(row["created_at"] or 0.0),
            expires_at=float(row["expires_at"] or 0.0),
            approved_at=float(row["approved_at"]) if row["approved_at"] is not None else None,
            executing_at=float(row["executing_at"]) if row["executing_at"] is not None else None,
            finished_at=float(row["finished_at"]) if row["finished_at"] is not None else None,
        )
