# -*- coding: utf-8 -*-
"""Durable memory events and an idempotent projection worker."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .memory_store import MemoryScope, ProfileCompatProvider


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    session_id: str
    run_id: str
    event_type: str
    payload: Mapping[str, Any]
    source: str
    capability_id: str
    created_at: float
    sensitivity: str = "normal"


def _json_payload(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, default=str)


class MemoryEventLog:
    """SQLite event log plus a durable pending-job table.

    Each operation uses a short-lived connection. This avoids sharing a
    sqlite connection between the agent thread and the worker thread and
    lets SQLite coordinate multiple FluxAgent processes safely.
    """

    def __init__(
        self,
        path: str,
        *,
        max_attempts: int = 5,
        retry_base_seconds: float = 0.5,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts 必须大于 0")
        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds 必须大于 0")
        self.path = os.path.abspath(path)
        self.max_attempts = int(max_attempts)
        self.retry_base_seconds = float(retry_base_seconds)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    source TEXT NOT NULL,
                    capability_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    sensitivity TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_jobs (
                    job_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until REAL NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT NOT NULL DEFAULT '',
                    updated_at REAL NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES memory_events(event_id)
                );
                CREATE TABLE IF NOT EXISTS memory_applied_events (
                    event_id TEXT PRIMARY KEY,
                    applied_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_jobs_claim
                    ON memory_jobs(status, locked_until, updated_at);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(memory_jobs)").fetchall()
            }
            if "next_attempt_at" not in columns:
                connection.execute(
                    "ALTER TABLE memory_jobs ADD COLUMN next_attempt_at REAL NOT NULL DEFAULT 0"
                )

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        session_id: str = "",
        run_id: str = "",
        source: str = "agent",
        capability_id: str = "",
        sensitivity: str = "normal",
        event_id: Optional[str] = None,
        created_at: Optional[float] = None,
    ) -> MemoryEvent:
        event = MemoryEvent(
            event_id=event_id or uuid.uuid4().hex,
            session_id=str(session_id or ""),
            run_id=str(run_id or ""),
            event_type=str(event_type),
            payload=dict(payload),
            source=str(source),
            capability_id=str(capability_id or ""),
            created_at=float(created_at if created_at is not None else time.time()),
            sensitivity=str(sensitivity or "normal"),
        )
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_events
                (event_id, session_id, run_id, event_type, payload, source,
                 capability_id, created_at, sensitivity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.session_id,
                    event.run_id,
                    event.event_type,
                    _json_payload(event.payload),
                    event.source,
                    event.capability_id,
                    event.created_at,
                    event.sensitivity,
                ),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO memory_jobs
                (job_id, event_id, status, updated_at, next_attempt_at)
                VALUES (?, ?, 'pending', ?, 0)
                """,
                (uuid.uuid4().hex, event.event_id, time.time()),
            )
        return event

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            payload = {}
        return MemoryEvent(
            event_id=str(row["event_id"]),
            session_id=str(row["session_id"]),
            run_id=str(row["run_id"]),
            event_type=str(row["event_type"]),
            payload=payload if isinstance(payload, dict) else {},
            source=str(row["source"]),
            capability_id=str(row["capability_id"]),
            created_at=float(row["created_at"]),
            sensitivity=str(row["sensitivity"]),
        )

    def claim(self, *, lock_seconds: float = 30.0) -> tuple[str, MemoryEvent] | None:
        now = time.time()
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE memory_jobs
                SET status='failed', locked_until=0, updated_at=?,
                    last_error=CASE WHEN last_error='' THEN '超过最大重试次数' ELSE last_error END
                WHERE status='processing' AND locked_until < ? AND attempts >= ?
                """,
                (now, now, self.max_attempts),
            )
            row = connection.execute(
                """
                SELECT j.job_id, e.*
                FROM memory_jobs j JOIN memory_events e ON e.event_id = j.event_id
                WHERE (
                    (j.status = 'pending' AND j.next_attempt_at <= ?)
                    OR (j.status = 'processing' AND j.locked_until < ? AND j.attempts < ?)
                )
                ORDER BY j.updated_at, j.job_id
                LIMIT 1
                """,
                (now, now, self.max_attempts),
            ).fetchone()
            if row is None:
                return None
            updated = connection.execute(
                """
                UPDATE memory_jobs
                SET status='processing', attempts=attempts+1,
                    locked_until=?, updated_at=?, next_attempt_at=0
                WHERE job_id=? AND (
                    (status='pending' AND next_attempt_at <= ?)
                    OR (status='processing' AND locked_until < ? AND attempts < ?)
                )
                """,
                (
                    now + max(0.01, lock_seconds),
                    now,
                    row["job_id"],
                    now,
                    now,
                    self.max_attempts,
                ),
            ).rowcount
            if updated != 1:
                return None
            return str(row["job_id"]), self._row_to_event(row)

    def complete(self, job_id: str) -> None:
        with self._lock, closing(self._connect()) as connection, connection:
            connection.execute(
                """
                UPDATE memory_jobs
                SET status='succeeded', locked_until=0, updated_at=?, last_error=''
                WHERE job_id=?
                """,
                (time.time(), job_id),
            )

    def fail(self, job_id: str, error: str, *, retry: bool = True) -> None:
        now = time.time()
        with self._lock, closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT attempts FROM memory_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            attempts = int(row["attempts"]) if row else self.max_attempts
            should_retry = retry and attempts < self.max_attempts
            delay = self.retry_base_seconds * (2 ** max(0, attempts - 1))
            connection.execute(
                """
                UPDATE memory_jobs
                SET status=?, locked_until=0, updated_at=?, next_attempt_at=?, last_error=?
                WHERE job_id=?
                """,
                (
                    "pending" if should_retry else "failed",
                    now,
                    now + min(delay, 60.0) if should_retry else 0,
                    str(error)[:500],
                    job_id,
                ),
            )

    def mark_applied(self, event_id: str) -> bool:
        """Return False when this event was projected before."""
        with self._lock, closing(self._connect()) as connection, connection:
            inserted = connection.execute(
                """
                INSERT OR IGNORE INTO memory_applied_events(event_id, applied_at)
                VALUES (?, ?)
                """,
                (event_id, time.time()),
            ).rowcount
        return inserted == 1

    def is_applied(self, event_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT 1 FROM memory_applied_events WHERE event_id=?", (event_id,)
            ).fetchone()
        return row is not None

    def pending_count(self) -> int:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_jobs WHERE status IN ('pending', 'processing')"
            ).fetchone()
        return int(row["count"] if row else 0)

    def iter_events(self) -> list[MemoryEvent]:
        """Return events in append order so a projection can be rebuilt."""
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM memory_events ORDER BY created_at, event_id"
            ).fetchall()
        return [self._row_to_event(row) for row in rows]


class MemoryProjector:
    """Apply verified events to the existing MemoryStore projection."""

    def __init__(self, event_log: MemoryEventLog, memory_store: Any, profile_path: str = ""):
        self.event_log = event_log
        self.memory_store = memory_store
        self.profile_path = profile_path

    def apply(self, event: MemoryEvent, *, force: bool = False) -> bool:
        if not force and self.event_log.is_applied(event.event_id):
            return False
        payload = dict(event.payload)
        event_type = event.event_type
        if event_type in {"tool_failed", "tool_succeeded"}:
            # Only the execution boundary can set this marker. Text generated
            # by the model is never accepted as long-lived tool experience.
            if payload.get("verified") is not True:
                self.event_log.mark_applied(event.event_id)
                return False
            key = str(payload.get("key") or "")
            value = str(payload.get("value") or "")
            if key and value:
                self.memory_store.add(
                    MemoryScope.EPISODIC,
                    value,
                    key=key,
                    source_id=f"event:{event.event_id}",
                    ttl_seconds=float(payload.get("ttl_seconds", 86400)),
                    capability_tags=tuple(payload.get("capability_tags") or ()),
                    intent_tags=tuple(payload.get("intent_tags") or ()),
                    confidence=float(payload.get("confidence", 0.8)),
                    session_id=event.session_id,
                    now=event.created_at,
                )
        elif event_type == "memory_change":
            operation = str(payload.get("operation") or "")
            scope = MemoryScope(str(payload.get("scope") or ""))
            key = str(payload.get("semantic_key") or "")
            if operation not in {"add", "update", "delete"} or not key:
                raise ValueError("invalid memory_change event")
            record_key = f"profile.{key}"
            if operation == "delete":
                self.memory_store.forget(f"{scope.value}:{record_key}")
            else:
                self.memory_store.add(
                    scope,
                    str(payload.get("value") or ""),
                    key=record_key,
                    source_id=f"event:{event.event_id}",
                    now=event.created_at,
                )
        persist_error = str(getattr(self.memory_store, "last_persist_error", "") or "")
        if persist_error:
            raise OSError(f"memory snapshot persistence failed: {persist_error}")
        # The marker is written only after the projection succeeds. If the
        # process crashes earlier, the job can be claimed and replayed.
        self.event_log.mark_applied(event.event_id)
        return True

    def process_one(self, *, lock_seconds: float = 30.0) -> bool:
        claimed = self.event_log.claim(lock_seconds=lock_seconds)
        if claimed is None:
            return False
        job_id, event = claimed
        try:
            self.apply(event)
        except Exception as exc:  # noqa: BLE001 - worker retries after a crash
            self.event_log.fail(job_id, repr(exc), retry=True)
            return False
        self.event_log.complete(job_id)
        return True

    def drain(self, *, max_events: int = 100) -> int:
        processed = 0
        for _ in range(max(0, max_events)):
            if not self.process_one():
                break
            processed += 1
        return processed

    def replay(self, *, max_events: int | None = None) -> int:
        """Rebuild the projection from the immutable event log."""
        events = self.event_log.iter_events()
        if max_events is not None:
            events = events[-max(0, int(max_events)) :]
        rebuilt = 0
        for event in events:
            if self.apply(event, force=True):
                rebuilt += 1
        return rebuilt


class MemoryWorker:
    """Small daemon worker; shutdown is bounded and safe to call repeatedly."""

    def __init__(self, projector: MemoryProjector, interval: float = 0.1):
        self.projector = projector
        self.interval = max(0.01, float(interval))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="memory-projector", daemon=True)
        self._thread.start()

    def submit(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                processed = self.projector.drain(max_events=20)
            except (OSError, sqlite3.Error):
                # Another process may briefly hold SQLite's write lock, or the
                # owning temporary workspace may already be closing.
                processed = 0
            if processed == 0:
                self._wake.wait(self.interval)
                self._wake.clear()

    def stop(self, *, drain: bool = True, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(max(0.1, timeout))
            if self._thread.is_alive():
                return
        self._thread = None
        if drain:
            try:
                self.projector.drain(max_events=1000)
            except (OSError, sqlite3.Error):
                pass
