# -*- coding: utf-8 -*-
"""Structured memory storage and one-time legacy profile migration."""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .retrieval import _atomic_write_json


class MemoryScope(str, Enum):
    USER_FACT = "user_fact"
    USER_PREFERENCE = "user_preference"
    EPISODIC = "episodic"
    SKILL_CONTEXT = "skill_context"


SENSITIVITY_LEVELS = frozenset({"normal", "sensitive"})
MEMORY_FILE_FORMAT = "1"


@dataclass(frozen=True)
class MemoryRecord:
    """验证后的记忆值，具有作用域、新鲜度和可见性边界。这个记忆是不可变的，更新会创建一个新的实例。"""

    record_id: str
    scope: MemoryScope
    value: str
    source_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl_seconds: Optional[float] = None
    capability_tags: Tuple[str, ...] = ()
    intent_tags: Tuple[str, ...] = ()
    sensitivity: str = "normal"
    confidence: float = 1.0
    session_id: str = ""

    def __post_init__(self) -> None:
        if self.sensitivity not in SENSITIVITY_LEVELS:
            raise ValueError(f"非法敏感级别: {self.sensitivity}")

    def is_expired(self, now: Optional[float] = None) -> bool:
        if self.ttl_seconds is None:
            return False
        reference = now if now is not None else time.time()
        return reference - self.created_at > self.ttl_seconds


class MemoryStore:
    """记忆存储，存储在内存中，并可选择持久化到磁盘。"""

    def __init__(self, persist_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()  # 用于保护对 _records 和 _counter 的访问
        self._records: dict[str, MemoryRecord] = {}  # 存储所有记忆记录，键为 record_id
        self._counter = 0  # 用于生成唯一的记忆记录ID的计数器

        self.persist_path = str(persist_path) if persist_path else ""  # 持久化文件路径
        self.last_persist_error: str = ""  # 如果持久化出现错误，就记录上一次持久化操作的错误信息

    def _auto_save(self) -> None:
        if not self.persist_path:
            return
        try:
            self.save(self.persist_path)
            self.last_persist_error = ""
        except OSError as exc:
            self.last_persist_error = f"{type(exc).__name__}: {exc}"[:200]

    def save(self, path: str) -> None:
        with self._lock:
            payload = {
                "format": MEMORY_FILE_FORMAT,
                "counter": self._counter,
                "records": [
                    {
                        "record_id": record.record_id,
                        "scope": record.scope.value,
                        "value": record.value,
                        "source_id": record.source_id,
                        "created_at": record.created_at,
                        "updated_at": record.updated_at,
                        "ttl_seconds": record.ttl_seconds,
                        "capability_tags": list(record.capability_tags),
                        "intent_tags": list(record.intent_tags),
                        "sensitivity": record.sensitivity,
                        "confidence": record.confidence,
                        "session_id": record.session_id,
                    }
                    for record in self._records.values()
                ],
            }
        _atomic_write_json(path, payload)

    def load(self, path: str) -> int:
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return 0
        if not isinstance(payload, dict) or payload.get("format") != MEMORY_FILE_FORMAT:
            return 0
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            return 0

        now = time.time()
        loaded = 0
        max_counter = 0
        with self._lock:
            for item in raw_records:
                if not isinstance(item, dict):
                    continue
                try:
                    record = MemoryRecord(
                        record_id=str(item["record_id"]),
                        scope=MemoryScope(item["scope"]),
                        value=str(item.get("value", "")),
                        source_id=str(item.get("source_id", "")),
                        created_at=float(item.get("created_at", 0.0)),
                        updated_at=float(item.get("updated_at", 0.0)),
                        ttl_seconds=(
                            float(item["ttl_seconds"])
                            if item.get("ttl_seconds") is not None
                            else None
                        ),
                        capability_tags=tuple(
                            str(tag) for tag in item.get("capability_tags", ())
                        ),
                        intent_tags=tuple(
                            str(tag) for tag in item.get("intent_tags", ())
                        ),
                        sensitivity=str(item.get("sensitivity", "normal")),
                        confidence=float(item.get("confidence", 1.0)),
                        session_id=str(item.get("session_id", "")),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                if record.is_expired(now):
                    continue
                self._records[record.record_id] = record
                loaded += 1
                if record.record_id.startswith("mem-"):
                    try:
                        max_counter = max(
                            max_counter, int(record.record_id[4:]))
                    except ValueError:
                        pass
            self._counter = max(self._counter, max_counter)
        return loaded

    def add(
        self,
        scope: MemoryScope,
        value: str,
        *,
        key: Optional[str] = None,
        source_id: str = "",
        ttl_seconds: Optional[float] = None,
        capability_tags: Sequence[str] = (),
        intent_tags: Sequence[str] = (),
        sensitivity: str = "normal",
        confidence: float = 1.0,
        session_id: str = "",
        now: Optional[float] = None,
    ) -> MemoryRecord:
        if "@@" in str(key or "") or "@@" in str(session_id or ""):
            raise ValueError('memory key/session_id 不允许包含 "@@"')
        with self._lock:
            self._counter += 1
            if key and session_id:
                record_id = f"{scope.value}:{key}@@{session_id}"
            elif key:
                record_id = f"{scope.value}:{key}"
            else:
                record_id = f"mem-{self._counter}"
            previous = self._records.get(record_id)
            timestamp = now if now is not None else time.time()
            capability_tags = tuple(capability_tags)
            intent_tags = tuple(intent_tags)
            unchanged = previous is not None and (
                previous.value == str(value)
                and previous.source_id == source_id
                and previous.ttl_seconds == ttl_seconds
                and previous.capability_tags == capability_tags
                and previous.intent_tags == intent_tags
                and previous.sensitivity == sensitivity
                and previous.confidence == confidence
                and previous.session_id == str(session_id or "")
            )
            record = MemoryRecord(
                record_id=record_id,
                scope=scope,
                value=str(value),
                source_id=source_id,
                created_at=previous.created_at if previous is not None else timestamp,
                updated_at=previous.updated_at if unchanged else timestamp,
                ttl_seconds=ttl_seconds,
                capability_tags=capability_tags,
                intent_tags=intent_tags,
                sensitivity=sensitivity,
                confidence=confidence,
                session_id=str(session_id or ""),
            )
            self._records[record_id] = record
        self._auto_save()
        return record

    def remove_namespace(self, prefix: str, keep_ids: Sequence[str] = ()) -> int:
        """移除所有以指定前缀开头的记录，除了指定的保留记录ID。"""
        keep = set(keep_ids)
        with self._lock:
            stale = [
                record_id
                for record_id in self._records
                if record_id.startswith(prefix) and record_id not in keep
            ]
            for record_id in stale:
                del self._records[record_id]
        if stale:
            self._auto_save()
        return len(stale)

    def query(
        self,
        *,
        scopes: Optional[Sequence[MemoryScope]] = None,
        capability_id: Optional[str] = None,
        capability_ids: Optional[Sequence[str]] = None,
        intent: Optional[str] = None,
        include_sensitive: bool = False,
        session_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> List[MemoryRecord]:
        with self._lock:
            candidates = list(self._records.values())
            if now is None:
                expired_ids = {
                    record.record_id
                    for record in candidates
                    if record.is_expired(None)
                }
                for record_id in expired_ids:
                    record = self._records.get(record_id)
                    if record is not None and record.is_expired(None):
                        del self._records[record_id]
            else:
                expired_ids = {
                    record.record_id for record in candidates if record.is_expired(now)
                }

        result: List[MemoryRecord] = []
        for record in candidates:
            if record.record_id in expired_ids:
                continue
            if scopes is not None and record.scope not in scopes:
                continue
            if not include_sensitive and record.sensitivity == "sensitive":
                continue
            if record.session_id and (
                session_id is None or record.session_id != session_id
            ):
                continue
            if record.capability_tags:
                allowed = set(capability_ids or ()) or (
                    {capability_id} if capability_id else set()
                )
                if not allowed or not (allowed & set(record.capability_tags)):
                    continue
            if intent and record.intent_tags and intent not in record.intent_tags:
                continue
            result.append(record)
        result.sort(key=lambda record: (-record.confidence, record.record_id))
        return result

    def forget(self, record_id: str) -> bool:
        with self._lock:
            removed = self._records.pop(record_id, None) is not None
        if removed:
            self._auto_save()
        return removed

    def forget_prefix(self, prefix: str) -> int:
        with self._lock:
            stale = [
                record_id for record_id in self._records if record_id.startswith(prefix)]
            for record_id in stale:
                del self._records[record_id]
            removed = len(stale)
        if removed:
            self._auto_save()
        return removed

    def all_records(self) -> List[MemoryRecord]:
        with self._lock:
            return list(self._records.values())


class ProfileCompatProvider:
    """One-time importer for the legacy Markdown user profile."""

    def __init__(self, profile_path: str) -> None:
        self.profile_path = str(profile_path)

    def load_into(self, store: MemoryStore) -> int:
        records = self._parse()
        for scope, value, key, source_id in records:
            store.add(scope, value, key=key, source_id=source_id)
        return len(records)

    def migrate_once(self, store: MemoryStore) -> int:
        """Import the legacy profile once, then preserve an immutable backup."""
        if not os.path.isfile(self.profile_path):
            return 0
        backup_path = self.profile_path + ".migrated.bak"
        if os.path.exists(backup_path):
            return 0
        imported = self.load_into(store)
        if store.persist_path:
            store.save(store.persist_path)
        shutil.copy2(self.profile_path, backup_path)
        return imported

    def _parse(self) -> list[tuple[MemoryScope, str, str, str]]:
        """解析 Markdown 用户资料文件，返回 (scope, value, key, source_id) 的元组列表。"""
        if not os.path.exists(self.profile_path):
            return []
        try:
            with open(self.profile_path, encoding="utf-8", errors="ignore") as handle:
                content = handle.read()
        except OSError:
            return []
        records = []
        for line in content.splitlines():
            match = re.match(
                r"^[-*]\s*\*\*(.+?)\*\*[：:]\s*(.+)$", line.strip())
            if not match:
                continue
            key, value = match.group(1).strip(), match.group(2).strip()
            if not key or not value:
                continue
            canonical_key = normalize_profile_key(key)
            records.append(
                (
                    MemoryScope.USER_FACT,
                    f"{key}：{value}",
                    f"profile.{canonical_key}",
                    f"user_profile.md#{key}",
                )
            )
        return records


def normalize_profile_key(label: str) -> str:
    raw = str(label or "").strip().casefold()
    normalized = re.sub(r"[^\w\u3400-\u9fff]+", "_", raw, flags=re.UNICODE)
    return normalized.strip("_") or "unnamed"
