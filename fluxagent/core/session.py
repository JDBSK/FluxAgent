from __future__ import annotations

# -*- coding: utf-8 -*-
"""多会话控制面。

设计选择：
  1. 元数据索引与 LangGraph checkpoint 分离——checkpoint 是不透明的 per-thread
     blob 存储，产品功能需要 list/search/rename/archive 而不扫描 sqlite 内部。
  2. session.id == LangGraph thread_id（1:1 映射），隔离天然免费。
  3. 全局 user_profile 保持全局（长期偏好）；会话笔记是任务线程的本地草稿。
  4. 软归档优先；硬删除只删索引行，checkpoint 可能残留（ID 永不复用）。
  5. 本地 CLI 单激活会话（更简单 UX），多会话数据共存。
"""

# -*- coding: utf-8 -*-
"""Multi-session control plane for FluxAgent.

Design choices (interview-ready):
1. Metadata index is separate from LangGraph checkpoint.
   Checkpoint is an opaque per-thread blob store; product features need
   list/search/rename/archive without scanning sqlite internals.
2. session.id == LangGraph thread_id (1:1). Isolation is free once IDs differ.
3. Global user_profile stays global (long-term prefs); session notes are local
   scratch for a task thread. Mixing them causes either preference loss or
   cross-task pollution.
4. Soft archive first; hard delete only removes the index row. Orphan
   checkpoints may remain (accepted): deleting checkpoint rows via saver APIs
   is backend-specific and not worth coupling. IDs are never reused.
5. Single active session in local CLI (simpler UX). Multi-session *data*
   coexists; concurrency across terminals is best-effort via file lock.
"""


import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from .config import SESSIONS_DIR, SESSIONS_INDEX, WORKSPACE_DIR

STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"

LEGACY_THREAD_ID = "local_geek_master"
DEFAULT_TITLE = "主会话"
MAX_TITLE_LEN = 60
MAX_PREVIEW_LEN = 120
MAX_TAGS = 8

_LOCK = threading.RLock()


def _now() -> float:
    return time.time()


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def new_session_id() -> str:
    """生成不会复用的会话标识。"""

    # Never reuse. Short, shell-friendly, stable prefix for resolve().
    return "sess_" + uuid.uuid4().hex[:12]


def normalize_title(title: str, fallback: str = DEFAULT_TITLE) -> str:
    """规范化并限制会话标题长度。"""

    text = re.sub(r"\s+", " ", (title or "").strip())
    if not text:
        text = fallback
    return text[:MAX_TITLE_LEN]


def auto_title_from_text(text: str) -> str:
    """从用户文本生成会话标题。"""

    raw = re.sub(r"\s+", " ", (text or "").strip())
    raw = re.sub(r"^/+", "", raw)
    if not raw:
        return DEFAULT_TITLE
    # Drop slash commands from auto-title source
    if raw.startswith("/"):
        return DEFAULT_TITLE
    return normalize_title(raw, fallback=DEFAULT_TITLE)


def thread_config(session_id: str, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """构建绑定会话线程的运行配置。"""

    cfg = {"configurable": {"thread_id": session_id}}
    if extra:
        cfg["configurable"].update(extra)
    return cfg


@dataclass
class SessionRecord:
    """保存单个会话的索引元数据。"""

    id: str
    title: str
    status: str = STATUS_ACTIVE
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    last_active_at: float = field(default_factory=_now)
    message_count: int = 0
    last_preview: str = ""
    tags: list[str] = field(default_factory=list)
    pinned: bool = False
    origin: str = "user"  # user | migrated | system

    def to_dict(self) -> dict[str, Any]:
        """将会话记录转换为可序列化字典。"""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionRecord":
        """从索引字典恢复会话记录。"""

        tags = data.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        return cls(
            id=str(data.get("id") or ""),
            title=normalize_title(str(data.get("title") or DEFAULT_TITLE)),
            status=str(data.get("status") or STATUS_ACTIVE),
            created_at=float(data.get("created_at") or _now()),
            updated_at=float(data.get("updated_at") or _now()),
            last_active_at=float(data.get("last_active_at") or _now()),
            message_count=int(data.get("message_count") or 0),
            last_preview=str(data.get("last_preview") or "")[:MAX_PREVIEW_LEN],
            tags=[str(t)[:24] for t in tags[:MAX_TAGS]],
            pinned=bool(data.get("pinned") or False),
            origin=str(data.get("origin") or "user"),
        )


class SessionError(ValueError):
    """User-facing session operation error."""


class SessionStore:
    """JSON index + per-session note files, process-safe via RLock + atomic replace.

    Not a distributed lock. For multi-process CLI, last writer wins on the
    whole index document — acceptable for local single-user product.
    """

    def __init__(self, index_path: Optional[str] = None, root_dir: Optional[str] = None):
        self.index_path = index_path or SESSIONS_INDEX
        self.root_dir = root_dir or SESSIONS_DIR
        os.makedirs(self.root_dir, exist_ok=True)

    # ---- persistence ----
    def _read(self) -> dict[str, Any]:
        if not os.path.exists(self.index_path):
            return {"version": 1, "active_id": None, "sessions": {}}
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("bad index")
            data.setdefault("version", 1)
            data.setdefault("active_id", None)
            data.setdefault("sessions", {})
            if not isinstance(data["sessions"], dict):
                data["sessions"] = {}
            return data
        except (OSError, json.JSONDecodeError, ValueError):
            # Corrupt index: quarantine and start clean rather than crash CLI.
            bak = self.index_path + f".corrupt.{int(_now())}"
            try:
                os.replace(self.index_path, bak)
            except OSError:
                pass
            return {"version": 1, "active_id": None, "sessions": {}}

    def _write(self, data: dict[str, Any]) -> None:
        _atomic_write_json(self.index_path, data)

    def _session_dir(self, session_id: str) -> str:
        safe = "".join(c for c in session_id if c.isalnum() or c in "-_") or "unknown"
        path = os.path.join(self.root_dir, safe)
        os.makedirs(path, exist_ok=True)
        return path

    def notes_path(self, session_id: str) -> str:
        """返回指定会话的笔记文件路径。"""

        return os.path.join(self._session_dir(session_id), "notes.md")

    # ---- bootstrap ----
    def bootstrap(self) -> SessionRecord:
        """Ensure at least one session exists; preserve legacy thread continuity."""
        with _LOCK:
            data = self._read()
            sessions = data["sessions"]
            if not sessions:
                rec = SessionRecord(
                    id=LEGACY_THREAD_ID,
                    title=DEFAULT_TITLE,
                    origin="migrated",
                )
                sessions[rec.id] = rec.to_dict()
                data["active_id"] = rec.id
                self._write(data)
                self._session_dir(rec.id)
                return rec

            active_id = data.get("active_id")
            if active_id and active_id in sessions:
                rec = SessionRecord.from_dict(sessions[active_id])
                if rec.status == STATUS_ARCHIVED:
                    # Active pointer must not point at archived.
                    data["active_id"] = None
                else:
                    return rec

            # Pick newest non-archived, else unarchive newest.
            recs = [SessionRecord.from_dict(v) for v in sessions.values()]
            live = [r for r in recs if r.status != STATUS_ARCHIVED]
            chosen = sorted(live or recs, key=lambda r: (r.pinned, r.last_active_at), reverse=True)[0]
            if chosen.status == STATUS_ARCHIVED:
                chosen.status = STATUS_ACTIVE
                chosen.updated_at = _now()
                sessions[chosen.id] = chosen.to_dict()
            data["active_id"] = chosen.id
            self._write(data)
            return chosen

    # ---- CRUD ----
    def create(
        self,
        title: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        make_active: bool = True,
        session_id: Optional[str] = None,
        origin: str = "user",
    ) -> SessionRecord:
        """创建会话并按需将其设为当前会话。"""

        with _LOCK:
            data = self._read()
            sid = session_id or new_session_id()
            if sid in data["sessions"]:
                raise SessionError(f"session id already exists: {sid}")
            rec = SessionRecord(
                id=sid,
                title=normalize_title(title or f"会话 {len(data['sessions']) + 1}"),
                tags=[str(t)[:24] for t in (tags or [])][:MAX_TAGS],
                origin=origin,
            )
            data["sessions"][sid] = rec.to_dict()
            if make_active:
                data["active_id"] = sid
            self._write(data)
            self._session_dir(sid)
            return rec

    def get(self, session_id: str) -> Optional[SessionRecord]:
        """按标识获取会话记录。"""

        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            return SessionRecord.from_dict(raw) if raw else None

    def get_active(self) -> Optional[SessionRecord]:
        """获取当前激活的会话记录。"""

        with _LOCK:
            data = self._read()
            aid = data.get("active_id")
            if not aid:
                return None
            raw = data["sessions"].get(aid)
            return SessionRecord.from_dict(raw) if raw else None

    def list(
        self,
        include_archived: bool = False,
        query: Optional[str] = None,
    ) -> list[SessionRecord]:
        """列出会话，并可按归档状态和关键词筛选。"""

        with _LOCK:
            data = self._read()
            recs = [SessionRecord.from_dict(v) for v in data["sessions"].values()]
        if not include_archived:
            recs = [r for r in recs if r.status != STATUS_ARCHIVED]
        if query:
            q = query.strip().lower()
            recs = [
                r
                for r in recs
                if q in r.id.lower()
                or q in r.title.lower()
                or q in (r.last_preview or "").lower()
                or any(q in t.lower() for t in r.tags)
            ]
        recs.sort(key=lambda r: (not r.pinned, -r.last_active_at, r.title))
        return recs

    def resolve(self, ref: str, include_archived: bool = True) -> SessionRecord:
        """Resolve by exact id, unique id prefix, or unique title substring."""
        text = (ref or "").strip()
        if not text:
            raise SessionError("empty session reference")
        with _LOCK:
            data = self._read()
            recs = [SessionRecord.from_dict(v) for v in data["sessions"].values()]
        if not include_archived:
            recs = [r for r in recs if r.status != STATUS_ARCHIVED]

        for r in recs:
            if r.id == text:
                return r
        prefix_hits = [r for r in recs if r.id.startswith(text)]
        if len(prefix_hits) == 1:
            return prefix_hits[0]
        if len(prefix_hits) > 1:
            raise SessionError(f"ambiguous id prefix '{text}': {[h.id for h in prefix_hits[:5]]}")

        title_hits = [r for r in recs if text.lower() in r.title.lower()]
        if len(title_hits) == 1:
            return title_hits[0]
        if len(title_hits) > 1:
            raise SessionError(
                f"ambiguous title '{text}': " + ", ".join(f"{h.id}({h.title})" for h in title_hits[:5])
            )
        raise SessionError(f"session not found: {text}")

    def set_active(self, session_id: str) -> SessionRecord:
        """将未归档会话设为当前会话。"""

        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            if not raw:
                raise SessionError(f"session not found: {session_id}")
            rec = SessionRecord.from_dict(raw)
            if rec.status == STATUS_ARCHIVED:
                raise SessionError("cannot activate archived session; unarchive first")
            data["active_id"] = session_id
            rec.last_active_at = _now()
            rec.updated_at = rec.last_active_at
            data["sessions"][session_id] = rec.to_dict()
            self._write(data)
            return rec

    def rename(self, session_id: str, title: str) -> SessionRecord:
        """更新指定会话的标题。"""

        title = normalize_title(title)
        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            if not raw:
                raise SessionError(f"session not found: {session_id}")
            rec = SessionRecord.from_dict(raw)
            rec.title = title
            rec.updated_at = _now()
            data["sessions"][session_id] = rec.to_dict()
            self._write(data)
            return rec

    def archive(self, session_id: str) -> SessionRecord:
        """归档非当前会话。"""

        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            if not raw:
                raise SessionError(f"session not found: {session_id}")
            if data.get("active_id") == session_id:
                raise SessionError("cannot archive the active session; switch first")
            rec = SessionRecord.from_dict(raw)
            rec.status = STATUS_ARCHIVED
            rec.pinned = False
            rec.updated_at = _now()
            data["sessions"][session_id] = rec.to_dict()
            self._write(data)
            return rec

    def unarchive(self, session_id: str) -> SessionRecord:
        """恢复已归档会话。"""

        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            if not raw:
                raise SessionError(f"session not found: {session_id}")
            rec = SessionRecord.from_dict(raw)
            rec.status = STATUS_ACTIVE
            rec.updated_at = _now()
            data["sessions"][session_id] = rec.to_dict()
            self._write(data)
            return rec

    def delete(self, session_id: str) -> None:
        """Hard-delete index row. Active session forbidden. Checkpoint may orphan."""
        with _LOCK:
            data = self._read()
            if session_id not in data["sessions"]:
                raise SessionError(f"session not found: {session_id}")
            if data.get("active_id") == session_id:
                raise SessionError("cannot delete the active session; switch first")
            del data["sessions"][session_id]
            self._write(data)

    def pin(self, session_id: str, pinned: bool = True) -> SessionRecord:
        """设置或取消会话置顶状态。"""

        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            if not raw:
                raise SessionError(f"session not found: {session_id}")
            rec = SessionRecord.from_dict(raw)
            if rec.status == STATUS_ARCHIVED and pinned:
                raise SessionError("cannot pin archived session")
            rec.pinned = bool(pinned)
            rec.updated_at = _now()
            data["sessions"][session_id] = rec.to_dict()
            self._write(data)
            return rec

    def touch(
        self,
        session_id: str,
        preview: Optional[str] = None,
        message_delta: int = 0,
        auto_title_from: Optional[str] = None,
    ) -> SessionRecord:
        """更新会话活跃时间、预览和消息统计。"""

        with _LOCK:
            data = self._read()
            raw = data["sessions"].get(session_id)
            if not raw:
                raise SessionError(f"session not found: {session_id}")
            rec = SessionRecord.from_dict(raw)
            now = _now()
            rec.last_active_at = now
            rec.updated_at = now
            if message_delta:
                rec.message_count = max(0, rec.message_count + int(message_delta))
            if preview is not None:
                rec.last_preview = str(preview).strip()[:MAX_PREVIEW_LEN]
            # Auto-title only when still default-ish and first real turn.
            if auto_title_from and rec.message_count <= 1:
                if rec.title in {DEFAULT_TITLE, "会话", rec.id} or rec.title.startswith("会话 "):
                    rec.title = auto_title_from_text(auto_title_from)
            data["sessions"][session_id] = rec.to_dict()
            self._write(data)
            return rec

    def read_notes(self, session_id: str) -> str:
        """读取指定会话的笔记内容。"""

        path = self.notes_path(session_id)
        if not os.path.exists(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def write_notes(self, session_id: str, content: str, append: bool = False) -> str:
        """覆盖或追加指定会话的笔记。"""

        if not self.get(session_id):
            raise SessionError(f"session not found: {session_id}")
        path = self.notes_path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            if append and content and not content.endswith("\n"):
                content = content + "\n"
            f.write(content)
        return path

    def stats(self) -> dict[str, Any]:
        """汇总会话数量和当前激活状态。"""

        recs = self.list(include_archived=True)
        return {
            "total": len(recs),
            "active": sum(1 for r in recs if r.status == STATUS_ACTIVE),
            "archived": sum(1 for r in recs if r.status == STATUS_ARCHIVED),
            "pinned": sum(1 for r in recs if r.pinned),
            "active_id": (self.get_active().id if self.get_active() else None),
        }


# ---- presentation / command parsing (CLI + tests share) ----

SESSION_COMMANDS = {
    "help",
    "session",
    "sessions",
    "new",
    "switch",
    "rename",
    "archive",
    "unarchive",
    "delete",
    "pin",
    "unpin",
    "history",
    "notes",
    "exit",
    "quit",
}


def is_slash_command(text: str) -> bool:
    """判断文本是否为斜杠命令。"""

    return bool((text or "").strip().startswith("/"))


def parse_slash_command(text: str) -> tuple[str, str]:
    """解析斜杠命令名称及其参数。"""

    raw = (text or "").strip()
    if not raw.startswith("/"):
        return "", ""
    body = raw[1:].strip()
    if not body:
        return "", ""
    parts = body.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""
    return cmd, arg


def format_session_line(rec: SessionRecord, active_id: Optional[str] = None) -> str:
    """将单个会话格式化为列表行。"""

    mark = "*" if rec.id == active_id else " "
    pin = " pin" if rec.pinned else ""
    st = "" if rec.status == STATUS_ACTIVE else f" [{rec.status}]"
    preview = (rec.last_preview or "").replace("\n", " ")
    if preview:
        preview = " | " + preview[:40]
    return f"{mark} {rec.id}  {rec.title}{st}{pin}  msgs={rec.message_count}{preview}"


def format_session_list(recs: list[SessionRecord], active_id: Optional[str] = None) -> str:
    """将会话记录格式化为多行列表。"""

    if not recs:
        return "（无会话）"
    lines = [format_session_line(r, active_id) for r in recs]
    return "\n".join(lines)


def format_history_messages(messages: list[Any], limit: int = 12) -> str:
    """Render recent Human/AI messages for /history. Skip tool noise by default."""
    if limit <= 0:
        limit = 12
    rows: list[str] = []
    for msg in messages:
        mtype = getattr(msg, "type", None) or msg.__class__.__name__.lower()
        content = str(getattr(msg, "content", "") or "").strip()
        if not content:
            continue
        if mtype in {"tool", "toolmessage"}:
            continue
        if mtype in {"system", "systemmessage"}:
            continue
        role = "You" if mtype in {"human", "humanmessage"} else "AI"
        if mtype in {"ai", "aimessage"} and getattr(msg, "tool_calls", None):
            # tool-call only turns: show names
            names = [tc.get("name", "?") for tc in (msg.tool_calls or [])]
            content = f"(tool_calls: {', '.join(names)})"
        rows.append(f"{role}: {content[:240]}")
    tail = rows[-limit:]
    if not tail:
        return "（当前会话暂无对话历史）"
    return "\n".join(tail)


def extract_message_count(messages: list[Any]) -> int:
    """统计消息集合中的用户和助手消息数。"""

    n = 0
    for msg in messages:
        mtype = getattr(msg, "type", None) or ""
        if mtype in {"human", "ai"}:
            n += 1
    return n


def help_text() -> str:
    """返回会话斜杠命令的帮助文本。"""

    return "\n".join(
        [
            "会话命令:",
            "  /session              当前会话",
            "  /sessions [all|q]     列表 / 含归档 / 搜索",
            "  /new [标题]           新建并切换",
            "  /switch <id|标题>     切换会话（审批中禁止）",
            "  /rename <标题>        重命名当前会话",
            "  /archive [ref]        归档（不能归档当前）",
            "  /unarchive <ref>      恢复归档",
            "  /delete <ref>         删除索引（不能删当前）",
            "  /pin | /unpin [ref]   置顶",
            "  /history [n]          查看当前会话历史",
            "  /notes [文本]         查看或追加会话笔记",
            "  /help                 帮助",
            "  /exit                 退出",
            "",
            "隔离模型: session.id == checkpoint thread_id；全局画像共享，会话笔记隔离。",
        ]
    )
