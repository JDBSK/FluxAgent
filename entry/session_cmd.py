from __future__ import annotations

# -*- coding: utf-8 -*-
"""Slash 命令分发器：处理 /new /switch /sessions /history /notes 等会话管理命令。

保持在 SessionStore 之外，使 store 不依赖 LangGraph/CLI。
所有状态读取走 async 接口（await apending_approval / await app.aget_state），
因为 checkpointer 是 AsyncSqliteSaver。
"""

# -*- coding: utf-8 -*-
"""Slash-command dispatcher for multi-session runtime.

Kept outside pure SessionStore so store stays free of LangGraph/CLI deps.
"""


from typing import Any, Callable, Optional

from fluxagent.core.approval import apending_approval
from fluxagent.core.session import (
    SessionError,
    SessionStore,
    format_history_messages,
    format_session_list,
    help_text,
    parse_slash_command,
    thread_config,
)


def _active_id(store: SessionStore) -> Optional[str]:
    cur = store.get_active()
    return cur.id if cur else None


async def handle_slash_command(
    text: str,
    store: SessionStore,
    app: Any,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """Return (handled, message). If handled, caller must not send text to agent."""
    raw = (text or "").strip()
    if not raw.startswith("/"):
        return False, ""

    cmd, arg = parse_slash_command(raw)
    if not cmd:
        return True, help_text()

    try:
        if cmd in {"exit", "quit"}:
            # Caller still needs to break loops; we only acknowledge.
            return True, "__EXIT__"

        if cmd == "help":
            return True, help_text()

        if cmd == "session":
            cur = store.get_active()
            if not cur:
                return True, "当前无激活会话，请 /new 创建一个。"
            note_len = len(store.read_notes(cur.id) or "")
            return True, "\n".join(
                [
                    f"当前会话: {cur.title}",
                    f"  id: {cur.id}",
                    f"  status: {cur.status}  pinned={cur.pinned}",
                    f"  messages≈{cur.message_count}  notes_chars={note_len}",
                    f"  preview: {cur.last_preview or '-'}",
                    f"  thread_id(=checkpoint): {config.get('configurable', {}).get('thread_id')}",
                ]
            )

        if cmd == "sessions":
            include_all = False
            query = None
            token = (arg or "").strip()
            if token.lower() in {"all", "-a", "--all"}:
                include_all = True
            elif token:
                # all <q> or just query
                parts = token.split(None, 1)
                if parts[0].lower() in {"all", "-a", "--all"}:
                    include_all = True
                    query = parts[1] if len(parts) > 1 else None
                else:
                    query = token
            recs = store.list(include_archived=include_all, query=query)
            header = f"会话列表 ({len(recs)})" + (" [含归档]" if include_all else "")
            return True, header + "\n" + format_session_list(recs, _active_id(store))

        if cmd == "new":
            # Refuse creating/switching while approval pending on current thread.
            if await apending_approval(app, config):
                return True, "当前会话有待审批操作，请先 y/n 处理完再新建会话。"
            title = arg.strip() if arg.strip() else None
            rec = store.create(title=title, make_active=True)
            config["configurable"]["thread_id"] = rec.id
            return True, f"已新建并切换到会话: {rec.title} ({rec.id})"

        if cmd == "switch":
            if not arg.strip():
                return True, "用法: /switch <id|标题>"
            if await apending_approval(app, config):
                return True, "当前会话有待审批操作，禁止切换（避免审批落到错误线程）。"
            rec = store.resolve(arg.strip(), include_archived=False)
            store.set_active(rec.id)
            config["configurable"]["thread_id"] = rec.id
            # Show a short history preview after switch.
            hist = ""
            try:
                snap = await app.aget_state(config)
                msgs = (snap.values or {}).get("messages") or []
                hist = format_history_messages(msgs, limit=6)
            except Exception:
                hist = "（无法读取历史，可能是空会话）"
            return True, f"已切换到: {rec.title} ({rec.id})\n--- 最近历史 ---\n{hist}"

        if cmd == "rename":
            if not arg.strip():
                return True, "用法: /rename <新标题>"
            cur = store.get_active()
            if not cur:
                return True, "无激活会话"
            rec = store.rename(cur.id, arg.strip())
            return True, f"已重命名: {rec.title}"

        if cmd == "archive":
            ref = arg.strip()
            cur = store.get_active()
            if ref:
                target = store.resolve(ref, include_archived=True)
            else:
                return True, "用法: /archive <id|标题>  （不能归档当前会话，请先 /switch）"
            if cur and target.id == cur.id:
                return True, "不能归档当前会话，请先 /switch 到其他会话。"
            rec = store.archive(target.id)
            return True, f"已归档: {rec.title} ({rec.id})"

        if cmd == "unarchive":
            if not arg.strip():
                return True, "用法: /unarchive <id|标题>"
            target = store.resolve(arg.strip(), include_archived=True)
            rec = store.unarchive(target.id)
            return True, f"已恢复: {rec.title} ({rec.id})"

        if cmd == "delete":
            if not arg.strip():
                return True, "用法: /delete <id|标题>"
            target = store.resolve(arg.strip(), include_archived=True)
            cur = store.get_active()
            if cur and target.id == cur.id:
                return True, "不能删除当前会话，请先 /switch。"
            store.delete(target.id)
            return True, f"已删除会话索引: {target.id}（checkpoint 可能残留，id 不复用）"

        if cmd == "pin":
            cur = store.get_active()
            target = store.resolve(arg.strip(), include_archived=False) if arg.strip() else cur
            if not target:
                return True, "无目标会话"
            rec = store.pin(target.id, True)
            return True, f"已置顶: {rec.title}"

        if cmd == "unpin":
            cur = store.get_active()
            target = store.resolve(arg.strip(), include_archived=True) if arg.strip() else cur
            if not target:
                return True, "无目标会话"
            rec = store.pin(target.id, False)
            return True, f"已取消置顶: {rec.title}"

        if cmd == "history":
            limit = 12
            if arg.strip().isdigit():
                limit = max(1, min(50, int(arg.strip())))
            try:
                snap = await app.aget_state(config)
                msgs = (snap.values or {}).get("messages") or []
            except Exception as exc:
                return True, f"读取历史失败: {exc}"
            return True, format_history_messages(msgs, limit=limit)

        if cmd == "notes":
            cur = store.get_active()
            if not cur:
                return True, "无激活会话"
            if not arg.strip():
                content = store.read_notes(cur.id)
                return True, content if content.strip() else "（会话笔记为空。用法: /notes 文本 追加一行）"
            path = store.write_notes(cur.id, arg.strip() + "\n", append=True)
            return True, f"已写入会话笔记: {path}"

        # Unknown slash: still handle so we don't leak to LLM as user intent noise.
        return True, f"未知命令 /{cmd}。输入 /help 查看会话命令。"

    except SessionError as exc:
        return True, f"会话操作失败: {exc}"
    except Exception as exc:
        return True, f"会话命令异常: {type(exc).__name__}: {exc}"


def sync_config_to_active(store: SessionStore, config: dict[str, Any]) -> str:
    """Ensure config thread_id matches active session; return session id."""
    active = store.bootstrap()
    config.setdefault("configurable", {})
    config["configurable"]["thread_id"] = active.id
    return active.id
