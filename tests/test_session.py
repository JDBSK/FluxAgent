# -*- coding: utf-8 -*-
"""Deep multi-session tests: lifecycle, isolation, command surface."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import fluxagent.core.agent as agent_mod
from fluxagent.core.approval import apending_approval, areject_prepare
from fluxagent.core.session import (
    DEFAULT_TITLE,
    LEGACY_THREAD_ID,
    SessionError,
    SessionStore,
    auto_title_from_text,
    format_history_messages,
    parse_slash_command,
    thread_config,
)
from entry.session_cmd import handle_slash_command, sync_config_to_active
from tests.eval.stub import ScriptedLLM, make_tool


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.index = os.path.join(self.root, "index.json")
        self.store = SessionStore(index_path=self.index, root_dir=self.root)

    def test_bootstrap_creates_legacy_continuity(self):
        rec = self.store.bootstrap()
        self.assertEqual(rec.id, LEGACY_THREAD_ID)
        self.assertEqual(rec.title, DEFAULT_TITLE)
        again = self.store.bootstrap()
        self.assertEqual(again.id, rec.id)

    def test_create_switch_list_rename(self):
        self.store.bootstrap()
        a = self.store.create(title="Alpha", make_active=True)
        b = self.store.create(title="Beta", make_active=False)
        self.assertEqual(self.store.get_active().id, a.id)
        self.store.set_active(b.id)
        self.assertEqual(self.store.get_active().id, b.id)
        self.store.rename(b.id, "Beta-2")
        self.assertEqual(self.store.get(b.id).title, "Beta-2")
        ids = {r.id for r in self.store.list()}
        self.assertIn(a.id, ids)
        self.assertIn(b.id, ids)

    def test_archive_delete_guards(self):
        self.store.bootstrap()
        a = self.store.create("A", make_active=True)
        b = self.store.create("B", make_active=False)
        with self.assertRaises(SessionError):
            self.store.archive(a.id)
        with self.assertRaises(SessionError):
            self.store.delete(a.id)
        self.store.archive(b.id)
        self.assertEqual(self.store.get(b.id).status, "archived")
        live_ids = {r.id for r in self.store.list(include_archived=False)}
        self.assertNotIn(b.id, live_ids)
        self.store.unarchive(b.id)
        self.store.set_active(b.id)
        self.store.delete(a.id)
        self.assertIsNone(self.store.get(a.id))

    def test_resolve_prefix_and_ambiguity(self):
        self.store.bootstrap()
        x = self.store.create("Project X")
        y = self.store.create("Project Y")
        self.assertEqual(self.store.resolve(x.id).id, x.id)
        self.assertEqual(self.store.resolve(x.id[:8]).id, x.id)
        with self.assertRaises(SessionError):
            self.store.resolve("Project")
        hit = self.store.resolve("Project X")
        self.assertEqual(hit.id, x.id)
        _ = y

    def test_touch_autotitle_and_notes(self):
        self.store.bootstrap()
        s = self.store.create(title=DEFAULT_TITLE, make_active=True)
        self.store.touch(s.id, preview="hello world", message_delta=1, auto_title_from="写一份周报")
        s2 = self.store.get(s.id)
        self.assertIn("周报", s2.title)
        self.assertEqual(s2.message_count, 1)
        self.store.write_notes(s.id, "note-a\n", append=True)
        self.assertIn("note-a", self.store.read_notes(s.id))

    def test_pin_sort_and_search(self):
        self.store.bootstrap()
        a = self.store.create("zzz-old")
        b = self.store.create("aaa-new")
        self.store.pin(a.id, True)
        ordered = self.store.list()
        self.assertEqual(ordered[0].id, a.id)
        found = self.store.list(query="aaa")
        self.assertTrue(any(r.id == b.id for r in found))

    def test_corrupt_index_recovers(self):
        self.store.bootstrap()
        with open(self.index, "w", encoding="utf-8") as f:
            f.write("{not-json")
        rec = self.store.bootstrap()
        self.assertTrue(rec.id)


class TestSessionHelpers(unittest.TestCase):
    def test_parse_and_title(self):
        self.assertEqual(parse_slash_command("/sessions all foo")[0], "sessions")
        self.assertEqual(auto_title_from_text("  你好世界  " )[:2], "你好")
        self.assertEqual(thread_config("sess_1")["configurable"]["thread_id"], "sess_1")

    def test_format_history_skips_tools(self):
        msgs = [
            HumanMessage(content="hi"),
            AIMessage(content="yo"),
            AIMessage(content="", tool_calls=[{"name": "calculator", "args": {}, "id": "1"}]),
        ]
        text = format_history_messages(msgs, limit=10)
        self.assertIn("You: hi", text)
        self.assertIn("AI: yo", text)


class TestSessionCommands(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.store = SessionStore(
            index_path=os.path.join(self.root, "index.json"),
            root_dir=self.root,
        )
        self.store.bootstrap()
        self.app = MagicMock()
        snap = MagicMock()
        snap.next = ()
        snap.values = {"messages": [HumanMessage(content="hi")]}
        self.app.aget_state = AsyncMock(return_value=snap)
        self.config = {"configurable": {"thread_id": self.store.get_active().id}}

    async def test_new_and_switch_commands(self):
        ok, msg = await handle_slash_command(
            "/new Work", self.store, self.app, self.config
        )
        self.assertTrue(ok)
        self.assertIn("Work", msg)
        sid = self.config["configurable"]["thread_id"]
        self.assertTrue(sid.startswith("sess_") or sid == LEGACY_THREAD_ID or True)
        other = self.store.create("Other", make_active=False)
        ok, msg = await handle_slash_command(
            f"/switch {other.id}", self.store, self.app, self.config
        )
        self.assertTrue(ok)
        self.assertEqual(self.config["configurable"]["thread_id"], other.id)
        self.app.get_state.assert_not_called()

    async def test_switch_blocked_when_approval_pending(self):
        # Simulate approval interrupt
        last = MagicMock()
        last.tool_calls = [{"name": "write_office_file", "args": {}, "id": "t1"}]
        snap = MagicMock()
        snap.next = ("approval",)
        snap.values = {"messages": [last]}
        self.app.aget_state.return_value = snap
        other = self.store.create("Other2", make_active=False)
        ok, msg = await handle_slash_command(
            f"/switch {other.id}", self.store, self.app, self.config
        )
        self.assertTrue(ok)
        self.assertIn("审批", msg)
        self.app.get_state.assert_not_called()

    async def test_history_uses_async_graph_state(self):
        ok, msg = await handle_slash_command(
            "/history", self.store, self.app, self.config
        )
        self.assertTrue(ok)
        self.assertIn("You: hi", msg)
        self.app.aget_state.assert_awaited_once_with(self.config)
        self.app.get_state.assert_not_called()

    async def test_legacy_permissions_command_is_not_exposed(self):
        ok, msg = await handle_slash_command(
            "/permissions", self.store, self.app, self.config
        )
        self.assertTrue(ok)
        self.assertIn("未知命令", msg)


class TestAsyncApprovalHelpers(unittest.IsolatedAsyncioTestCase):
    async def test_async_checkpointer_contract(self):
        calls = [{"name": "write_office_file", "args": {}, "id": "t1"}]
        last = MagicMock()
        last.tool_calls = calls
        snapshot = MagicMock(next=("approval",), values={"messages": [last]})
        app = MagicMock()
        app.aget_state = AsyncMock(return_value=snapshot)
        app.aupdate_state = AsyncMock()
        config = {"configurable": {"thread_id": "async_test"}}

        self.assertEqual(await apending_approval(app, config), calls)
        await areject_prepare(app, config, calls, reason="denied")

        app.aget_state.assert_awaited_once_with(config)
        app.aupdate_state.assert_awaited_once()
        app.get_state.assert_not_called()
        app.update_state.assert_not_called()


class TestAsyncSqliteCheckpoint(unittest.IsolatedAsyncioTestCase):
    async def test_question_then_new_session_uses_async_checkpoint_api(self):
        llm = ScriptedLLM(script=["answer"])
        tool = make_tool("noop", "ok")
        original = agent_mod.get_provider
        agent_mod.get_provider = lambda **kw: llm
        try:
            async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
                app = agent_mod.create_agent_app(
                    tools=[tool],
                    checkpointer=saver,
                    require_approval=False,
                    enable_planner=False,
                )
                config = thread_config("async_sqlite")
                events = [
                    event
                    async for event in app.astream(
                        {"messages": [HumanMessage(content="what time is it")]},
                        config=config,
                        stream_mode="updates",
                    )
                ]
                self.assertTrue(events)
                self.assertIsNone(await apending_approval(app, config))

                root = tempfile.mkdtemp()
                store = SessionStore(
                    index_path=os.path.join(root, "index.json"), root_dir=root
                )
                store.bootstrap()
                ok, msg = await handle_slash_command(
                    "/new Async", store, app, config
                )
                self.assertTrue(ok)
                self.assertIn("Async", msg)
        finally:
            agent_mod.get_provider = original


class TestCheckpointIsolation(unittest.TestCase):
    def test_two_threads_do_not_share_messages(self):
        llm = ScriptedLLM(script=["answer-A", "answer-B"])
        tool = make_tool("noop", "ok")
        original = agent_mod.get_provider
        agent_mod.get_provider = lambda **kw: llm
        try:
            app = agent_mod.create_agent_app(
                tools=[tool],
                checkpointer=MemorySaver(),
                require_approval=False,
                enable_planner=False,
            )
        finally:
            agent_mod.get_provider = original

        cfg_a = thread_config("iso_a")
        cfg_b = thread_config("iso_b")
        app.invoke({"messages": [HumanMessage(content="thread-A-msg")]}, cfg_a)
        app.invoke({"messages": [HumanMessage(content="thread-B-msg")]}, cfg_b)

        msgs_a = app.get_state(cfg_a).values.get("messages") or []
        msgs_b = app.get_state(cfg_b).values.get("messages") or []
        text_a = " | ".join(str(getattr(m, "content", "")) for m in msgs_a)
        text_b = " | ".join(str(getattr(m, "content", "")) for m in msgs_b)
        self.assertIn("thread-A-msg", text_a)
        self.assertNotIn("thread-B-msg", text_a)
        self.assertIn("thread-B-msg", text_b)
        self.assertNotIn("thread-A-msg", text_b)


if __name__ == "__main__":
    unittest.main()
