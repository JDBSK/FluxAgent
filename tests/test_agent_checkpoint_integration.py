from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from fluxagent.core.agent import create_agent_app
from fluxagent.core.approval import aapprove, apending_approval, areject
from fluxagent.core.permission_broker import PermissionBroker
from fluxagent.core.tools import sandbox_tools
from tests.support.stub import ScriptedLLM


class TestAgentCheckpointIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_main_agent_approval_survives_restart_and_executes_once(self):
        config = {"configurable": {"thread_id": "main-recovery"}}
        call = {
            "name": "write_office_file",
            "id": "write-once",
            "args": {"filepath": "recovered.txt", "content": "persisted", "mode": "w"},
        }

        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "state.sqlite3")
            office = Path(directory) / "office"
            office.mkdir()

            with patch("fluxagent.core.agent.MEMORY_DIR", directory), \
                    patch("fluxagent.core.agent.get_provider",
                          return_value=ScriptedLLM(script=[[call]])), \
                    patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                async with AsyncSqliteSaver.from_conn_string(database) as saver:
                    with PermissionBroker(database) as broker:
                        app = create_agent_app(
                            tools=[sandbox_tools.write_office_file],
                            checkpointer=saver,
                            permission_broker=broker,
                            enable_planner=False,
                            require_approval=True,
                        )
                        await app.ainvoke(
                            {"messages": [HumanMessage(content="写入 recovered.txt")]},
                            config,
                        )
                        pending = await apending_approval(app, config)
                        self.assertEqual(len(pending or []), 1)
                        self.assertEqual(pending[0]["id"], "write-once")
                        self.assertFalse((office / "recovered.txt").exists())
                        waiting = await app.aget_state(config)
                        self.assertEqual(waiting.next, ("approval",))

            with patch("fluxagent.core.agent.MEMORY_DIR", directory), \
                    patch("fluxagent.core.agent.get_provider",
                          return_value=ScriptedLLM(script=["好的"])), \
                    patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                async with AsyncSqliteSaver.from_conn_string(database) as saver:
                    with PermissionBroker(database) as broker:
                        app = create_agent_app(
                            tools=[sandbox_tools.write_office_file],
                            checkpointer=saver,
                            permission_broker=broker,
                            enable_planner=False,
                            require_approval=True,
                        )
                        restored = await app.aget_state(config)
                        self.assertEqual(restored.next, ("approval",))
                        pending = await apending_approval(app, config)
                        self.assertEqual(pending[0]["action_id"], broker.pending("main-recovery")[0].action_id)
                        result = await aapprove(app, config, pending)
                        self.assertEqual(result["task_status"], "done")
                        self.assertEqual(
                            (office / "recovered.txt").read_text(encoding="utf-8"),
                            "persisted",
                        )
                        record = broker.get(pending[0]["action_id"])
                        self.assertEqual(record.status.value, "succeeded")
                        self.assertEqual(record.tool_call_id, "write-once")

                        # A second resume has no pending graph work and cannot execute the action again.
                        again = await app.ainvoke(None, config)
                        self.assertEqual(again["task_status"], "done")
                        self.assertEqual(
                            (office / "recovered.txt").read_text(encoding="utf-8"),
                            "persisted",
                        )

    async def test_rejected_main_action_does_not_write_or_complete(self):
        config = {"configurable": {"thread_id": "main-reject"}}
        call = {
            "name": "write_office_file",
            "id": "write-reject",
            "args": {"filepath": "rejected.txt", "content": "must-not-write", "mode": "w"},
        }
        with tempfile.TemporaryDirectory() as directory:
            database = str(Path(directory) / "state.sqlite3")
            office = Path(directory) / "office"
            office.mkdir()
            with patch("fluxagent.core.agent.MEMORY_DIR", directory), \
                    patch("fluxagent.core.agent.get_provider",
                          return_value=ScriptedLLM(script=[[call]])), \
                    patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                async with AsyncSqliteSaver.from_conn_string(database) as saver:
                    with PermissionBroker(database) as broker:
                        app = create_agent_app(
                            tools=[sandbox_tools.write_office_file],
                            checkpointer=saver,
                            permission_broker=broker,
                            enable_planner=False,
                            require_approval=True,
                        )
                        await app.ainvoke({"messages": [HumanMessage(content="拒绝写入")]}, config)
                        pending = await apending_approval(app, config)
                        result = await areject(app, config, pending)
                        self.assertNotEqual(result["task_status"], "done")
                        self.assertFalse((office / "rejected.txt").exists())
