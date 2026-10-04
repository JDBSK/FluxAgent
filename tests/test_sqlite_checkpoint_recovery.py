from __future__ import annotations

import os
import tempfile
import unittest
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph


class RecoveryState(TypedDict, total=False):
    task_plan: list[dict]
    step_index: int
    task_status: str
    messages: list
    pending_action_ids: list[str]
    executed_action_ids: list[str]


def build_recovery_graph():
    graph = StateGraph(RecoveryState)

    def before_approval(state: RecoveryState):
        return {"task_status": "waiting_approval"}

    def approval(state: RecoveryState):
        action_id = state["pending_action_ids"][0]
        executed = list(state.get("executed_action_ids") or [])
        if action_id not in executed:
            executed.append(action_id)
        return {
            "pending_action_ids": [],
            "executed_action_ids": executed,
            "task_status": "running",
        }

    graph.add_node("before_approval", before_approval)
    graph.add_node("approval", approval)
    graph.add_edge(START, "before_approval")
    graph.add_edge("before_approval", "approval")
    graph.add_edge("approval", END)
    return graph.compile(interrupt_before=["approval"])


class TestSqliteCheckpointRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_interrupt_survives_close_reopen_and_resumes_same_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = os.path.join(tmp, "state.sqlite3")
            config = {"configurable": {"thread_id": "recover-thread"}}
            seed = {
                "task_plan": [{"step": "write report", "status": "running"}],
                "step_index": 2,
                "task_status": "running",
                "messages": [HumanMessage(content="请生成报告")],
                "pending_action_ids": ["action-pending"],
                "executed_action_ids": ["action-already-done"],
            }

            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                app = build_recovery_graph().with_config({})
                app.checkpointer = saver
                # Recompile with the saver attached; checkpointer is not mutable
                # after compilation in LangGraph versions that freeze graphs.
                graph = StateGraph(RecoveryState)

                def before_approval(state):
                    return {"task_status": "waiting_approval"}

                def approval(state):
                    action_id = state["pending_action_ids"][0]
                    executed = list(state.get("executed_action_ids") or [])
                    if action_id not in executed:
                        executed.append(action_id)
                    return {"pending_action_ids": [], "executed_action_ids": executed, "task_status": "running"}

                graph.add_node("before_approval", before_approval)
                graph.add_node("approval", approval)
                graph.add_edge(START, "before_approval")
                graph.add_edge("before_approval", "approval")
                graph.add_edge("approval", END)
                app = graph.compile(checkpointer=saver, interrupt_before=["approval"])
                await app.ainvoke(seed, config)
                waiting = await app.aget_state(config)
                self.assertEqual(waiting.next, ("approval",))
                self.assertEqual(waiting.values["pending_action_ids"], ["action-pending"])

            async with AsyncSqliteSaver.from_conn_string(database) as saver:
                graph = StateGraph(RecoveryState)

                def before_approval(state):
                    return {"task_status": "waiting_approval"}

                def approval(state):
                    action_id = state["pending_action_ids"][0]
                    executed = list(state.get("executed_action_ids") or [])
                    if action_id not in executed:
                        executed.append(action_id)
                    return {"pending_action_ids": [], "executed_action_ids": executed, "task_status": "running"}

                graph.add_node("before_approval", before_approval)
                graph.add_node("approval", approval)
                graph.add_edge(START, "before_approval")
                graph.add_edge("before_approval", "approval")
                graph.add_edge("approval", END)
                app = graph.compile(checkpointer=saver, interrupt_before=["approval"])
                restored = await app.aget_state(config)
                self.assertEqual(restored.values["task_plan"], seed["task_plan"])
                self.assertEqual(restored.values["step_index"], 2)
                self.assertEqual(restored.values["pending_action_ids"], ["action-pending"])
                self.assertEqual(restored.values["messages"][0].content, "请生成报告")
                await app.ainvoke(None, config)
                done = await app.aget_state(config)
                self.assertEqual(done.values["pending_action_ids"], [])
                self.assertEqual(done.values["executed_action_ids"], ["action-already-done", "action-pending"])
                self.assertEqual(done.values["executed_action_ids"].count("action-already-done"), 1)


if __name__ == "__main__":
    unittest.main()
