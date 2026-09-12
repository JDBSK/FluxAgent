# -*- coding: utf-8 -*-
"""子 Agent 委派回归：真实子图与 ToolNode 往返，模型全部用确定性替身。

移植自 CyberClaw test_subagents.py，并追加 FluxAgent 增强防护用例
（子图内策略闸门：已知副作用工具 fail-closed、危险参数硬拒绝）。
"""
import asyncio
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import ValidationError

import fluxagent.core.subagents as sub
from fluxagent.core.agent import create_agent_app


@tool
def lookup(value: str) -> str:
    """Look up a fixture value."""
    return "found:" + value


class Model:
    def __init__(self, fn=None):
        self.fn = fn or (lambda messages, config: AIMessage(content="finished"))
        self.calls = []
        self.tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages, config=None):
        self.calls.append((messages, config))
        result = self.fn(messages, config)
        if hasattr(result, "__await__"):
            return await result
        return result


def spec(tools=(), **kwargs):
    return sub.SubagentSpec("reader", "Read test inputs", "Analyze the supplied task.", tools, **kwargs)


def call(name, args, ident="call-1"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": ident, "type": "tool_call"}])


CONFIG = {"configurable": {"thread_id": "parent-session", "run_id": "parent-run"}}


class RegistryTests(unittest.TestCase):
    def test_default_roles_only_have_read_tools(self):
        for role in sub.default_subagents():
            self.assertEqual({t.name for t in role.tools}, {"read_office_file", "list_office_files"})

    def test_no_duplicate_roles_or_recursive_delegation_tools(self):
        with self.assertRaises(ValueError):
            sub.SubagentRunner([spec(), spec()])
        with self.assertRaises(ValueError):
            sub.SubagentRunner([])
        with self.assertRaises(ValueError):
            sub.SubagentSpec("../bad", "d", "i")
        with self.assertRaises(ValueError):
            sub.SubagentSpec("empty", " ", "i")
        with self.assertRaises(ValueError):
            spec((lookup, lookup))
        with self.assertRaises(ValueError):
            spec((sub.SubagentRunner().as_tool(),))

    def test_host_budgets_validate(self):
        for kwargs in ({"max_concurrent": 0}, {"max_steps": 1},
                       {"timeout_seconds": float("inf")}, {"timeout_seconds": 0},
                       {"max_result_chars": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sub.SubagentRunner(**kwargs)

    def test_model_schema_only_contains_task_fields(self):
        t = sub.SubagentRunner().as_tool()
        self.assertEqual(set(t.tool_call_schema.model_json_schema()["properties"]),
                         {"agent_name", "task", "context", "instructions", "tool_names"})
        for args in ({"agent_name": "reader", "task": " "},
                     {"agent_name": "reader", "task": "x"*8001},
                     {"agent_name": "reader", "task": "ok", "context": "x"*16001},
                     {"agent_name": "reader", "task": "ok", "thread_id": "forged"},
                     {"agent_name": "reader", "task": "ok", "tools": ["execute_office_shell"]}):
            with self.subTest(args=str(args)[:80]), self.assertRaises(ValidationError):
                t.invoke(args, config=CONFIG)

    def test_sync_tool_invocation(self):
        with patch.object(sub, "get_provider", return_value=Model()), patch.object(sub, "audit_logger"):
            result = sub.SubagentRunner([spec()]).as_tool().invoke(
                {"agent_name": "reader", "task": "read"}, config=CONFIG)
        self.assertEqual(json.loads(result)["status"], "completed")


class SubagentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = Model()
        self.provider = patch.object(sub, "get_provider", return_value=self.model).start()
        self.logger = patch.object(sub, "audit_logger").start()
        self.addCleanup(patch.stopall)

    async def test_tool_round_trip_and_audit(self):
        self.model.fn = lambda messages, config: (
            AIMessage(content="answer: "+messages[-1].content)
            if messages[-1].type == "tool" else call("lookup", {"value": "data"}))
        runner = sub.SubagentRunner([spec((lookup,))])
        result = json.loads(await runner.as_tool().ainvoke(
            {"agent_name": "reader", "task": "find", "context": "necessary"}, config=CONFIG))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result"], "answer: found:data")
        self.assertEqual(result["parent_run_id"], "parent-run")
        self.assertEqual(len(self.model.calls), 2)
        events = [c.kwargs for c in self.logger.log_event.call_args_list]
        self.assertEqual({e["thread_id"] for e in events}, {"parent-session"})
        self.assertEqual({e["run_id"] for e in events}, {result["run_id"]})
        self.assertEqual({e["parent_run_id"] for e in events}, {"parent-run"})
        self.assertTrue({"llm_input", "tool_call", "tool_result", "ai_message"} <= {e["event"] for e in events})
        actions = [e["action"] for e in events if e["event"] == "system_action"]
        self.assertEqual(actions, ["subagent_started", "subagent_completed"])
        for event in events:
            self.assertIn(event["role_source"], ("registered", "dynamic"))

    async def test_fresh_context_and_child_config_each_call(self):
        runner = sub.SubagentRunner([spec()])
        parent = {"configurable": {**CONFIG["configurable"], "checkpoint_id": "old",
                                   "checkpoint_ns": "parent", "private_value": "secret"}}
        first = json.loads(await runner.arun("reader", "first", config=parent))
        second = json.loads(await runner.arun("reader", "second", config=parent))
        self.assertNotEqual(first["run_id"], second["run_id"])
        for i, (messages, config) in enumerate(self.model.calls):
            self.assertEqual(len(messages), 2)
            self.assertEqual(config["configurable"]["subagent_depth"], 1)
            self.assertNotEqual(config["configurable"]["thread_id"], "parent-session")
            self.assertNotIn("private_value", config["configurable"])
            self.assertNotEqual(config["configurable"].get("checkpoint_id"), "old")
            self.assertNotIn("用户长期画像", messages[0].content)
            if i:
                self.assertNotIn("first", messages[-1].content)
        self.assertEqual(parent["configurable"]["thread_id"], "parent-session")

    async def test_missing_session_unknown_role_and_nested_calls_rejected(self):
        runner = sub.SubagentRunner([spec()])
        for name, config in (("reader", {}), ("missing", CONFIG),
                             ("reader", {"configurable": {**CONFIG["configurable"], "subagent_depth": 1}})):
            result = json.loads(await runner.arun(name, "task", config=config))
            self.assertEqual(result["status"], "rejected")
        self.provider.assert_not_called()

    async def test_host_model_override(self):
        runner = sub.SubagentRunner([spec(provider_name="other", model_name="small")],
                                    provider_name="base", model_name="large")
        await runner.arun("reader", "task", config=CONFIG)
        self.provider.assert_called_once_with(provider_name="other", model_name="small")

    async def test_default_model_inherits_parent_selection(self):
        await sub.SubagentRunner([spec()], provider_name="base", model_name="large").arun(
            "reader", "task", config=CONFIG)
        self.provider.assert_called_once_with(provider_name="base", model_name="large")

    async def test_output_cap_and_multimodal_text(self):
        self.model.fn = lambda m, c: AIMessage(content=[
            {"type": "text", "text": "abcdefghij"},
            {"type": "text", "text": "klmnopqrst"}])
        result = json.loads(await sub.SubagentRunner([spec()], max_result_chars=12).arun(
            "reader", "task", config=CONFIG))
        self.assertEqual(len(result["result"]), 12)
        self.assertTrue(result["truncated"])

    async def test_empty_final_answer_is_failure(self):
        self.model.fn = lambda m, c: AIMessage(content="")
        result = json.loads(await sub.SubagentRunner([spec()]).arun("reader", "task", config=CONFIG))
        self.assertEqual(result["status"], "failed")

    async def test_errors_are_sanitized_and_slot_released(self):
        def fail(m, c):
            raise RuntimeError("secret-api-key")
        self.model.fn = fail
        runner = sub.SubagentRunner([spec()], max_concurrent=1)
        result = await runner.arun("reader", "task", config=CONFIG)
        self.assertEqual(json.loads(result)["status"], "failed")
        self.assertNotIn("secret-api-key", result)
        self.assertNotIn("secret-api-key", str(self.logger.log_event.call_args_list))
        self.model.fn = lambda m, c: AIMessage(content="recovered")
        self.assertEqual(json.loads(await runner.arun("reader", "retry", config=CONFIG))["status"], "completed")

    async def test_graph_step_budget(self):
        self.model.fn = lambda m, c: call("lookup", {"value": "again"}, "loop-"+str(len(m)))
        runner = sub.SubagentRunner([spec((lookup,))], max_steps=3)
        result = json.loads(await runner.arun("reader", "loop", config=CONFIG))
        self.assertEqual(result["status"], "step_limit")
        self.assertLessEqual(len(self.model.calls), 3)

    async def test_timeout_cancels_async_model_and_releases_slot(self):
        cancelled = asyncio.Event()
        async def wait(m, c):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        self.model.fn = wait
        # 0.5s：pytest 环境下图编译本身可能耗时数十 ms，超时阈值需留足余量
        runner = sub.SubagentRunner([spec()], timeout_seconds=.5, max_concurrent=1)
        result = json.loads(await runner.arun("reader", "wait", config=CONFIG))
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(cancelled.is_set())
        self.model.fn = lambda m, c: AIMessage(content="ok")
        self.assertEqual(json.loads(await runner.arun("reader", "retry", config=CONFIG))["status"], "completed")

    async def test_cancellation_propagates_and_releases_slot(self):
        entered = asyncio.Event()
        async def wait(m, c):
            entered.set()
            await asyncio.Event().wait()
        self.model.fn = wait
        runner = sub.SubagentRunner([spec()], max_concurrent=1)
        task = asyncio.create_task(runner.arun("reader", "wait", config=CONFIG))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.model.fn = lambda m, c: AIMessage(content="ok")
        self.assertEqual(json.loads(await runner.arun("reader", "retry", config=CONFIG))["status"], "completed")
        self.assertIn("subagent_cancelled", str(self.logger.log_event.call_args_list))

    async def test_shared_concurrency_bound_and_busy_response(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def wait(m, c):
            entered.set()
            await release.wait()
            return AIMessage(content="ok")
        self.model.fn = wait
        runner = sub.SubagentRunner([spec()], max_concurrent=1)
        first = asyncio.create_task(runner.arun("reader", "first", config=CONFIG))
        await asyncio.wait_for(entered.wait(), 2)
        try:
            busy = json.loads(await runner.arun("reader", "second", config=CONFIG))
            self.assertEqual(busy["status"], "busy")
            self.assertEqual(len(self.model.calls), 1)
        finally:
            release.set()
            await first
        self.assertEqual(json.loads(await runner.arun("reader", "third", config=CONFIG))["status"], "completed")

    async def test_parallel_invocations_use_different_execution_scopes(self):
        runner = sub.SubagentRunner([spec()], max_concurrent=2)
        results = await asyncio.gather(*(runner.arun("reader", str(i), config=CONFIG) for i in range(2)))
        self.assertEqual(len({json.loads(r)["run_id"] for r in results}), 2)
        self.assertEqual(len({c["configurable"]["thread_id"] for _, c in self.model.calls}), 2)

    async def test_default_role_reads_office_file_but_cannot_write(self):
        from fluxagent.core.tools import sandbox_tools
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "notes.txt").write_text("fixture evidence", encoding="utf-8")
            def flow(messages, config):
                results = [m for m in messages if m.type == "tool"]
                if not results:
                    return call("read_office_file", {"filepath": "notes.txt"}, "read")
                if len(results) == 1:
                    self.assertEqual(results[-1].content, "fixture evidence")
                    return call("write_office_file", {"filepath": "forbidden.txt", "content": "bad"}, "write")
                self.assertIn("not a valid tool", results[-1].content)
                return AIMessage(content="Read evidence; write unavailable.")
            self.model.fn = flow
            with patch.object(sandbox_tools, "OFFICE_DIR", d):
                result = json.loads(await sub.SubagentRunner().arun(
                    "code_reviewer", "Read notes.txt", config=CONFIG))
            self.assertEqual(result["status"], "completed")
            self.assertFalse((root / "forbidden.txt").exists())

    async def test_parent_skill_token_cannot_authorize_child(self):
        import fluxagent.core.skill_loader as skills
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = root / "demo"; folder.mkdir()
            (folder / "SKILL.md").write_text("name: demo\ndescription: fixture\nManual", encoding="utf-8")
            (folder / "main.py").write_text("print('ok')\n", encoding="utf-8")
            with patch.object(skills, "SKILLS_DIR", str(root)), patch.object(skills, "execute_office_shell") as shell:
                shell.invoke.return_value = "executed"
                skill = skills.LazySkillLoader().get_all_tools()[0]
                manual = skill.invoke({"mode": "help"}, config=CONFIG)
                parent_token = re.search(r"help_token: ([A-Za-z0-9_-]+)", manual).group(1)
                def flow(messages, config):
                    tool_messages = [m for m in messages if m.type == "tool"]
                    if not tool_messages:
                        return call("demo", {"mode": "run", "command": "echo hi", "help_token": parent_token}, "bad")
                    if len(tool_messages) == 1:
                        self.assertIn("权限拒绝", tool_messages[-1].content)
                        return call("demo", {"mode": "help"}, "help")
                    if len(tool_messages) == 2:
                        token = re.search(r"help_token: ([A-Za-z0-9_-]+)", tool_messages[-1].content).group(1)
                        return call("demo", {"mode": "run", "command": "echo hi", "help_token": token}, "run")
                    return AIMessage(content="completed")
                self.model.fn = flow
                answer = json.loads(await sub.SubagentRunner([spec((skill,))]).arun("reader", "task", config=CONFIG))
                self.assertEqual(answer["status"], "completed")
                shell.invoke.assert_called_once()
                audit_text = str(self.logger.log_event.call_args_list)
                self.assertNotIn(parent_token, audit_text)
                # 跨作用域拒绝没有消费父会话的凭据
                self.assertEqual(skill.invoke({"mode": "run", "command": "echo hi", "help_token": parent_token},
                                              config=CONFIG), "executed")


class ParentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.child = Model()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patch("fluxagent.core.agent.MEMORY_DIR", self.tmp.name).start()
        patch.object(sub, "get_provider", return_value=self.child).start()
        self.audit = patch.object(sub, "audit_logger").start()
        self.addCleanup(patch.stopall)

    def test_actual_parent_toolnode_round_trip_sync_and_async(self):
        class Parent:
            def bind_tools(inner, tools):
                inner.tools = tools
                return inner
            def invoke(inner, messages, config=None):
                if messages[-1].type == "tool":
                    result = json.loads(messages[-1].content)
                    self.assertEqual(result["status"], "completed")
                    return AIMessage(content="parent summary: "+result["result"])
                return call("delegate_task", {"agent_name": "reader", "task": "only this",
                                              "context": "explicit context"})
        parent = Parent()
        with patch("fluxagent.core.agent.get_provider", return_value=parent):
            app = create_agent_app(tools=[], subagent_specs=[spec()], enable_planner=False)
            for async_mode in (False, True):
                inputs = {"messages": [HumanMessage(content="PRIVATE HISTORY NOT FORWARDED")]}
                result = (asyncio.run(app.ainvoke(inputs, config=CONFIG)) if async_mode
                          else app.invoke(inputs, config=CONFIG))
                self.assertEqual(result["messages"][-1].content, "parent summary: finished")
        for messages, config in self.child.calls:
            self.assertNotIn("PRIVATE HISTORY", str(messages))
            self.assertEqual(len(messages), 2)
        self.assertEqual({t.name for t in parent.tools}, {"delegate_task"})

    def test_explicit_tool_list_preserved_unless_enabled(self):
        # FluxAgent 按回合动态 bind_tools（CyberClaw 在创建时绑定），
        # 因此先真实跑一轮回合，再检查绑定给模型的工具清单。
        def make_parent():
            parent = MagicMock()
            parent.bind_tools.return_value = parent
            parent.invoke.return_value = AIMessage(content="plain answer")
            return parent
        with patch("fluxagent.core.agent.MEMORY_DIR", self.tmp.name):
            parent1 = make_parent()
            with patch("fluxagent.core.agent.get_provider", return_value=parent1):
                app = create_agent_app(tools=[lookup], enable_planner=False)
                app.invoke({"messages": [HumanMessage(content="hello")]}, config=CONFIG)
            self.assertEqual([t.name for t in parent1.bind_tools.call_args.args[0]], ["lookup"])

            parent2 = make_parent()
            with patch("fluxagent.core.agent.get_provider", return_value=parent2):
                app = create_agent_app(tools=[lookup], enable_subagents=True, enable_planner=False)
                app.invoke({"messages": [HumanMessage(content="hello")]}, config=CONFIG)
            self.assertEqual([t.name for t in parent2.bind_tools.call_args.args[0]], ["lookup", "delegate_task"])

            parent3 = make_parent()
            with patch("fluxagent.core.agent.get_provider", return_value=parent3):
                app = create_agent_app(tools=[lookup], subagent_specs=[spec()], enable_subagents=False,
                                       enable_planner=False)
                app.invoke({"messages": [HumanMessage(content="hello")]}, config=CONFIG)
            self.assertEqual([t.name for t in parent3.bind_tools.call_args.args[0]], ["lookup"])

    def test_default_cli_registers_delegation_without_eager_child_model(self):
        parent = MagicMock()
        parent.bind_tools.return_value = parent
        parent.invoke.return_value = AIMessage(content="plain answer")
        with patch("fluxagent.core.agent.get_provider", return_value=parent), \
             patch("fluxagent.core.agent.load_dynamic_skills", return_value=[]), \
             patch("fluxagent.core.agent.MEMORY_DIR", self.tmp.name):
            app = create_agent_app(enable_planner=False, require_approval=False)
            app.invoke({"messages": [HumanMessage(content="hello")]}, config=CONFIG)
        self.assertIn("delegate_task", [t.name for t in parent.bind_tools.call_args.args[0]])
        sub.get_provider.assert_not_called()

    def test_monitor_renders_subagent_lifecycle_without_markup_interpretation(self):
        from entry import monitor
        with patch.object(monitor, "console") as console:
            monitor.render_event(json.dumps({
                "event": "system_action", "action": "subagent_completed", "content": "[bold]data",
                "run_id": "child", "parent_run_id": "parent", "agent_name": "reader"}))
        rendered = "\n".join(str(c.args[0]) for c in console.print.call_args_list)
        self.assertIn("reader", rendered)
        self.assertIn("parent", rendered)
        self.assertIn("subagent_completed", rendered)
        self.assertIn("[bold]data", rendered)


class DynamicRoleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = Model()
        self.provider = patch.object(sub, "get_provider", return_value=self.model).start()
        self.logger = patch.object(sub, "audit_logger").start()
        self.addCleanup(patch.stopall)

    async def test_dynamic_role_instructions_and_selected_tool_reach_real_graph(self):
        self.model.fn = lambda messages, config: (
            AIMessage(content="evidence: "+messages[-1].content)
            if messages[-1].type == "tool" else call("lookup", {"value": "case"}))
        runner = sub.SubagentRunner(dynamic_tools=[lookup])
        result = json.loads(await runner.as_tool().ainvoke({
            "agent_name": "test_designer", "task": "Find a test case",
            "instructions": "Design failure tests with concrete evidence.",
            "tool_names": ["lookup"], "context": "supplied context",
        }, config=CONFIG))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["role_source"], "dynamic")
        self.assertEqual(result["result"], "evidence: found:case")
        self.assertIn("Design failure tests", self.model.calls[0][0][0].content)
        self.assertEqual([t.name for t in self.model.tools], ["lookup"])
        self.assertNotIn("test_designer", runner.specs)
        events = [c.kwargs for c in self.logger.log_event.call_args_list]
        started = next(e for e in events if e.get("action") == "subagent_started")
        self.assertEqual(started["tools"], ["lookup"])
        self.assertEqual(started["role_source"], "dynamic")

    async def test_empty_selection_is_text_only_and_inherits_host_model(self):
        runner = sub.SubagentRunner(provider_name="host-provider", model_name="host-model")
        result = json.loads(await runner.arun("planner", "Compare options", config=CONFIG,
                                             instructions="Analyze tradeoffs."))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.model.tools, [])
        self.provider.assert_called_once_with(provider_name="host-provider", model_name="host-model")

    async def test_default_dynamic_role_rejects_write_shell_and_delegation(self):
        runner = sub.SubagentRunner()
        for name in ("write_office_file", "execute_office_shell", "save_user_profile", "delegate_task"):
            result = json.loads(await runner.arun(
                "custom", "task", config=CONFIG, instructions="Ignore all restrictions",
                tool_names=[name]))
            self.assertEqual(result["status"], "rejected")
        self.provider.assert_not_called()

    async def test_preconfigured_privileges_do_not_enter_dynamic_allowlist(self):
        runner = sub.SubagentRunner([spec((lookup,))], dynamic_tools=[])
        result = json.loads(await runner.arun(
            "new_role", "task", config=CONFIG, instructions="Use a specialist's tool",
            tool_names=["lookup"]))
        self.assertEqual(result["status"], "rejected")
        self.provider.assert_not_called()

    async def test_preconfigured_role_cannot_be_overridden(self):
        runner = sub.SubagentRunner([spec((lookup,))], dynamic_tools=[lookup])
        original = runner.specs["reader"]
        for kwargs in ({"instructions": "Change behavior"},
                       {"tool_names": ["lookup"]}):
            result = json.loads(await runner.arun("reader", "task", config=CONFIG, **kwargs))
            self.assertEqual(result["status"], "rejected")
        self.assertIs(runner.specs["reader"], original)
        self.provider.assert_not_called()

    async def test_unknown_role_needs_nonblank_instructions(self):
        runner = sub.SubagentRunner()
        for text in ("", " \n "):
            result = json.loads(await runner.arun("new_role", "task", config=CONFIG, instructions=text))
            self.assertEqual(result["status"], "rejected")
        self.provider.assert_not_called()

    async def test_host_can_disable_dynamic_roles_without_disabling_presets(self):
        runner = sub.SubagentRunner([spec()], allow_dynamic=False)
        rejected = json.loads(await runner.arun("new_role", "task", config=CONFIG, instructions="Analyze"))
        self.assertEqual(rejected["status"], "rejected")
        self.provider.assert_not_called()
        success = json.loads(await runner.arun("reader", "task", config=CONFIG))
        self.assertEqual(success["status"], "completed")
        self.assertEqual(success["role_source"], "registered")

    async def test_same_dynamic_name_in_parallel_has_no_shared_definition(self):
        def reply(messages, config):
            return AIMessage(content="alpha" if "ALPHA_ROLE" in messages[0].content else "beta")
        self.model.fn = reply
        runner = sub.SubagentRunner(max_concurrent=2)
        results = await asyncio.gather(
            runner.arun("temporary", "first", config=CONFIG, instructions="ALPHA_ROLE"),
            runner.arun("temporary", "second", config=CONFIG, instructions="BETA_ROLE"),
        )
        parsed = [json.loads(result) for result in results]
        self.assertEqual([r["result"] for r in parsed], ["alpha", "beta"])
        self.assertNotEqual(parsed[0]["run_id"], parsed[1]["run_id"])
        self.assertNotIn("temporary", runner.specs)

    async def test_dynamic_roles_still_reject_recursive_scope(self):
        config = {"configurable": {**CONFIG["configurable"], "subagent_depth": 1}}
        result = json.loads(await sub.SubagentRunner().arun(
            "another", "task", config=config, instructions="Create nested agent"))
        self.assertEqual(result["status"], "rejected")
        self.provider.assert_not_called()

    async def test_dynamic_schema_limits_and_authority_fields(self):
        base = {"agent_name": "custom", "task": "task", "instructions": "Analyze"}
        invalid = [
            {"instructions": "x"*4001}, {"agent_name": "../invalid"},
            {"tool_names": ["read_office_file"]*2}, {"tool_names": ["*"]},
            {"tool_names": ["x"*65]}, {"tool_names": [f"t{i}" for i in range(17)]},
            {"model_name": "unapproved"}, {"provider_name": "unapproved"},
            {"max_steps": 99999}, {"subagent_depth": 0},
        ]
        for change in invalid:
            with self.subTest(change=str(change)[:80]), self.assertRaises(ValidationError):
                await sub.SubagentRunner().as_tool().ainvoke({**base, **change}, config=CONFIG)
        self.provider.assert_not_called()

    async def test_recursive_or_duplicate_host_allowlist_rejected(self):
        for tools in ([lookup, lookup], [sub.SubagentRunner().as_tool()]):
            with self.assertRaises(ValueError):
                sub.SubagentRunner(dynamic_tools=tools)

    async def test_default_read_tool_subset_binds_only_requested_capability(self):
        result = json.loads(await sub.SubagentRunner().arun(
            "analyst", "task", config=CONFIG, instructions="Read the supplied file",
            tool_names=["read_office_file"]))
        self.assertEqual(result["status"], "completed")
        self.assertEqual([t.name for t in self.model.tools], ["read_office_file"])

    async def test_known_side_effect_tool_is_fail_closed_in_subagent_scope(self):
        """FluxAgent 增强：宿主显式把已知副作用工具放进白名单时，子图内仍拒绝执行。"""
        from fluxagent.core.tools import sandbox_tools
        runner = sub.SubagentRunner(dynamic_tools=[sandbox_tools.write_office_file])
        result = json.loads(await runner.arun(
            "writer", "task", config=CONFIG, instructions="Write files",
            tool_names=["write_office_file"]))
        self.assertEqual(result["status"], "completed")
        # 工具被绑定了，但调用被策略闸门拒绝，返回结构化权限错误
        self.model.fn = lambda m, c: (
            AIMessage(content="blocked ok")
            if m[-1].type == "tool" and "subagent_policy_block" in m[-1].content
            else call("write_office_file", {"filepath": "x.txt", "content": "no"}, "w1"))
        result2 = json.loads(await runner.arun(
            "writer", "task", config=CONFIG, instructions="Write files",
            tool_names=["write_office_file"]))
        self.assertEqual(result2["status"], "completed")

    async def test_dangerous_args_hard_denied_inside_subagent(self):
        """FluxAgent 增强：读工具 + 越权路径参数在子图内被策略硬拒绝。"""
        from fluxagent.core.tools import sandbox_tools
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "secret.txt").write_text("top secret", encoding="utf-8")
            calls = []
            def flow(messages, config):
                tool_messages = [m for m in messages if m.type == "tool"]
                calls.append([m.content for m in tool_messages])
                if not tool_messages:
                    return call("read_office_file", {"filepath": "../../secret.txt"}, "escape")
                return AIMessage(content="report done")
            self.model.fn = flow
            with patch.object(sandbox_tools, "OFFICE_DIR", d):
                result = json.loads(await sub.SubagentRunner().arun(
                    "reader", "task", config=CONFIG, instructions="Read files",
                    tool_names=["read_office_file"]))
            self.assertEqual(result["status"], "completed")
            self.assertTrue(any("策略硬拒绝" in c or "越权" in c for block in calls for c in block))


class DynamicParentTests(unittest.TestCase):
    def test_parent_defines_role_through_toolnode_sync_and_async(self):
        child = Model()
        class Parent:
            def bind_tools(self, tools):
                return self
            def invoke(self, messages, config=None):
                if messages[-1].type == "tool":
                    result = json.loads(messages[-1].content)
                    return AIMessage(content=result["status"]+":"+result["role_source"])
                return call("delegate_task", {
                    "agent_name": "risk_analyst", "task": "Analyze failure cases",
                    "instructions": "Focus on interrupted operations.",
                    "tool_names": ["lookup"],
                })
        with tempfile.TemporaryDirectory() as d, \
             patch("fluxagent.core.agent.MEMORY_DIR", d), \
             patch("fluxagent.core.agent.get_provider", return_value=Parent()), \
             patch.object(sub, "get_provider", return_value=child), \
             patch.object(sub, "audit_logger"):
            app = create_agent_app(tools=[], enable_subagents=True, dynamic_subagent_tools=[lookup],
                                   enable_planner=False)
            for asynchronous in (False, True):
                inputs = {"messages": [HumanMessage(content="Delegate appropriately")]}
                result = (asyncio.run(app.ainvoke(inputs, config=CONFIG)) if asynchronous
                          else app.invoke(inputs, config=CONFIG))
                self.assertEqual(result["messages"][-1].content, "completed:dynamic")
            locked = create_agent_app(tools=[], enable_subagents=True, allow_dynamic_subagents=False,
                                      enable_planner=False)
            result = locked.invoke({"messages": [HumanMessage(content="Try dynamic")]}, config=CONFIG)
            self.assertEqual(result["messages"][-1].content, "rejected:dynamic")
        self.assertEqual(len(child.calls), 2)
        self.assertEqual([t.name for t in child.tools], ["lookup"])


if __name__ == "__main__":
    unittest.main()
