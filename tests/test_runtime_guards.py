# -*- coding: utf-8 -*-
"""运行时门禁确定性测试：技能凭据、路径边界、日志背压、上下文水位。

无 API 请求、无真实 shell 动作（移植自 CyberClaw test_runtime_guards，
按 FluxAgent 实现适配断言目标）。
"""
import concurrent.futures
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch, mock_open

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt import ToolNode

import fluxagent.core.skill_loader as skills
from fluxagent.core.logger import JSONLEventLogger
from fluxagent.core.tools import sandbox_tools
from fluxagent.core.context import trim_context_messages


class SkillGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(skills, "SKILLS_DIR", str(self.root)).start()
        self.shell = patch.object(skills, "execute_office_shell").start().invoke
        self.shell.return_value = "EXECUTED"
        self.md = self.make_skill("demo", "manual v1")
        self.loader = skills.LazySkillLoader()
        self.tool = self.loader.get_all_tools()[0]
        self.config = {"configurable": {"thread_id": "alice"}}

    def make_skill(self, name, content):
        d = self.root / name
        d.mkdir(exist_ok=True)
        md = d / "SKILL.md"
        md.write_text(f"name: {name}\ndescription: test\n{content}", encoding="utf-8")
        # FluxAgent 合并版 loader：带脚本才会走 run 凭据门禁路径
        (d / "main.py").write_text("print('ok')\n", encoding="utf-8")
        return md

    def help(self, tool=None, config=None):
        result = (tool or self.tool).invoke({"mode": "help"}, config=config or self.config)
        return re.search(r"help_token: ([A-Za-z0-9_-]+)", result).group(1)

    def run_skill(self, token="", tool=None, config=None):
        return (tool or self.tool).invoke(
            {"mode": "run", "command": "echo {{baseDir}}".replace("{{", "{").replace("}}", "}"),
            "help_token": token},
            config=config or self.config)

    def test_run_without_help_never_dispatches(self):
        self.assertIn("权限拒绝", self.run_skill())
        self.assertIn("权限拒绝", self.run_skill("forged"))
        self.shell.assert_not_called()

    def test_help_run_and_replay(self):
        token = self.help()
        self.assertEqual(self.run_skill(token), "EXECUTED")
        self.assertIn("权限拒绝", self.run_skill(token))
        self.shell.assert_called_once()
        self.assertEqual(self.shell.call_args.args[0]["command"], "echo skills/demo")

    def test_cross_session_rejected_without_consuming_owner_grant(self):
        token = self.help()
        self.assertIn("权限拒绝", self.run_skill(token, config={"configurable": {"thread_id": "bob"}}))
        self.assertEqual(self.run_skill(token), "EXECUTED")

    def test_cross_skill_rejected(self):
        self.make_skill("other", "other manual")
        other = [t for t in self.loader.get_all_tools(True) if t.name == "other"][0]
        self.assertIn("权限拒绝", self.run_skill(self.help(), tool=other))
        self.shell.assert_not_called()

    def test_same_manual_symlink_does_not_share_grant_between_skills(self):
        self.md.write_text("description: shared manual")
        alias = self.root / "alias"
        alias.mkdir()
        try:
            (alias / "SKILL.md").symlink_to(self.md)
        except OSError:
            self.skipTest("当前 Windows 配置不允许创建测试符号链接")
        (alias / "main.py").write_text("print('ok')\n", encoding="utf-8")
        tools = {t.name: t for t in self.loader.get_all_tools(True)}
        token = self.help(tool=tools["demo"])
        self.assertIn("权限拒绝", self.run_skill(token, tool=tools["alias"]))
        self.shell.assert_not_called()
        self.assertEqual(self.run_skill(token, tool=tools["demo"]), "EXECUTED")

    def test_missing_session_fails_closed(self):
        result = self.tool.invoke({"mode": "help"})
        self.assertNotIn("help_token:", result)
        self.assertIn("权限拒绝", self.tool.invoke(
            {"mode": "run", "command": "echo hi", "help_token": self.help()}))
        self.shell.assert_not_called()

    def test_expired_token(self):
        with patch.object(skills.time, "monotonic", return_value=100):
            token = self.help()
        with patch.object(skills.time, "monotonic", return_value=401):
            self.assertIn("权限拒绝", self.run_skill(token))
        self.shell.assert_not_called()

    def test_same_mtime_same_size_change_invalidates_token(self):
        token = self.help()
        st = self.md.stat()
        self.md.write_text(self.md.read_text(encoding="utf-8").replace("v1", "v2"), encoding="utf-8")
        os.utime(self.md, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertIn("权限拒绝", self.run_skill(token))
        current = self.tool.invoke({"mode": "help"}, config=self.config)
        self.assertIn("manual v2", current)
        self.shell.assert_not_called()

    def test_deleted_manual_rejected(self):
        token = self.help()
        self.md.unlink()
        self.assertIn("权限拒绝", self.run_skill(token))
        self.shell.assert_not_called()

    def test_clear_cache_revokes_grants(self):
        token = self.help()
        self.loader.clear_cache()
        self.assertIn("权限拒绝", self.run_skill(token))

    def test_new_help_revokes_previous_grant(self):
        first, second = self.help(), self.help()
        self.assertIn("权限拒绝", self.run_skill(first))
        self.assertEqual(self.run_skill(second), "EXECUTED")

    def test_restart_requires_new_help(self):
        token = self.help()
        restarted_tool = skills.LazySkillLoader().get_all_tools()[0]
        self.assertIn("权限拒绝", self.run_skill(token, tool=restarted_tool))

    def test_full_manual_or_rejection_never_silent_truncation(self):
        self.md.write_text("name: demo\n" + "x" * 5000 + "\nIMPORTANT_END", encoding="utf-8")
        self.assertIn("IMPORTANT_END", self.tool.invoke({"mode": "help"}, config=self.config))
        self.md.write_bytes(b"x" * (skills.MAX_SKILL_BYTES + 1))
        self.assertIn("权限拒绝", self.tool.invoke({"mode": "help"}, config=self.config))

    def test_cache_and_grant_bounds(self):
        loader = skills.LazySkillLoader(cache_size=1, max_grants=1)
        other_md = self.make_skill("other", "other")
        for md in (self.md, other_md):
            loader._snapshot(str(md))
        self.assertEqual(len(loader._content_cache), 1)
        a = loader._issue_grant("a", "p", "v")
        loader._issue_grant("b", "p", "v")
        self.assertFalse(loader._consume_grant(a, "a", "p", "v"))

    def test_concurrent_replay_executes_once(self):
        token = self.help()
        barrier = threading.Barrier(2)
        def run(_):
            barrier.wait()
            return self.run_skill(token)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            results = list(pool.map(run, range(2)))
        self.assertEqual(results.count("EXECUTED"), 1)
        self.shell.assert_called_once()

    def test_dispatch_failure_still_consumes_token(self):
        token = self.help()
        self.shell.side_effect = RuntimeError("fake shell failure")
        with self.assertRaises(RuntimeError):
            self.run_skill(token)
        self.assertIn("权限拒绝", self.run_skill(token))
        self.shell.assert_called_once()

    def test_config_is_not_model_controlled_schema(self):
        properties = self.tool.args_schema.model_json_schema()["properties"]
        self.assertNotIn("config", properties)
        self.assertNotIn("thread_id", properties)

    def test_real_toolnode_propagates_config_and_gate(self):
        from langgraph.graph import StateGraph, START, END
        from fluxagent.core.context import AgentState
        graph = StateGraph(AgentState)
        graph.add_node("tools", ToolNode([self.tool]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        node = graph.compile()
        help_state = {"messages": [AIMessage(content="", tool_calls=[
            {"name": "demo", "args": {"mode": "help"}, "id": "help-1", "type": "tool_call"}])]}
        reply = node.invoke(help_state, config=self.config)["messages"][-1]
        token = re.search(r"help_token: ([A-Za-z0-9_-]+)", reply.content).group(1)
        state = {"messages": [AIMessage(content="", tool_calls=[
            {"name": "demo", "args": {"mode": "run", "command": "echo hi", "help_token": token},
             "id": "run-1", "type": "tool_call"}])]}
        result = node.invoke(state, config=self.config)["messages"][-1]
        self.assertEqual(result.content, "EXECUTED")

    def test_full_agent_loop_with_fake_model(self):
        from fluxagent.core.agent import create_agent_app
        class FakeModel:
            def bind_tools(inner, tools):
                return inner
            def invoke(inner, messages, config=None):
                self.assertEqual(config["configurable"]["thread_id"], "alice")
                tool_messages = [m for m in messages if m.type == "tool"]
                if not tool_messages:
                    args = {"mode": "help"}
                elif "help_token:" in tool_messages[-1].content:
                    token = re.search(r"help_token: ([A-Za-z0-9_-]+)", tool_messages[-1].content).group(1)
                    args = {"mode": "run", "command": "echo hi", "help_token": token}
                else:
                    return AIMessage(content="Done")
                return AIMessage(content="", tool_calls=[{"name": "demo", "args": args,
                                                          "id": "call-" + str(len(tool_messages)), "type": "tool_call"}])
        with patch("fluxagent.core.agent.get_provider", return_value=FakeModel()), \
             patch("fluxagent.core.agent.MEMORY_DIR", str(self.root)):
            app = create_agent_app(tools=[self.tool], require_approval=False, enable_planner=False)
            result = app.invoke({"messages": [HumanMessage(content="test")]}, config=self.config)
        self.assertEqual(result["messages"][-1].content, "Done")
        self.shell.assert_called_once()
        import asyncio
        with patch("fluxagent.core.agent.MEMORY_DIR", str(self.root)):
            result = asyncio.run(app.ainvoke({"messages": [HumanMessage(content="async test")]}, config=self.config))
        self.assertEqual(result["messages"][-1].content, "Done")
        self.assertEqual(self.shell.call_count, 2)

    def test_duplicate_skill_names_rejected(self):
        self.make_skill("other", "manual").write_text("name: demo\ndescription: duplicate", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "冲突"):
            self.loader.get_all_tools(True)


class PathTests(unittest.TestCase):
    def test_sibling_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as d:
            office = Path(d) / "office"; office.mkdir()
            outside = Path(d) / "office_other"; outside.mkdir()
            try:
                (office / "link").symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("当前 Windows 配置不允许创建测试符号链接")
            with patch.object(sandbox_tools, "OFFICE_DIR", str(office)):
                for value in ("../office_other/a", "link/a", "/etc/passwd", "C:\\test", "D:relative"):
                    with self.subTest(value=value), self.assertRaises(PermissionError):
                        sandbox_tools._get_safe_path(value)
                self.assertEqual(sandbox_tools._get_safe_path("sub/new.txt"), str(office / "sub/new.txt"))

    def test_command_name_path_prefix_rejected(self):
        for command in ("./python script.py", "/usr/bin/python script.py", "bin/ls"):
            with self.subTest(command=command), self.assertRaises(PermissionError):
                sandbox_tools._parse_command(command)

    def test_read_has_bounded_io(self):
        stream = mock_open(read_data="x" * (sandbox_tools.MAX_READ_BYTES + 1))
        with patch.object(sandbox_tools, "_get_safe_path", return_value="fake"), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True), \
             patch("os.path.getsize", return_value=sandbox_tools.MAX_READ_BYTES), \
             patch("builtins.open", stream):
            result = sandbox_tools.read_office_file.invoke({"filepath": "fake"})
        stream().read.assert_called_once_with(sandbox_tools.MAX_READ_BYTES + 1)
        self.assertEqual(len(result), sandbox_tools.MAX_READ_BYTES)

    def test_read_oversized_file_is_structured_error(self):
        with patch.object(sandbox_tools, "_get_safe_path", return_value="fake"), \
             patch("os.path.exists", return_value=True), \
             patch("os.path.isfile", return_value=True), \
             patch("os.path.getsize", return_value=sandbox_tools.MAX_READ_BYTES + 5):
            result = sandbox_tools.read_office_file.invoke({"filepath": "fake"})
        self.assertIn("file_too_large", result)


class LoggerTests(unittest.TestCase):
    def test_drain_repeated_shutdown_and_post_close(self):
        with tempfile.TemporaryDirectory() as d:
            logger = JSONLEventLogger(d)
            for i in range(30):
                self.assertTrue(logger.log_event("test", "event", n=i))
            logger.shutdown()
            logger.shutdown()
            self.assertFalse(logger.log_event("test", "late"))
            self.assertFalse(logger.worker_thread.is_alive())
            events = [json.loads(x) for x in (Path(d) / "test.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([x["n"] for x in events], list(range(30)))

    def test_bounded_queue_has_explicit_drop_counter(self):
        with tempfile.TemporaryDirectory() as d:
            logger = JSONLEventLogger(d, max_queue_size=1)
            import queue
            with patch.object(logger.log_queue, "put_nowait", side_effect=queue.Full):
                self.assertFalse(logger.log_event("test", "event"))
            self.assertEqual(logger.dropped_events, 1)
            logger.shutdown()


class ContextValidationTests(unittest.TestCase):
    def test_invalid_watermarks(self):
        for trigger, keep in ((0, 1), (2, 0), (2, 3)):
            with self.subTest(trigger=trigger, keep=keep), self.assertRaises(ValueError):
                trim_context_messages([HumanMessage(content="test")], trigger, keep)


if __name__ == "__main__":
    unittest.main()
