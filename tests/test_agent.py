import unittest
import os
import sys
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fluxagent.core.context import AgentState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


class TestAgent(unittest.TestCase):

    def test_planner_prompt_preserves_parallel_investigation_as_one_step(self):
        from fluxagent.core.agent import _build_planner_prompt

        prompt = _build_planner_prompt("分别调查审批和恢复两条独立工作线，最后汇总")

        self.assertIn("合并成同一个步骤", prompt)
        self.assertIn("不要把它们拆成前后串行步骤", prompt)
        self.assertIn("综合判断放在并行调查步骤之后", prompt)

    @patch('fluxagent.core.agent.get_provider')
    def test_simple_request_skips_planner_in_real_graph(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM
        llm = ScriptedLLM(script=["直接回答"])
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[],
            require_approval=False,
            enable_planner=True,
        )

        result = app.invoke({"messages": [HumanMessage(content="hello")]})

        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["llm_calls"], 1)
        self.assertEqual(llm.cursor, 1)
        self.assertEqual(result["task_plan"], [])

    @patch('fluxagent.core.agent.get_provider')
    def test_unverified_action_claim_forces_structured_tool_call(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from fluxagent.core.budget import RunBudget
        from tests.eval.stub import ScriptedLLM, counting_tool

        calls = {"write_office_file": 0}
        tool = counting_tool("write_office_file", calls)
        llm = ScriptedLLM(
            script=[
                "我已调用工具 write_office_file，文件已创建成功",
                [{"name": "write_office_file", "args": {"value": "ok"}}],
                "文件已通过工具写入并验证",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[tool],
            require_approval=False,
            enable_planner=False,
            budget=RunBudget(max_llm_calls=6),
        )

        result = app.invoke({"messages": [HumanMessage(content="创建一个文件")]})

        self.assertEqual(calls["write_office_file"], 1)
        self.assertEqual(result["tool_rounds"], 1)
        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["last_failure_class"], "")
        self.assertGreaterEqual(llm.cursor, 3)

    @patch('fluxagent.core.agent.get_provider')
    def test_repeated_plain_text_after_unverified_claim_cannot_finish(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from fluxagent.core.budget import RunBudget
        from tests.eval.stub import ScriptedLLM

        llm = ScriptedLLM(
            script=[
                "我已调用工具，文件已创建成功",
                "文件已经完成",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[],
            require_approval=False,
            enable_planner=False,
            budget=RunBudget(max_llm_calls=2),
        )

        result = app.invoke({"messages": [HumanMessage(content="创建一个文件")]})

        self.assertEqual(result["task_status"], "failed")
        self.assertEqual(result["last_failure_class"], "model_protocol")

    @patch('fluxagent.core.agent.get_provider')
    def test_planned_text_steps_do_not_end_with_an_incomplete_plan(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM

        llm = ScriptedLLM(
            script=[
                '["检查当前配置", "给出最终结论"]',
                "配置检查完成",
                "最终结论",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[],
            require_approval=False,
            enable_planner=True,
        )

        result = app.invoke(
            {"messages": [HumanMessage(content="请先检查当前配置，然后给出最终结论")]}
        )

        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["llm_calls"], 3)
        self.assertTrue(all(step["status"] == "done" for step in result["task_plan"]))
        self.assertEqual(result["messages"][-1].content, "最终结论")

    @patch('fluxagent.core.agent.get_provider')
    def test_planned_tool_step_gets_final_text_response(self, mock_get_provider):
        """A successful final tool step must not terminate before synthesis."""
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM, counting_tool

        tool = counting_tool("demo_skill", {}, result="说明书内容")
        llm = ScriptedLLM(
            script=[
                '["读取技能说明"]',
                [{"name": "demo_skill", "args": {"value": "ok"}}],
                "根据说明书给出的最终回答",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[tool],
            require_approval=False,
            enable_planner=True,
        )

        result = app.invoke(
            {"messages": [HumanMessage(content="请先读取技能说明，然后根据说明给出最终回答")]}
        )

        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["messages"][-1].content, "根据说明书给出的最终回答")

    @patch('fluxagent.core.agent.get_provider')
    def test_install_step_does_not_terminate_following_plan_steps(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM, counting_tool

        calls = {}
        installer = counting_tool(
            "install_mcp_server",
            calls,
            result="MCP Server 已安装、验证并登记: demo",
        )
        llm = ScriptedLLM(
            script=[
                '["安装 MCP Server", "执行安装后的验证"]',
                [{"name": installer.name, "args": {"value": "demo"}}],
                "安装后的验证完成",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[installer],
            require_approval=False,
            enable_planner=True,
        )

        result = app.invoke({"messages": [HumanMessage(content="请先安装 MCP Server，然后执行安装后的验证")]})

        self.assertEqual(calls[installer.name], 1)
        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["messages"][-1].content, "安装后的验证完成")
        self.assertEqual(llm.cursor, 3)

    @patch('fluxagent.core.agent.get_provider')
    def test_successful_unplanned_skill_gets_final_synthesis(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM, counting_tool

        skill = counting_tool("persona_skill", {}, result="技能说明内容")
        llm = ScriptedLLM(
            script=[
                [{"name": skill.name, "args": {"value": "祖师爷"}}],
                "这是整合后的最终回答",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[skill],
            require_approval=False,
            enable_planner=False,
        )

        result = app.invoke({"messages": [HumanMessage(content="请使用 persona skill 回答")]})

        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["messages"][-1].content, "这是整合后的最终回答")
        self.assertEqual(llm.cursor, 2)

    def test_explicit_backend_name_scopes_model_tools(self):
        from fluxagent.core.agent import _explicit_tool_scope
        from tests.eval.stub import make_tool

        tools = [
            make_tool("mcp_stock-sdk-mcp_get_a_share_quotes"),
            make_tool("mcp_HowToCook-mcp_mcp_howtocook_whatToEat"),
            make_tool("tong-jincheng-perspective"),
        ]

        scoped, label = _explicit_tool_scope("请用 HowToCook MCP 写菜单", tools)
        self.assertIn("howtocook", label.casefold())
        self.assertEqual([tool.name for tool in scoped], [tools[1].name])

        scoped, label = _explicit_tool_scope("直接用 tong-jincheng skill 回答", tools)
        self.assertIn("tong", label.casefold())
        self.assertEqual([tool.name for tool in scoped], [tools[2].name])

        scoped, label = _explicit_tool_scope("请用 HowToCook MCP 写菜单", [tools[0]])
        self.assertIn("howtocook", label.casefold())
        self.assertEqual(scoped, [])

    def test_metadata_alias_routes_without_service_specific_table(self):
        from fluxagent.core.agent import _explicit_tool_scope
        from langchain_core.tools import StructuredTool

        def _run(value: str = "") -> str:
            return value

        recipe = StructuredTool.from_function(
            func=_run,
            name="planner_backend",
            description="recipe planner",
            metadata={
                "capability": "recipe_backend",
                "aliases": ["recipe planner", "菜单编排"],
            },
        )
        unrelated = StructuredTool.from_function(
            func=_run,
            name="search_backend",
            description="search",
            metadata={"capability": "search_backend", "aliases": ["search"]},
        )

        scoped, label = _explicit_tool_scope("请使用菜单编排服务生成晚餐", [recipe, unrelated])

        self.assertEqual(label, "recipe_backend")
        self.assertEqual([tool.name for tool in scoped], [recipe.name])

    def test_declared_chinese_alias_supports_shortened_user_phrase(self):
        from fluxagent.core.agent import _explicit_tool_scope
        from langchain_core.tools import StructuredTool

        def _run(value: str = "") -> str:
            return value

        persona = StructuredTool.from_function(
            func=_run,
            name="persona_skill",
            description="response persona",
            metadata={
                "capability": "persona_skill",
                "aliases": ["深情祖师爷"],
            },
        )
        recipes = StructuredTool.from_function(
            func=_run,
            name="recipe_backend",
            description="recipes",
            metadata={"capability": "recipe_backend", "aliases": ["菜单"]},
        )

        scoped, label = _explicit_tool_scope("祖师爷教你追女朋友技巧", [persona, recipes])

        self.assertEqual(label, "persona_skill")
        self.assertEqual([tool.name for tool in scoped], [persona.name])

    def test_implicit_request_retrieves_relevant_external_capability(self):
        from fluxagent.core.agent import _tool_scope_for_turn
        from langchain_core.tools import StructuredTool

        def _run(value: str = "") -> str:
            return value

        recipes = StructuredTool.from_function(
            func=_run,
            name="mcp_recipe_recommendMeals",
            description="根据人数、忌口和过敏原推荐中餐菜谱和膳食计划",
            metadata={"capability": "recipe_backend", "kind": "mcp"},
        )
        stocks = StructuredTool.from_function(
            func=_run,
            name="mcp_market_quotes",
            description="查询股票行情和证券报价",
            metadata={"capability": "market_backend", "kind": "mcp"},
        )

        scoped, label = _tool_scope_for_turn("帮我推荐中餐菜谱", [recipes, stocks])

        self.assertEqual(label, "")
        self.assertEqual([tool.name for tool in scoped], [recipes.name])

    @patch('fluxagent.core.agent.get_provider')
    def test_explicit_backend_blocks_wrong_tool_call_before_execution(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM, counting_tool

        calls = {}
        stock = counting_tool("mcp_stock-sdk-mcp_get_a_share_quotes", calls)
        cook = counting_tool("mcp_HowToCook-mcp_mcp_howtocook_whatToEat", calls)
        llm = ScriptedLLM(
            script=[
                [{"name": stock.name, "args": {"value": "AAPL"}}],
                "已按 HowToCook 要求整理菜单",
            ]
        )
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[stock, cook],
            require_approval=False,
            enable_planner=False,
        )

        result = app.invoke({"messages": [HumanMessage(content="请用 HowToCook MCP 写菜单")]})

        self.assertEqual(calls, {})
        self.assertEqual(result["task_status"], "done")
        self.assertEqual(result["messages"][-1].content, "已按 HowToCook 要求整理菜单")

    @patch('fluxagent.core.agent.get_provider')
    def test_explicit_backend_switch_drops_previous_backend_turn(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM, counting_tool

        cook = counting_tool("mcp_HowToCook-mcp_mcp_howtocook_whatToEat", {})
        tong = counting_tool("tong-jincheng-perspective", {})
        llm = ScriptedLLM(script=["旧菜单内容", "新的童锦程回答"])
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[cook, tong],
            require_approval=False,
            enable_planner=False,
        )
        config = {"configurable": {"thread_id": "backend-switch"}}

        app.invoke({"messages": [HumanMessage(content="请用 HowToCook MCP 写菜单")]}, config)
        result = app.invoke(
            {"messages": [HumanMessage(content="换个话题，直接用 tong-jincheng skill 回复我")]},
            config,
        )

        self.assertEqual(result["messages"][-1].content, "新的童锦程回答")
        second_call = llm.calls[-1]
        self.assertFalse(any("旧菜单内容" in str(message.content) for message in second_call))

    @patch('fluxagent.core.agent.get_provider')
    def test_context_trim_metrics_reach_run_summary(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM

        llm = ScriptedLLM(script=["压缩摘要", "最终回答"])
        mock_get_provider.return_value = llm
        app = create_agent_app(
            tools=[],
            require_approval=False,
            enable_planner=False,
        )
        messages = []
        for index in range(19):
            messages.extend([
                HumanMessage(content=f"用户消息{index}"),
                AIMessage(content=f"AI消息{index}"),
            ])
        messages.append(HumanMessage(content="第20轮问题"))

        result = app.invoke({"messages": messages})
        summary = result["last_run_summary"]

        self.assertEqual(result["task_status"], "done")
        self.assertEqual(summary["context_trim_count"], 1)
        self.assertEqual(summary["context_turns_before_trim"], 20)
        self.assertEqual(summary["context_turns_kept"], 10)
        self.assertEqual(summary["context_turns_discarded"], 10)
        self.assertEqual(summary["summary_llm_calls"], 1)
        self.assertEqual(summary["llm_calls"], 2)

    def test_agent_state_initialization(self):
        """测试 AgentState 的初始化"""
        from fluxagent.core.context import AgentState

        initial_state = AgentState(
            messages=[],
            summary=""
        )

        self.assertEqual(initial_state["messages"], [])
        self.assertEqual(initial_state["summary"], "")

    @patch('fluxagent.core.provider.get_provider')
    @patch('fluxagent.core.skill_loader.load_dynamic_skills')
    @patch('fluxagent.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_basic(self, mock_load_skills, mock_get_provider):
        """测试创建基础代理应用（带 Mock）"""
        from fluxagent.core.agent import create_agent_app

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        try:
            app = create_agent_app(provider_name="openai", model_name="gpt-4o-mini")
            self.assertIsNotNone(app)
        except Exception as e:
            # 即使出现其他错误也记录
            print(f"Unexpected error: {e}")
            raise

    @patch('fluxagent.core.provider.get_provider')
    @patch('fluxagent.core.skill_loader.load_dynamic_skills')
    @patch('fluxagent.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_with_custom_tools(self, mock_load_skills, mock_get_provider):
        """测试创建带有自定义工具的代理应用（带 Mock）"""
        from fluxagent.core.agent import create_agent_app
        from langchain_core.tools import tool

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        # 创建一个真正的 mock 工具（使用@tool 装饰器）
        @tool
        def mock_tool(test_param: str) -> str:
            """A mock tool for testing"""
            return f"mock result: {test_param}"

        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                tools=[mock_tool]
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise

    @patch('fluxagent.core.provider.get_provider')
    @patch('fluxagent.core.skill_loader.load_dynamic_skills')
    @patch('fluxagent.core.tools.builtins.BUILTIN_TOOLS', [])
    def test_create_agent_app_with_checkpointer(self, mock_load_skills, mock_get_provider):
        """测试创建带有检查点的代理应用（带 Mock）"""
        from fluxagent.core.agent import create_agent_app
        from langgraph.checkpoint.memory import MemorySaver

        # Mock provider 返回值
        mock_provider = Mock()
        mock_provider.bind_tools.return_value = Mock()
        mock_get_provider.return_value = mock_provider

        # Mock 动态技能加载
        mock_load_skills.return_value = []

        memory_saver = MemorySaver()
        try:
            app = create_agent_app(
                provider_name="openai",
                model_name="gpt-4o-mini",
                checkpointer=memory_saver
            )
            self.assertIsNotNone(app)
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise


    @patch('fluxagent.core.agent.get_provider')
    def test_entry_route_uses_small_model_when_rules_abstain(self, mock_get_provider):
        """规则说不准的请求（两个动作、没有先后词）才愿意花一次小模型调用。"""
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM

        classifier = ScriptedLLM(script=['{"needs_plan": true}'])
        main = ScriptedLLM(
            script=['["读取文件内容", "写入摘要"]', "读取完成", "摘要写好了"]
        )

        def provider_for(provider_name=None, model_name=None, **kwargs):
            return classifier if model_name == "classifier-model" else main

        mock_get_provider.side_effect = provider_for
        with patch.dict(
            os.environ,
            {
                "DEFAULT_MODEL": "classifier-model",
                "DEFAULT_PROVIDER": "stub",
                "FLUXAGENT_PLAN_CLASSIFIER": "1",
            },
        ):
            app = create_agent_app(tools=[], require_approval=False, enable_planner=True)
            result = app.invoke(
                {"messages": [HumanMessage(content="读取文件内容，写入摘要")]}
            )

        self.assertEqual(result["entry_route_source"], "model")
        self.assertTrue(result["entry_needs_plan"])
        self.assertEqual(result["planning_request"], "读取文件内容，写入摘要")
        self.assertEqual(len(result["task_plan"]), 2)
        self.assertEqual(result["task_status"], "done")
        self.assertEqual(classifier.cursor, 1)

    @patch('fluxagent.core.agent.get_provider')
    def test_entry_route_falls_back_conservatively_when_classifier_is_broken(
        self, mock_get_provider
    ):
        """分类器不吐合法布尔就保守回退成"不规划"，不能把请求卡死在入口。"""
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM

        classifier = ScriptedLLM(script=["我觉得需要规划吧"])
        main = ScriptedLLM(script=["直接回答"])

        def provider_for(provider_name=None, model_name=None, **kwargs):
            return classifier if model_name == "classifier-model" else main

        mock_get_provider.side_effect = provider_for
        with patch.dict(
            os.environ,
            {
                "DEFAULT_MODEL": "classifier-model",
                "DEFAULT_PROVIDER": "stub",
                "FLUXAGENT_PLAN_CLASSIFIER": "1",
            },
        ):
            app = create_agent_app(tools=[], require_approval=False, enable_planner=True)
            result = app.invoke(
                {"messages": [HumanMessage(content="读取文件内容，写入摘要")]}
            )

        self.assertEqual(result["entry_route_source"], "fallback")
        self.assertFalse(result["entry_needs_plan"])
        self.assertEqual(result["task_plan"], [])
        self.assertEqual(result["task_status"], "done")

    @patch('fluxagent.core.agent.get_provider')
    def test_rule_fast_path_never_spends_a_classifier_call(self, mock_get_provider):
        from fluxagent.core.agent import create_agent_app
        from tests.eval.stub import ScriptedLLM

        classifier = ScriptedLLM(script=['{"needs_plan": true}'])
        main = ScriptedLLM(
            script=['["检查当前配置", "给出最终结论"]', "配置检查完成", "最终结论"]
        )

        def provider_for(provider_name=None, model_name=None, **kwargs):
            return classifier if model_name == "classifier-model" else main

        mock_get_provider.side_effect = provider_for
        with patch.dict(
            os.environ,
            {
                "DEFAULT_MODEL": "classifier-model",
                "DEFAULT_PROVIDER": "stub",
                "FLUXAGENT_PLAN_CLASSIFIER": "1",
            },
        ):
            app = create_agent_app(tools=[], require_approval=False, enable_planner=True)
            result = app.invoke(
                {"messages": [HumanMessage(content="请先检查当前配置，然后给出最终结论")]}
            )

        self.assertEqual(result["entry_route_source"], "rule")
        self.assertEqual(classifier.cursor, 0)
        self.assertEqual(result["task_status"], "done")

    @patch('fluxagent.core.agent.get_provider')
    def test_identical_failing_tool_call_is_not_retried_forever(self, mock_get_provider):
        """同一个工具 + 完全相同的参数失败两次后，重试不再产生新信息。"""
        import fluxagent.core.agent as agent_mod
        from fluxagent.core.agent import create_agent_app
        from fluxagent.core.budget import RunBudget
        from tests.eval.stub import ScriptedLLM, make_tool

        def reply(messages):
            prompt = " ".join(str(getattr(m, "content", "")) for m in messages)
            if "replan module" in prompt:
                return '["换一种方式备份数据", "确认备份结果"]'
            if "智能规划器" in prompt:
                return '["备份数据"]'
            return [{"name": "备份数据", "args": {"value": "same"}}]

        mock_get_provider.return_value = ScriptedLLM(script=[reply] * 12)
        with patch.object(agent_mod.audit_logger, "log_event") as log_event:
            app = create_agent_app(
                tools=[make_tool("备份数据", fail=True)],
                require_approval=False,
                enable_planner=True,
                budget=RunBudget(max_retries_per_step=5, max_replans=2),
            )
            result = app.invoke(
                {
                    "messages": [
                        HumanMessage(content="先备份数据然后清理日志最后生成报告")
                    ]
                }
            )

        actions = [call.kwargs.get("action") for call in log_event.call_args_list]
        self.assertIn("retry_suppressed_duplicate", actions)
        self.assertEqual(result["task_status"], "failed")
        self.assertEqual(result["last_failure_class"], "exhausted")
        self.assertGreaterEqual(
            max(result["attempted_tool_calls"].values()), 3
        )


if __name__ == '__main__':
    unittest.main()
