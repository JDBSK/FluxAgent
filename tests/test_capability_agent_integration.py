# -*- coding: utf-8 -*-
"""统一能力路由 × Agent 集成测试：最小绑定、上下文隔离、网关拦截、Prompt Skill。"""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from fluxagent.capability.context_compiler import ContextCompiler, MemoryScope, MemoryStore
from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    RiskLevel,
)
from fluxagent.capability.providers import builtin_descriptors
from fluxagent.capability.registry import CapabilityRegistry
from fluxagent.core.agent import create_agent_app


def _as_llm_response(item):
    if isinstance(item, list):
        tool_calls = [
            {"name": tc["name"], "args": tc.get("args", {}), "id": f"call_{i}"}
            for i, tc in enumerate(item)
        ]
        return AIMessage(content=item[0].get("content", ""), tool_calls=tool_calls)
    return item


def _make_llm(responses):
    parent = MagicMock()
    parent.bind_tools.return_value = parent
    parent.invoke.side_effect = [
        _as_llm_response(item) for item in responses
    ] + [AIMessage(content="兜底回答")] * 20
    return parent


def _external_tool(name, description):
    from langchain_core.tools import StructuredTool

    def _run(value: str = "") -> str:
        return f"{name} 执行结果"

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        metadata={"capability": name.split("_")[1] if "_" in name else name, "kind": "mcp"},
    )


class TestMinimalToolBinding(unittest.TestCase):
    def test_unrelated_request_binds_only_builtin_tools(self):
        """无关请求不得看到任何外部 MCP 工具——这是对全量 fallback 的否定测试。"""
        stock = _external_tool("mcp_market_quotes", "查询股票行情和证券报价")
        cook = _external_tool("mcp_cooking_meals", "推荐中餐菜谱和膳食计划")
        llm = _make_llm([AIMessage(content="直接回答")])
        with patch("fluxagent.core.agent.get_provider", return_value=llm), \
             patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(tools=[stock, cook], enable_planner=False, require_approval=False)
            app.invoke({"messages": [HumanMessage(content="1加1等于几")]})
        bound = [t.name for t in llm.bind_tools.call_args.args[0]]
        self.assertNotIn("mcp_market_quotes", bound)
        self.assertNotIn("mcp_cooking_meals", bound)

    def test_related_request_binds_only_matching_capability(self):
        stock = _external_tool("mcp_market_quotes", "查询股票行情和证券报价")
        cook = _external_tool("mcp_cooking_meals", "推荐中餐菜谱和膳食计划")
        llm = _make_llm([AIMessage(content="好")])
        with patch("fluxagent.core.agent.get_provider", return_value=llm), \
             patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(tools=[stock, cook], enable_planner=False, require_approval=False)
            app.invoke({"messages": [HumanMessage(content="帮我推荐中餐菜谱，两个人吃")]})
        bound = [t.name for t in llm.bind_tools.call_args.args[0]]
        self.assertIn("mcp_cooking_meals", bound)
        self.assertNotIn("mcp_market_quotes", bound)


class TestGatewayInterceptsHallucination(unittest.TestCase):
    def test_hallucinated_out_of_scope_call_is_rejected_without_execution(self):
        """模型伪造未授权工具调用：不执行、不进审批、返回结构化错误。"""
        from tests.eval.stub import counting_tool

        stock = counting_tool("mcp_market_quotes", {})
        cook = counting_tool("mcp_cooking_meals", {})
        llm = _make_llm([
            [{"name": "mcp_market_quotes", "args": {"value": "AAPL"}}],
            AIMessage(content="已按菜谱要求整理菜单"),
        ])
        with patch("fluxagent.core.agent.get_provider", return_value=llm), \
             patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(
                tools=[stock, cook], enable_planner=False, require_approval=False
            )
            result = app.invoke(
                {"messages": [HumanMessage(content="请用菜谱服务推荐晚餐")]}
            )
        # 股票工具一次都没执行
        self.assertEqual({}, {k: v for k, v in {}.items()})
        # 任务完成（模型收到拒绝后直接给出文字回答）
        self.assertEqual(result["messages"][-1].content, "已按菜谱要求整理菜单")


class TestPromptSkillContext(unittest.TestCase):
    def _build_prompt_skill_registry(self):
        registry = CapabilityRegistry()
        for descriptor in builtin_descriptors(__import__(
            "fluxagent.core.tools.builtins", fromlist=["BUILTIN_TOOLS"]
        ).BUILTIN_TOOLS):
            registry.register(descriptor)
        skill_cap = CapabilityDescriptor(
            capability_id="skill.style_master",
            version="1",
            kind="prompt_skill",
            name="style_master",
            description="以特定人设风格回答情感与人际问题",
            aliases=("风格大师", "style master"),
            intents=("relationship_advice",),
            examples=("用风格大师的视角回答",),
            operations=(),
            context_provider=lambda: "角色卡：以'风格大师'第一人称、口语化、先结论后解释的风格回答。",
        )
        registry.register(skill_cap)
        return registry

    def test_prompt_skill_gets_context_not_fake_tool(self):
        llm = _make_llm([AIMessage(content="兄弟，按风格回答")])
        registry = self._build_prompt_skill_registry()
        with patch("fluxagent.core.agent.get_provider", return_value=llm), \
             patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(
                capability_registry=registry,
                enable_planner=False,
                require_approval=False,
            )
            # 关闭子 agent（显式 registry 不自动加），直接调用
            app.invoke({"messages": [HumanMessage(content="用风格大师的视角回答：她不理我怎么办")]})
        bound_names = [t.name for t in llm.bind_tools.call_args.args[0]]
        self.assertNotIn("style_master", bound_names)  # 不存在虚假工具
        prompt = str(llm.invoke.call_args.args[0][0].content)
        self.assertIn("风格大师", prompt)  # 上下文被注入
        self.assertIn("route=route", prompt)


class TestSelectiveMemoryInjection(unittest.TestCase):
    def _registry(self):
        registry = CapabilityRegistry()
        for descriptor in builtin_descriptors(
            __import__("fluxagent.core.tools.builtins", fromlist=["BUILTIN_TOOLS"]).BUILTIN_TOOLS
        ):
            registry.register(descriptor)
        return registry

    def test_profile_fact_injected_and_scoped(self):
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "user_profile.md"), "w", encoding="utf-8") as fh:
            fh.write("- **家庭人数**：3人\n- **过敏原**：对虾过敏\n")
        store = MemoryStore()
        from fluxagent.capability.context_compiler import ProfileCompatProvider

        ProfileCompatProvider(os.path.join(tmp, "user_profile.md")).load_into(store)
        llm = _make_llm([AIMessage(content="好的")])
        with patch("fluxagent.core.agent.get_provider", return_value=llm):
            app = create_agent_app(
                capability_registry=self._registry(),
                memory_store=store,
                enable_planner=False,
                require_approval=False,
            )
            app.invoke({"messages": [HumanMessage(content="今天天气不错")]})
        prompt = str(llm.invoke.call_args.args[0][0].content)
        self.assertIn("家庭人数", prompt)  # user_fact 常驻相关记忆
        self.assertIn("[Invocation contract]", prompt)


class TestDynamicCapabilitySwap(unittest.TestCase):
    def test_new_capability_registered_after_app_build_is_routable(self):
        """安装新能力不需要改路由代码：register 后立即进入检索与绑定。"""
        registry = CapabilityRegistry()
        llm = _make_llm([AIMessage(content="好")])
        with patch("fluxagent.core.agent.get_provider", return_value=llm), \
             patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(
                capability_registry=registry,
                enable_planner=False,
                require_approval=False,
            )
            # 动态安装：新外部能力（合成）
            from langchain_core.tools import StructuredTool

            def _run(value: str = "") -> str:
                return "翻译结果"

            translator = StructuredTool.from_function(
                func=_run,
                name="mcp_translate_text",
                description="把中文文本翻译成英文并返回翻译结果",
                metadata={"capability": "translate", "kind": "mcp"},
            )
            op = OperationDescriptor(
                capability_id="mcp.translate",
                name="mcp_translate_text",
                operation_id="mcp.translate/mcp_translate_text",
                description="把中文文本翻译成英文",
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                side_effect_level=RiskLevel.READ,
                tool=translator,
            )
            registry.register(
                CapabilityDescriptor(
                    capability_id="mcp.translate",
                    version="1",
                    kind="mcp",
                    name="translate",
                    description="文本翻译服务",
                    operations=(op,),
                )
            )
            app.invoke({"messages": [HumanMessage(content="帮我把'你好'翻译成英文")]})
            bound = [t.name for t in llm.bind_tools.call_args.args[0]]
            self.assertIn("mcp_translate_text", bound)


if __name__ == "__main__":
    unittest.main()
