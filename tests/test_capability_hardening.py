# -*- coding: utf-8 -*-
"""能力控制面加固测试：LLM 重排、索引持久化、记忆写侧、泄漏修复。"""
from __future__ import annotations

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from fluxagent.capability.context_compiler import CompilationContext, ContextCompiler, MemoryScope, MemoryStore
from fluxagent.capability.retrieval import DenseIndex, DeterministicEmbedding, OpenAICompatibleEmbedding
from fluxagent.capability.registry import CapabilityRegistry
from fluxagent.capability.router import (
    Candidate,
    HeuristicReranker,
    HybridRouter,
    LLMReranker,
)
from tests.test_hybrid_router import make_capability


def _candidate(capability, score, operation=None):
    return Candidate(
        doc_id=f"cap:{capability.capability_id}",
        capability_id=capability.capability_id,
        kind=capability.kind,
        name=capability.name,
        text=capability.searchable_text(),
        score=score,
        capability=capability,
        operation=operation,
    )


class TestLLMReranker(unittest.TestCase):
    def _candidates(self):
        cooking = make_capability("cap.cooking", name="Cook", description="菜谱推荐做菜")
        market = make_capability("cap.market", name="Market", description="股票行情走势查询")
        return [_candidate(cooking, 0.9), _candidate(market, 0.8)]

    def test_structured_selection_reorders(self):
        chat = MagicMock()
        chat.invoke.return_value = AIMessage(content='{"selected": 1, "reason": "股票查询"}')
        reranker = LLMReranker(chat)
        ranked = reranker.rerank("查询茅台股价", self._candidates())
        self.assertEqual(ranked[0][0].capability_id, "cap.market")
        self.assertEqual(reranker.last_fallback_reason, "")

    def test_malformed_output_falls_back_to_heuristic(self):
        chat = MagicMock()
        chat.invoke.return_value = AIMessage(content="我觉得是第一个")
        reranker = LLMReranker(chat)
        ranked = reranker.rerank("推荐菜谱", self._candidates())
        self.assertTrue(ranked)  # 回退启发式仍有结果
        self.assertTrue(reranker.last_fallback_reason)

    def test_explicit_abstain_returns_empty(self):
        chat = MagicMock()
        chat.invoke.return_value = AIMessage(content='{"selected": -1, "reason": "无关"}')
        reranker = LLMReranker(chat)
        self.assertEqual(reranker.rerank("1加1", self._candidates()), [])

    def test_llm_exception_falls_back(self):
        chat = MagicMock()
        chat.invoke.side_effect = RuntimeError("api down")
        reranker = LLMReranker(chat)
        ranked = reranker.rerank("推荐菜谱", self._candidates())
        self.assertTrue(ranked)
        self.assertIn("RuntimeError", reranker.last_fallback_reason)


class TestIndexPersistence(unittest.TestCase):
    def test_save_load_roundtrip_preserves_decisions(self):
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.alpha", name="AlphaSvc", description="菜谱推荐服务"))
        registry.register(make_capability("cap.beta", name="BetaSvc", description="股票行情服务"))
        snapshot = registry.snapshot()

        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
        router.rebuild_index(snapshot)
        decision_a = router.decide("推荐菜谱", snapshot)

        with tempfile.TemporaryDirectory() as tmp:
            router.save_index(tmp)
            router2 = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
            self.assertTrue(router2.load_index(tmp, snapshot))
            decision_b = router2.decide("推荐菜谱", snapshot)
        self.assertEqual(decision_a.capability_ids, decision_b.capability_ids)
        self.assertEqual(decision_a.route, decision_b.route)

    def test_version_mismatch_refuses_stale_index(self):
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.alpha", name="AlphaSvc", description="菜谱推荐服务"))
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
        router.rebuild_index(snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            router.save_index(tmp)
            registry.register(make_capability("cap.new", name="NewSvc", description="翻译文本服务"))
            snapshot2 = registry.snapshot()
            router2 = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
            self.assertFalse(router2.load_index(tmp, snapshot2))

    def test_embedding_tag_mismatch_refuses(self):
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.alpha", name="AlphaSvc", description="菜谱推荐服务"))
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
        router.rebuild_index(snapshot)
        with tempfile.TemporaryDirectory() as tmp:
            router.save_index(tmp)
            router2 = HybridRouter(embedding_provider=DeterministicEmbedding(dim=64))
            self.assertFalse(router2.load_index(tmp, snapshot))

    def test_health_flap_does_not_trigger_index_rebuild(self):
        """健康翻转推进 snapshot.version（契约/网关失效）但 content_version
        不变：decide() 不得重建检索索引；内容变化（新能力注册）仍触发。"""
        from fluxagent.capability.descriptors import ProviderHealth

        registry = CapabilityRegistry()
        registry.register(make_capability("cap.alpha", name="AlphaSvc", description="菜谱推荐服务"))
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
        router.decide("推荐菜谱", snapshot)  # 首次 decide 完成索引构建

        rebuild_calls: list[str] = []
        original_rebuild = router.rebuild_index

        def counting_rebuild(snap):
            rebuild_calls.append(snap.content_version)
            return original_rebuild(snap)

        router.rebuild_index = counting_rebuild

        # 健康翻转（含转为不可用/恢复）：version 推进，但索引不重建
        v_healthy = registry.snapshot().version
        registry.update_health("cap.alpha", ProviderHealth.UNAVAILABLE)
        self.assertNotEqual(v_healthy, registry.snapshot().version)
        decision = router.decide(
            "推荐菜谱", registry.snapshot(), explicit_capability="cap.alpha"
        )
        self.assertEqual(decision.route, "unavailable")
        registry.update_health("cap.alpha", ProviderHealth.HEALTHY)
        router.decide("推荐菜谱", registry.snapshot())
        self.assertEqual(rebuild_calls, [])

        # 内容变化：新能力注册必须触发重建（索引语料随之更新）
        registry.register(
            make_capability("cap.new", name="NewSvc", description="翻译文本服务")
        )
        router.decide("翻译文本", registry.snapshot())
        self.assertEqual(len(rebuild_calls), 1)


class TestHeuristicRerankerOrders(unittest.TestCase):
    def test_retrieval_order_preserves_fused_rank(self):
        alpha = make_capability("cap.alpha", name="A", description="通用数据服务 alpha 内容")
        beta = make_capability("cap.beta", name="B", description="另一服务 beta 内容")
        # fused 分数：alpha 0.9 > beta 0.5，但 blended（覆盖度主导）可能反转
        reranker = HeuristicReranker(order_by="retrieval")
        ranked = reranker.rerank("通用 数据 服务", [_candidate(alpha, 0.9), _candidate(beta, 0.5)])
        self.assertEqual(ranked[0][0].capability_id, "cap.alpha")

    def test_invalid_order_mode_rejected(self):
        with self.assertRaises(ValueError):
            HeuristicReranker(order_by="bogus")


class TestMemoryWriteSide(unittest.TestCase):
    def test_agent_writes_verified_episodic_memory(self):
        from fluxagent.capability.descriptors import OperationDescriptor
        from fluxagent.capability.providers import builtin_descriptors
        from fluxagent.core.agent import create_agent_app
        from fluxagent.core.tools.builtins import BUILTIN_TOOLS

        registry = CapabilityRegistry()
        for descriptor in builtin_descriptors(BUILTIN_TOOLS):
            registry.register(descriptor)
        store = MemoryStore()

        parent = MagicMock()
        parent.bind_tools.return_value = parent
        parent.invoke.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "calculator", "args": {"expression": "1+", "value": ""}, "id": "c1"}
            ]),
            AIMessage(content="计算失败了，如实说明"),
        ] + [AIMessage(content="兜底")] * 10

        with patch("fluxagent.core.agent.get_provider", return_value=parent), \
             patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(
                capability_registry=registry,
                memory_store=store,
                enable_planner=False,
                require_approval=False,
            )
            app.invoke(
                {"messages": [HumanMessage(content="帮我算一下 1+")]},
                {"configurable": {"thread_id": "mem-test"}},
            )

        record_ids = {r.record_id for r in store.all_records()}
        # episodic 生产者：工具失败已写入（calculator 对非法表达式返回失败）
        episodic = [r for r in store.all_records() if r.record_id.startswith("episodic:")]
        self.assertTrue(episodic, "工具失败应写入 episodic 记忆")
        # 会话边界：episodic 记录必须归属本会话
        self.assertTrue(all(r.session_id == "mem-test" for r in episodic))

    def test_tagged_memory_excluded_from_abstain_turns(self):
        """跨任务泄漏回归：abstain 回合不得看到其他能力的带标签记忆。"""
        store = MemoryStore()
        store.add(
            MemoryScope.EPISODIC, "正在分析茅台", key="task.market",
            capability_tags=("mcp.market",),
        )
        store.add(MemoryScope.USER_FACT, "通用事实", key="profile.generic")
        compiler = ContextCompiler(store, token_budget=4000)
        snapshot = compiler.compile(
            CompilationContext(user_text="1加1", route="answer_without_tools")
        )
        self.assertNotIn("茅台", snapshot.system_prompt)
        self.assertNotIn("通用事实", snapshot.system_prompt)

    def test_cross_session_memory_boundary(self):
        """跨会话数据边界：会话 A 的经验不得进入会话 B；全局记录不受限。"""
        store = MemoryStore()
        store.add(
            MemoryScope.EPISODIC, "会话A正在分析茅台", key="task:A",
            capability_tags=("mcp.market",), session_id="session-A",
        )
        store.add(
            MemoryScope.EPISODIC, "会话A工具失败记录", key="ep:A",
            capability_tags=("mcp.market",), session_id="session-A",
        )
        store.add(MemoryScope.USER_FACT, "全局过敏原：对虾", key="profile.allergy")
        compiler = ContextCompiler(store, token_budget=4000)

        # 会话 B 的市场请求：不得看到会话 A 的任何记录，全局事实可见
        snapshot_b = compiler.compile(
            CompilationContext(
                user_text="看盘和对虾过敏原", route="route", intent="quote_lookup",
                capability_ids=("mcp.market",),
                memory_scopes=("user_fact", "episodic"),
                session_id="session-B",
            )
        )
        self.assertNotIn("会话A", snapshot_b.system_prompt)
        self.assertIn("对虾", snapshot_b.system_prompt)

        # 会话 A 自己的请求：可见本会话记录
        snapshot_a = compiler.compile(
            CompilationContext(
                user_text="看盘和茅台K线", route="route", intent="quote_lookup",
                capability_ids=("mcp.market",),
                memory_scopes=("user_fact", "episodic"),
                session_id="session-A",
            )
        )
        self.assertIn("会话A正在分析茅台", snapshot_a.system_prompt)

        # fail-closed：编译器未提供会话上下文时，会话级记录一律排除
        snapshot_noctx = compiler.compile(
            CompilationContext(
                user_text="看盘", route="route", intent="quote_lookup",
                capability_ids=("mcp.market",),
                memory_scopes=("user_fact", "episodic"),
            )
        )
        self.assertNotIn("会话A", snapshot_noctx.system_prompt)


class TestDenseIndexRoundtrip(unittest.TestCase):
    def test_save_load_with_text_payloads(self):
        provider = DeterministicEmbedding(dim=64)
        index = DenseIndex()
        index.rebuild([("d1", "股票行情查询"), ("d2", "菜谱推荐")], provider)
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/dense.json"
            index.save(path)
            loaded = DenseIndex.load(path)
            self.assertEqual(
                index.search("股票 行情", provider, top_k=2),
                loaded.search("股票 行情", provider, top_k=2),
            )


class TestOpenAIEmbeddingBatching(unittest.TestCase):
    @patch("openai.OpenAI")
    def test_requests_are_split_at_service_batch_limit(self, openai_cls):
        client = openai_cls.return_value

        def make_response(batch):
            return type("Response", (), {
                "data": [type("Item", (), {"embedding": [float(i)]}) for i, _ in enumerate(batch)]
            })()

        client.embeddings.create.side_effect = lambda **kwargs: make_response(kwargs["input"])
        provider = OpenAICompatibleEmbedding("https://example.test/v1", "key", "embed")
        vectors = provider.embed_texts([f"text-{i}" for i in range(21)])

        self.assertEqual(client.embeddings.create.call_count, 2)
        calls = client.embeddings.create.call_args_list
        self.assertEqual(len(calls[0].kwargs["input"]), 20)
        self.assertEqual(len(calls[1].kwargs["input"]), 1)
        self.assertEqual(len(vectors), 21)




class TestSessionContextRescue(unittest.TestCase):
    def test_router_rescues_anaphoric_turn_with_context(self):
        """指代轮（"回到刚才的话题"）单独检索 abstain，带会话上下文应路由回原能力。"""
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.cooking", name="CookSvc", description="菜谱推荐做菜家常菜"))
        registry.register(make_capability("cap.market", name="MarketSvc", description="股票行情证券数据"))
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
        router.rebuild_index(snapshot)

        alone = router.decide("回到刚才的话题，帮我推荐明天的菜谱", snapshot)
        with_context = router.decide(
            "回到刚才的话题，帮我推荐明天的菜谱",
            snapshot,
            context_turns=["帮我推荐中餐菜谱，两个人吃", "股票行情怎么样"],
        )
        # 单独文本证据不足时不强制；带上下文必须路由回菜谱
        if alone.route != "route":
            self.assertEqual(with_context.route, "route")
            self.assertEqual(with_context.capability_ids, ("cap.cooking",))
            self.assertTrue(with_context.audit.get("context_rescue"))
        else:
            self.assertEqual(with_context.capability_ids, ("cap.cooking",))

    def test_context_rescue_does_not_perturb_confident_routes(self):
        """已有明确路由的请求不受上下文救援扰动。"""
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.cooking", name="CookSvc", description="菜谱推荐做菜"))
        registry.register(make_capability("cap.market", name="MarketSvc", description="股票行情证券"))
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
        router.rebuild_index(snapshot)
        plain = router.decide("查询贵州茅台的股票行情", snapshot)
        with_context = router.decide(
            "查询贵州茅台的股票行情", snapshot,
            context_turns=["帮我推荐中餐菜谱"],
        )
        self.assertEqual(plain.capability_ids, with_context.capability_ids)
        self.assertNotIn("context_rescue", with_context.audit)

    def test_agent_two_turn_anaphoric_routing(self):
        """agent 两轮会话：第二轮指代轮路由回第一轮能力。"""
        from langchain_core.tools import StructuredTool

        from fluxagent.capability.descriptors import (
            CapabilityDescriptor,
            OperationDescriptor,
            RiskLevel,
        )
        from fluxagent.capability.providers import builtin_descriptors
        from fluxagent.core.agent import create_agent_app

        def _run(value: str = "") -> str:
            return "菜谱执行结果"

        cooking = StructuredTool.from_function(
            func=_run, name="mcp_cooking_meals",
            description="推荐中餐菜谱和膳食计划",
            metadata={"capability": "cooking", "kind": "mcp"},
        )
        store = MemoryStore()
        llm = MagicMock()
        llm.bind_tools.return_value = llm
        llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[
                {"name": "mcp_cooking_meals", "args": {"value": "x"}, "id": "c1"}
            ]),
            AIMessage(content="第一轮菜谱回答"),
            AIMessage(content="回到菜谱话题的第二轮回答"),
        ] + [AIMessage(content="兜底")] * 10

        registry = CapabilityRegistry()
        for descriptor in builtin_descriptors(
            __import__("fluxagent.core.tools.builtins", fromlist=["BUILTIN_TOOLS"]).BUILTIN_TOOLS
        ):
            registry.register(descriptor)
        op = OperationDescriptor(
            capability_id="mcp.cooking",
            name="mcp_cooking_meals",
            operation_id="mcp.cooking/mcp_cooking_meals",
            description="推荐中餐菜谱和膳食计划",
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            side_effect_level=RiskLevel.READ,
            tool=cooking,
        )
        registry.register(
            CapabilityDescriptor(
                capability_id="mcp.cooking",
                version="1",
                kind="mcp",
                name="cooking",
                description="推荐中餐菜谱和膳食计划",
                aliases=("cooking", "菜谱"),
                operations=(op,),
            )
        )
        with patch("fluxagent.core.agent.get_provider", return_value=llm),              patch("fluxagent.core.agent.MEMORY_DIR", tempfile.mkdtemp()):
            app = create_agent_app(
                capability_registry=registry,
                memory_store=store,
                enable_planner=False,
                require_approval=False,
            )
            config = {"configurable": {"thread_id": "anaphora-test"}}
            app.invoke({"messages": [HumanMessage(content="帮我推荐中餐菜谱")]}, config)
            app.invoke({"messages": [HumanMessage(content="回到刚才的话题，再推荐一个菜")]}, config)
            # 第二轮（无词法信号的指代轮）经上下文救援路由回 cooking：
            # LLM 第二次被调用时 bind_tools 的列表里应有 cooking 工具
            second_call_tools = [t.name for t in llm.bind_tools.call_args_list[1].args[0]]
            self.assertIn("mcp_cooking_meals", second_call_tools)


if __name__ == "__main__":
    unittest.main()
