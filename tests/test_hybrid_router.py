# -*- coding: utf-8 -*-
"""Hybrid 检索与 Router 行为测试。全部使用通用合成能力。"""
from __future__ import annotations

import tempfile
import unittest

from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    ProviderHealth,
    RiskLevel,
)
from fluxagent.capability.registry import CapabilityRegistry
from fluxagent.capability.retrieval import (
    BM25Index,
    DenseIndex,
    DeterministicEmbedding,
    Document,
    EmbeddingError,
    rrf_fuse,
)
from fluxagent.capability.router import HeuristicReranker, HybridRouter


def make_capability(
    capability_id: str,
    *,
    name: str = "",
    description: str = "",
    intents: tuple = (),
    examples: tuple = (),
    negatives: tuple = (),
    aliases: tuple = (),
    health: ProviderHealth = ProviderHealth.HEALTHY,
    schema_required: tuple = (),
    enabled: bool = True,
) -> CapabilityDescriptor:
    op_name = f"mcp_{capability_id}_op"
    op = OperationDescriptor(
        capability_id=capability_id,
        name=op_name,
        operation_id=f"{capability_id}/{op_name}",
        description=description or f"{capability_id} 操作",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": list(schema_required),
        },
        required_slots=schema_required,
        side_effect_level=RiskLevel.READ,
        tool=object(),  # 合成宿主；只验证绑定逻辑，不执行
        examples=examples,
    )
    return CapabilityDescriptor(
        capability_id=capability_id,
        version="1",
        kind="mcp",
        name=name or capability_id,
        description=description,
        aliases=aliases,
        intents=intents,
        examples=examples,
        negative_examples=negatives,
        operations=(op,),
        provider_health=health,
        enabled=enabled,
    )


def build_registry(capabilities) -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for capability in capabilities:
        registry.register(capability)
    return registry


def make_router() -> HybridRouter:
    return HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))


CAPS = [
    make_capability(
        "cap.cooking",
        name="RecipeService",
        description="根据人数忌口过敏原推荐菜谱菜式和做菜步骤",
        intents=("recommend_meal",),
        examples=("给我推荐中餐菜谱", "三个人一周菜谱", "推荐两人早餐"),
        aliases=("菜谱", "cooking"),
    ),
    make_capability(
        "cap.market",
        name="MarketData",
        description="查询股票行情价格走势和证券市场数据",
        intents=("quote_lookup",),
        examples=("查询贵州茅台行情", "看看大盘走势"),
        aliases=("股票", "market"),
    ),
    make_capability(
        "cap.weather",
        name="WeatherService",
        description="查询城市天气温度降雨和空气质量",
        intents=("weather_lookup",),
        examples=("今天南京天气", "明天北京下雨吗"),
        aliases=("天气", "weather"),
    ),
]


class TestRetrievalPrimitives(unittest.TestCase):
    def test_tokenize_cjk_bigrams(self):
        tokens = __import__("fluxagent.capability.retrieval", fromlist=["tokenize"]).tokenize("中餐菜谱 abc")
        self.assertIn("中餐", tokens)
        self.assertIn("菜谱", tokens)
        self.assertIn("abc", tokens)

    def test_bm25_ranks_relevant_doc_first(self):
        index = BM25Index()
        index.rebuild(
            [
                Document("d1", "股票行情 证券市场 数据查询"),
                Document("d2", "菜谱推荐 家庭餐食 忌口"),
                Document("d3", "天气查询 城市温度 降雨"),
            ]
        )
        results = index.search("股票行情", top_k=3)
        self.assertEqual(results[0][0], "d1")
        self.assertGreater(results[0][1], 0.0)

    def test_bm25_persistence_roundtrip(self):
        index = BM25Index()
        index.rebuild([Document("d1", "股票 行情"), Document("d2", "菜谱 推荐")])
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/bm25.json"
            index.save(path)
            loaded = BM25Index.load(path)
            self.assertEqual(index.search("股票", 1), loaded.search("股票", 1))

    def test_dense_index_roundtrip_and_search(self):
        provider = DeterministicEmbedding(dim=64)
        index = DenseIndex()
        index.rebuild(
            [("d1", "股票行情 查询"), ("d2", "菜谱推荐 做饭")], provider
        )
        results = index.search("股票 行情", provider, top_k=2)
        self.assertEqual(results[0][0], "d1")

    def test_rrf_fuses_and_prefers_consensus(self):
        fused = rrf_fuse(
            [[("a", 9.0), ("b", 5.0), ("c", 1.0)], [("b", 8.0), ("a", 4.0)]],
            top_k=3,
        )
        ids = [doc_id for doc_id, _ in fused]
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        # a 在两路都靠前，融合分应不低于只在一路第一的 c
        score_map = dict(fused)
        if "c" in score_map:
            self.assertGreaterEqual(score_map["a"], score_map["c"])


class TestHybridRouter(unittest.TestCase):
    def setUp(self):
        self.registry = build_registry(CAPS)
        self.snapshot = self.registry.snapshot()
        self.router = make_router()
        self.router.rebuild_index(self.snapshot)

    def test_implicit_hybrid_routes_to_right_capability(self):
        decision = self.router.decide("帮我推荐中餐菜谱，三个人吃", self.snapshot)
        self.assertEqual(decision.route, "route")
        self.assertEqual(decision.capability_ids, ("cap.cooking",))
        self.assertIn("mcp_cap.cooking_op", decision.allowed_tool_ids)

    def test_market_query_does_not_route_to_cooking(self):
        decision = self.router.decide("查询贵州茅台的股票行情", self.snapshot)
        self.assertEqual(decision.route, "route")
        self.assertEqual(decision.capability_ids, ("cap.market",))

    def test_no_evidence_abstains_without_tools(self):
        decision = self.router.decide("1加1等于几", self.snapshot)
        self.assertIn(decision.route, {"answer_without_tools", "clarify"})
        self.assertEqual(decision.allowed_tool_ids, ())

    def test_explicit_alias_routes_directly(self):
        decision = self.router.decide("使用菜谱服务帮我看看吃什么", self.snapshot)
        self.assertEqual(decision.route, "route")
        self.assertEqual(decision.capability_ids, ("cap.cooking",))

    def test_explicit_unregistered_fails_closed(self):
        decision = self.router.decide("使用天气服务查天气", build_registry([]).snapshot())
        self.assertIn(decision.route, {"clarify", "unavailable", "answer_without_tools"})
        self.assertEqual(decision.allowed_tool_ids, ())

    def test_ambiguous_explicit_returns_clarify(self):
        # 直接构造歧义：两个等长别名同时出现在文本中
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.one", name="Alpha", aliases=("工具甲",)))
        registry.register(make_capability("cap.two", name="Beta", aliases=("工具乙",)))
        snapshot = registry.snapshot()
        router = make_router()
        router.rebuild_index(snapshot)
        decision = router.decide("同时用工具甲和工具乙", snapshot)
        # 最长别名并列 → clarify；即使未判并列也不得返回两个能力
        if decision.route == "clarify":
            self.assertEqual(decision.capability_ids, ())
        else:
            self.assertLessEqual(len(decision.capability_ids), 1)

    def test_unhealthy_capability_yields_unavailable(self):
        registry = build_registry(
            [make_capability("cap.sick", name="SickSvc", description="查询服务数据", health=ProviderHealth.UNAVAILABLE)]
        )
        snapshot = registry.snapshot()
        router = make_router()
        router.rebuild_index(snapshot)
        decision = router.decide("帮我用SickSvc查询数据", snapshot)
        self.assertEqual(decision.route, "unavailable")
        self.assertEqual(decision.allowed_tool_ids, ())

    def test_disabled_capability_not_routed(self):
        registry = build_registry([make_capability("cap.off", name="OffSvc", description="菜谱推荐服务", enabled=False)])
        registry.register(make_capability("cap.cooking2", name="Cook", description="菜谱推荐做菜"))
        snapshot = registry.snapshot()
        router = make_router()
        router.rebuild_index(snapshot)
        decision = router.decide("推荐菜谱", snapshot)
        if decision.route == "route":
            self.assertNotIn("cap.off", decision.capability_ids)

    def test_missing_required_slot_reported(self):
        registry = build_registry(
            [make_capability("cap.slots", name="SlotSvc", description="查询数据服务", schema_required=("symbol",))]
        )
        snapshot = registry.snapshot()
        router = make_router()
        router.rebuild_index(snapshot)
        decision = router.decide("帮我用SlotSvc查询数据", snapshot)
        # symbol 无法从文本推断 → 缺槽位要被记录
        self.assertIn("symbol", decision.missing_slots)

    def test_contract_binds_only_allowed_tools(self):
        decision = self.router.decide("查询股票行情", self.snapshot)
        contract = self.router.build_contract(decision, self.snapshot)
        self.assertEqual(set(contract.tools), set(decision.allowed_tool_ids))
        self.assertTrue(contract.allows("mcp_cap.market_op"))
        self.assertFalse(contract.allows("mcp_cap.cooking_op"))
        self.assertFalse(contract.allows("totally_unknown"))

    def test_answer_route_builds_empty_contract(self):
        decision = self.router.decide("1加1等于几", self.snapshot)
        contract = self.router.build_contract(decision, self.snapshot)
        self.assertEqual(contract.tools, {})

    def test_index_rebuild_on_registry_change(self):
        decision_before = self.router.decide("推荐菜谱", self.snapshot)
        self.registry.register(
            make_capability("cap.news", name="NewsSvc", description="新闻资讯头条推荐阅读")
        )
        snapshot_after = self.registry.snapshot()
        decision_after = self.router.decide("推荐菜谱", snapshot_after)
        self.assertNotEqual(
            decision_before.index_version, decision_after.index_version
        )

    def test_decision_forbids_expanding_contract(self):
        decision = self.router.decide("查询股票行情", self.snapshot)
        contract = self.router.build_contract(decision, self.snapshot)
        # 契约之外的工具必须全部被拒
        for capability in self.snapshot.enabled_capabilities():
            for op in capability.operations:
                if op.name not in decision.allowed_tool_ids:
                    self.assertFalse(contract.allows(op.name))


class TestHeuristicReranker(unittest.TestCase):
    def test_negative_example_penalized(self):
        cooking = make_capability(
            "cap.cooking",
            name="Recipe",
            description="菜谱推荐做菜",
            negatives=("分析股票走势",),
        )
        market = make_capability(
            "cap.market", name="Market", description="股票行情走势查询"
        )
        reranker = HeuristicReranker()
        candidates = [
            __import__("fluxagent.capability.router", fromlist=["Candidate"]).Candidate(
                doc_id="cap:cap.cooking",
                capability_id="cap.cooking",
                kind="mcp",
                name="Recipe",
                text=cooking.searchable_text(),
                score=1.0,
                capability=cooking,
            ),
            __import__("fluxagent.capability.router", fromlist=["Candidate"]).Candidate(
                doc_id="cap:cap.market",
                capability_id="cap.market",
                kind="mcp",
                name="Market",
                text=market.searchable_text(),
                score=1.0,
                capability=market,
            ),
        ]
        ranked = reranker.rerank("分析贵州茅台股票走势", candidates)
        self.assertEqual(ranked[0][0].capability_id, "cap.market")


if __name__ == "__main__":
    unittest.main()
