# -*- coding: utf-8 -*-
"""Execution Gateway、统一错误协议与熔断测试。"""
from __future__ import annotations

import unittest

from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    ProviderHealth,
    RiskLevel,
)
from fluxagent.capability.gateway import (
    CircuitBreaker,
    ExecutionGateway,
    GatewayResult,
    ToolCallStatus,
)
from fluxagent.capability.registry import CapabilityRegistry
from fluxagent.capability.router import HybridRouter
from fluxagent.capability.retrieval import DeterministicEmbedding


def make_capability(capability_id, *, required=("query",), health=ProviderHealth.HEALTHY, tool_name=None):
    op_name = tool_name or f"mcp_{capability_id}_op"
    op = OperationDescriptor(
        capability_id=capability_id,
        name=op_name,
        operation_id=f"{capability_id}/{op_name}",
        description="查询操作",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": list(required),
        },
        required_slots=required,
        side_effect_level=RiskLevel.READ,
        tool=object(),
    )
    return CapabilityDescriptor(
        capability_id=capability_id,
        version="1",
        kind="mcp",
        name=capability_id,
        description="测试能力",
        intents=("lookup",),
        operations=(op,),
        provider_health=health,
    )


def build_gateway(capability, registry_version_lock: bool = True):
    registry = CapabilityRegistry()
    registry.register(capability)
    snapshot = registry.snapshot()
    router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=64))
    router.rebuild_index(snapshot)
    decision = router.decide("查询数据", snapshot, explicit_capability=capability.capability_id)
    contract = router.build_contract(decision, snapshot)
    return ExecutionGateway(contract, snapshot), registry, snapshot, router


class TestGatewayValidation(unittest.TestCase):
    def setUp(self):
        self.capability = make_capability("cap.alpha")
        self.gateway, self.registry, self.snapshot, self.router = build_gateway(self.capability)

    def test_valid_call_passes(self):
        result = self.gateway.validate("mcp_cap.alpha_op", {"query": "anything"})
        self.assertEqual(result.status, ToolCallStatus.SUCCESS)

    def test_route_mismatch_for_unknown_tool(self):
        result = self.gateway.validate("mcp_cap.other_op", {"query": "x"})
        self.assertEqual(result.status, ToolCallStatus.ROUTE_MISMATCH)
        self.assertIn("禁止", result.detail)
        # 错误 envelope 必须可序列化且标记不可重试
        import json

        payload = json.loads(result.to_tool_message_content())
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "route_mismatch")
        self.assertEqual(payload["error"]["retry_policy"], "stop")

    def test_schema_error_reports_path_and_stops(self):
        result = self.gateway.validate("mcp_cap.alpha_op", {"limit": "not-an-int"})
        self.assertEqual(result.status, ToolCallStatus.SCHEMA_ERROR)
        self.assertIn("query", result.detail)
        self.assertIn("禁止换参数重试", result.detail)

    def test_missing_required_slot_is_schema_error(self):
        result = self.gateway.validate("mcp_cap.alpha_op", {})
        self.assertEqual(result.status, ToolCallStatus.SCHEMA_ERROR)

    def test_registry_change_makes_contract_stale(self):
        self.registry.register(make_capability("cap.beta"))
        new_snapshot = self.registry.snapshot()
        gateway = ExecutionGateway(self.gateway.contract, new_snapshot)
        result = gateway.validate("mcp_cap.alpha_op", {"query": "x"})
        self.assertEqual(result.status, ToolCallStatus.STALE_CONTRACT)

    def test_health_degradation_after_contract_forces_reroute(self):
        # 真实语义：契约签发后能力健康恶化 -> 快照版本变化 -> stale，必须重新路由；
        # Router 在路由时就会拒绝不健康能力（router 测试覆盖 unavailable）。
        registry = CapabilityRegistry()
        registry.register(make_capability("cap.flip"))
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=64))
        router.rebuild_index(snapshot)
        decision = router.decide("查询数据", snapshot, explicit_capability="cap.flip")
        contract = router.build_contract(decision, snapshot)
        gateway = ExecutionGateway(contract, snapshot)
        self.assertEqual(
            gateway.validate("mcp_cap.flip_op", {"query": "x"}).status,
            ToolCallStatus.SUCCESS,
        )
        registry.update_health("cap.flip", ProviderHealth.UNAVAILABLE)
        gateway_new_snapshot = ExecutionGateway(contract, registry.snapshot())
        result = gateway_new_snapshot.validate("mcp_cap.flip_op", {"query": "x"})
        self.assertEqual(result.status, ToolCallStatus.STALE_CONTRACT)

    def test_defensive_unavailable_check_on_frozen_snapshot(self):
        # 分层防御：不健康能力在重排阶段即被过滤（永不成为候选），隐式路由
        # 得到 answer_without_tools；其契约不含任何工具，Gateway 对任何调用
        # 按 ROUTE_MISMATCH 拒绝——两层都 fail-closed，故障能力既不可路由
        # 也不可执行。
        capability = make_capability("cap.sick", health=ProviderHealth.UNAVAILABLE)
        registry = CapabilityRegistry()
        registry.register(capability)
        snapshot = registry.snapshot()
        router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=64))
        router.rebuild_index(snapshot)
        decision = router.decide("查询数据", snapshot)
        self.assertEqual(decision.route, "answer_without_tools")
        contract = router.build_contract(decision, snapshot)
        gateway = ExecutionGateway(contract, snapshot)
        result = gateway.validate("mcp_cap.sick_op", {"query": "x"})
        self.assertEqual(result.status, ToolCallStatus.ROUTE_MISMATCH)


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.clock = {"t": 0.0}
        self.breaker = CircuitBreaker(
            failure_threshold=3, cooldown_seconds=60.0, now_fn=lambda: self.clock["t"]
        )

    def test_opens_after_threshold(self):
        for _ in range(3):
            self.breaker.record_failure("cap.x")
        allowed, detail = self.breaker.allow("cap.x")
        self.assertFalse(allowed)
        self.assertIn("open_cooldown", detail)
        self.assertEqual(self.breaker.state_of("cap.x"), "open")

    def test_success_resets_count(self):
        self.breaker.record_failure("cap.x")
        self.breaker.record_failure("cap.x")
        self.breaker.record_success("cap.x")
        self.breaker.record_failure("cap.x")
        self.breaker.record_failure("cap.x")
        allowed, _ = self.breaker.allow("cap.x")
        self.assertTrue(allowed)  # 只有 2 次连续失败，未达阈值

    def test_isolated_per_key(self):
        for _ in range(5):
            self.breaker.record_failure("cap.bad")
        allowed_bad, _ = self.breaker.allow("cap.bad")
        allowed_good, _ = self.breaker.allow("cap.good")
        self.assertFalse(allowed_bad)
        self.assertTrue(allowed_good)

    def test_half_open_probe_and_recovery(self):
        for _ in range(3):
            self.breaker.record_failure("cap.x")
        self.clock["t"] = 61.0
        allowed1, detail1 = self.breaker.allow("cap.x")
        self.assertTrue(allowed1)
        self.assertEqual(detail1, "half_open_probe")
        # 探测并发时第二次仍拒绝
        allowed2, _ = self.breaker.allow("cap.x")
        self.assertFalse(allowed2)
        # 探测成功 -> closed
        self.breaker.record_success("cap.x")
        self.assertEqual(self.breaker.state_of("cap.x"), "closed")

    def test_probe_failure_reopens(self):
        for _ in range(3):
            self.breaker.record_failure("cap.x")
        self.clock["t"] = 61.0
        self.breaker.allow("cap.x")  # 进入探测
        self.breaker.record_failure("cap.x")
        self.clock["t"] = 70.0
        allowed, _ = self.breaker.allow("cap.x")
        self.assertFalse(allowed)


class TestGatewayBreakerIntegration(unittest.TestCase):
    def test_repeated_schema_failures_trip_breaker(self):
        capability = make_capability("cap.fragile")
        gateway, _, snapshot, _ = build_gateway(capability)
        # 熔断阈值 3：三次 schema 错误后，即使参数正确也拒绝
        for _ in range(3):
            result = gateway.validate("mcp_cap.fragile_op", {})
            self.assertEqual(result.status, ToolCallStatus.SCHEMA_ERROR)
        result = gateway.validate("mcp_cap.fragile_op", {"query": "ok"})
        self.assertEqual(result.status, ToolCallStatus.PROVIDER_UNAVAILABLE)
        self.assertIn("熔断", result.detail)

    def test_success_keeps_breaker_closed(self):
        capability = make_capability("cap.stable")
        gateway, _, _, _ = build_gateway(capability)
        for _ in range(5):
            result = gateway.validate("mcp_cap.stable_op", {"query": "ok"})
            self.assertEqual(result.status, ToolCallStatus.SUCCESS)
            gateway.record_execution("mcp_cap.stable_op", success=True)
        result = gateway.validate("mcp_cap.stable_op", {"query": "ok"})
        self.assertEqual(result.status, ToolCallStatus.SUCCESS)


if __name__ == "__main__":
    unittest.main()
