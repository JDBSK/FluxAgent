# -*- coding: utf-8 -*-
"""Execution Gateway：模型输出不是授权依据。

每次工具调用在真正执行前必须通过：
  1. InvocationContract 成员校验（越权 -> route_mismatch，不进审批）；
  2. Registry 快照版本校验（快照过期 -> stale_contract）；
  3. 真实 JSON Schema 参数校验（失败 -> schema_error，禁止换参数重试）；
  4. 能力健康状态（不可用 -> provider_unavailable）；
  5. 熔断状态机（open -> provider_unavailable，不影响无关能力）。

风险/审批仍由 Agent 图中的 policy/assessment 层负责（单一事实源），
Gateway 只在执行边界做机械校验。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

from jsonschema import Draft202012Validator

from .descriptors import ProviderHealth
from .registry import RegistrySnapshot
from .router import InvocationContract


class ToolCallStatus(str, Enum):
    SUCCESS = "success"
    SCHEMA_ERROR = "schema_error"
    ROUTE_MISMATCH = "route_mismatch"
    STALE_CONTRACT = "stale_contract"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_REJECTED = "approval_rejected"
    AUTH_DENIED = "auth_denied"
    TRANSIENT_TRANSPORT = "transient_transport"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    EXECUTION_ERROR = "execution_error"


# 当前契约内必须立即停止重试的状态（失败分类器据此禁止 retry）
STOP_RETRY_STATUSES = frozenset(
    {
        ToolCallStatus.SCHEMA_ERROR,
        ToolCallStatus.ROUTE_MISMATCH,
        ToolCallStatus.STALE_CONTRACT,
        ToolCallStatus.APPROVAL_REJECTED,
        ToolCallStatus.AUTH_DENIED,
    }
)


@dataclass(frozen=True)
class GatewayResult:
    status: ToolCallStatus
    tool: str
    detail: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ToolCallStatus.SUCCESS

    def to_tool_message_content(self) -> str:
        """统一错误 envelope：失败分类器和模型都消费这一格式。"""
        import json

        retryable = self.status == ToolCallStatus.TRANSIENT_TRANSPORT
        return json.dumps(
            {
                "ok": False,
                "error": {
                    "code": self.status.value,
                    "message": self.detail,
                    "retryable": retryable,
                    "retry_policy": (
                        "stop"
                        if self.status in STOP_RETRY_STATUSES
                        else "limited_backoff"
                    ),
                },
            },
            ensure_ascii=False,
        )


# ---------- 熔断状态机 ----------

class CircuitBreaker:
    """按 key（capability/provider）维度的 closed -> open -> half_open 熔断。

    - 连续 failure_threshold 次失败 -> open；
    - cooldown 内拒绝所有调用（不影响其他 key）；
    - cooldown 结束进入 half_open，放行一次探测；
    - 探测成功 -> closed；失败 -> 重新 open。
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 60.0,
        now_fn: Optional[Any] = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold 至少为 1")
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._now = now_fn or time.monotonic
        self._lock = threading.RLock()
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}
        self._probing: set[str] = set()

    def allow(self, key: str) -> tuple[bool, str]:
        now = self._now()
        with self._lock:
            if key not in self._opened_at:
                return True, "closed"
            elapsed = now - self._opened_at[key]
            if elapsed >= self.cooldown_seconds:
                if key not in self._probing:
                    self._probing.add(key)
                    return True, "half_open_probe"
                return False, "half_open_probe_in_flight"
            return False, f"open_cooldown:{elapsed:.1f}s/{self.cooldown_seconds:.0f}s"

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
            self._opened_at.pop(key, None)
            self._probing.discard(key)

    def record_failure(self, key: str) -> None:
        now = self._now()
        with self._lock:
            count = self._failures.get(key, 0) + 1
            self._failures[key] = count
            self._probing.discard(key)
            if count >= self.failure_threshold or key in self._opened_at:
                self._opened_at[key] = now

    def state_of(self, key: str) -> str:
        with self._lock:
            if key not in self._opened_at:
                return "closed"
            if key in self._probing:
                return "half_open"
            if self._now() - self._opened_at[key] >= self.cooldown_seconds:
                return "half_open"
            return "open"


# ---------- Gateway ----------

class ExecutionGateway:
    """执行边界的机械校验层；持有一份契约和一份 registry snapshot。"""

    def __init__(
        self,
        contract: InvocationContract,
        snapshot: RegistrySnapshot,
        *,
        breaker: Optional[CircuitBreaker] = None,
    ) -> None:
        self.contract = contract
        self.snapshot = snapshot
        self.breaker = breaker or CircuitBreaker()

    # ---------- 校验 ----------

    def validate(self, tool_name: str, args: Optional[Mapping[str, Any]]) -> GatewayResult:
        """参数网关：在执行前检查契约、注册表、能力健康、熔断状态和 JSON Schema。
        返回 GatewayResult，status=SUCCESS 表示可以执行；否则禁止执行。"""
        args = dict(args or {})
        if self.contract.is_expired():
            return GatewayResult(
                ToolCallStatus.STALE_CONTRACT, tool_name, "契约已过期，需要重新路由"
            )
        if self.contract.decision.registry_version != self.snapshot.version:
            return GatewayResult(
                ToolCallStatus.STALE_CONTRACT,
                tool_name,
                f"契约版本 {self.contract.decision.registry_version} 与当前注册表 "
                f"{self.snapshot.version} 不一致，需要重新路由",
            )
        if not self.contract.allows(tool_name):
            return GatewayResult(
                ToolCallStatus.ROUTE_MISMATCH,
                tool_name,
                "工具不在本回合 Invocation contract 允许集合内，"
                "禁止调用，也禁止换参数重试；请直接回答用户或重新路由。",
                {"allowed": sorted(self.contract.tools)},
            )

        operation = self.snapshot.find_by_operation(tool_name)
        if operation is None:
            return GatewayResult(
                ToolCallStatus.ROUTE_MISMATCH, tool_name, "工具未在注册表中登记"
            )
        capability = self.snapshot.get(operation.capability_id)
        if capability is None:
            return GatewayResult(
                ToolCallStatus.ROUTE_MISMATCH, tool_name, "能力未在注册表中登记"
            )
        if capability.provider_health == ProviderHealth.UNAVAILABLE:
            return GatewayResult(
                ToolCallStatus.PROVIDER_UNAVAILABLE,
                tool_name,
                f"能力 {capability.capability_id} 当前不可用",
            )

        allowed, breaker_detail = self.breaker.allow(capability.capability_id)
        if not allowed:
            return GatewayResult(
                ToolCallStatus.PROVIDER_UNAVAILABLE,
                tool_name,
                f"能力 {capability.capability_id} 熔断中: {breaker_detail}",
            )

        schema_error = self._validate_schema(operation.input_schema, args)
        if schema_error:
            self.breaker.record_failure(capability.capability_id)
            return GatewayResult(
                ToolCallStatus.SCHEMA_ERROR,
                tool_name,
                f"参数未通过 JSON Schema 校验: {schema_error}。"
                "本回合禁止换参数重试同一工具。",
            )
        
        return GatewayResult(
            ToolCallStatus.SUCCESS,
            tool_name,
            "",
            {"capability_id": capability.capability_id},
        )

    def record_execution(self, tool_name: str, *, success: bool, transient: bool = False) -> None:
        """执行后回报：schema/路由类失败不应在这里重复计熔断（validate 已计）。"""
        operation = self.snapshot.find_by_operation(tool_name)
        key = operation.capability_id if operation else tool_name
        if success:
            self.breaker.record_success(key)
        elif transient:
            self.breaker.record_failure(key)

    @staticmethod
    def _validate_schema(schema: Mapping[str, Any], args: Mapping[str, Any]) -> str:
        try:
            validator = Draft202012Validator(dict(schema))
            errors = sorted(validator.iter_errors(dict(args)), key=lambda e: list(e.path))
            if not errors:
                return ""
            first = errors[0]
            path = ".".join(str(p) for p in first.path) or "(root)"
            return f"{path}: {first.message}"
        except Exception as exc:  # noqa: BLE001 - schema 本身坏掉也视为校验失败
            return f"schema 编译失败: {exc}"
