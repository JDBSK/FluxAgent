# -*- coding: utf-8 -*-
"""Capability Registry：全量能力的唯一登记处。

设计要点：
- 全量能力只存在这里和检索索引中，永远不会整体交给执行模型；
- fail-closed：重名 capability、跨能力别名冲突、坏 JSON Schema、非法结构
  一律拒绝注册，不允许"静默跳过"隐藏部署错误；
- 版本化快照：Router / Compiler / Gateway / MCP Server 消费同一份不可变
  snapshot，版本号 = 全部指纹的 SHA-256，任何变更都会使旧 snapshot 失效；
- JSON Schema 用 jsonschema 编译校验，保证 Gateway 执行前校验可用。
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from jsonschema import Draft202012Validator

from .descriptors import CapabilityDescriptor, OperationDescriptor


class RegistryError(ValueError):
    """注册表 fail-closed 错误：调用方必须修复描述而不是绕过校验。"""


def validate_operation_schema(schema: Mapping[str, Any]) -> None:
    """校验 operation 的 input_schema 可以作为 JSON Schema 编译。"""
    try:
        Draft202012Validator.check_schema(dict(schema))
    except Exception as exc:  # noqa: BLE001 - 任何 schema 错误都必须 fail closed
        raise RegistryError(f"operation input_schema 非法: {exc}") from exc


@dataclass(frozen=True)
class RegistrySnapshot:
    """不可变注册表视图：Router/Compiler/Gateway 的统一输入。

    version 与 content_version 的分工：
    - version（含健康+启用）：契约与网关的失效 key——能力运行中转为
      UNAVAILABLE 时，旧契约必须被 Gateway 拒绝并强制重路由（安全属性）；
    - content_version（仅内容+启用）：检索索引的失效 key——语料与语料
      选择不随健康变化，健康翻转不应触发（可能含远程 embedding 的）
      全量索引重建；运行时 reranker/Gateway 已按健康过滤。
    """

    version: str
    capabilities: Tuple[CapabilityDescriptor, ...]
    content_version: str = ""

    def get(self, capability_id: str) -> Optional[CapabilityDescriptor]:
        for capability in self.capabilities:
            if capability.capability_id == capability_id:
                return capability
        return None

    def find_by_operation(self, tool_name: str) -> Optional[OperationDescriptor]:
        """按执行工具名反查 operation（Gateway 二次校验用）。"""
        for capability in self.capabilities:
            for op in capability.operations:
                if op.name == tool_name:
                    return op
        return None

    def enabled_capabilities(self) -> Tuple[CapabilityDescriptor, ...]:
        return tuple(c for c in self.capabilities if c.enabled)


class CapabilityRegistry:
    """线程安全的统一能力注册表。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._capabilities: Dict[str, CapabilityDescriptor] = {}

    # ---------- 注册 / 变更 ----------

    def register(self, descriptor: CapabilityDescriptor) -> None:
        """注册新能力；capability_id 重复或别名冲突时 fail-closed。"""
        self._validate(descriptor)
        with self._lock:
            if descriptor.capability_id in self._capabilities:
                raise RegistryError(
                    f"capability_id 已存在，请使用 refresh() 替换: {descriptor.capability_id}"
                )
            self._check_alias_conflicts(descriptor, ignore={descriptor.capability_id})
            self._capabilities[descriptor.capability_id] = descriptor
        self._invalidate_cache()

    def refresh(self, descriptor: CapabilityDescriptor) -> None:
        """替换同 id 能力（版本升级/健康变化）；仍校验别名冲突。"""
        self._validate(descriptor)
        with self._lock:
            if descriptor.capability_id not in self._capabilities:
                raise RegistryError(
                    f"capability_id 不存在，请使用 register(): {descriptor.capability_id}"
                )
            self._check_alias_conflicts(
                descriptor, ignore={descriptor.capability_id}
            )
            self._capabilities[descriptor.capability_id] = descriptor
        self._invalidate_cache()

    def unregister(self, capability_id: str) -> bool:
        with self._lock:
            existed = self._capabilities.pop(capability_id, None) is not None
        if existed:
            self._invalidate_cache()
        return existed

    def set_enabled(self, capability_id: str, enabled: bool) -> bool:
        with self._lock:
            capability = self._capabilities.get(capability_id)
            if capability is None:
                return False
            replacement = CapabilityDescriptor(
                capability_id=capability.capability_id,
                version=capability.version,
                kind=capability.kind,
                name=capability.name,
                description=capability.description,
                aliases=capability.aliases,
                intents=capability.intents,
                examples=capability.examples,
                negative_examples=capability.negative_examples,
                operations=capability.operations,
                provider_health=capability.provider_health,
                enabled=enabled,
                context_provider=capability.context_provider,
                metadata=capability.metadata,
            )
            self._capabilities[capability_id] = replacement
        self._invalidate_cache()
        return True

    def update_health(self, capability_id: str, health) -> bool:
        with self._lock:
            capability = self._capabilities.get(capability_id)
            if capability is None:
                return False
            replacement = CapabilityDescriptor(
                capability_id=capability.capability_id,
                version=capability.version,
                kind=capability.kind,
                name=capability.name,
                description=capability.description,
                aliases=capability.aliases,
                intents=capability.intents,
                examples=capability.examples,
                negative_examples=capability.negative_examples,
                operations=capability.operations,
                provider_health=health,
                enabled=capability.enabled,
                context_provider=capability.context_provider,
                metadata=capability.metadata,
            )
            self._capabilities[capability_id] = replacement
        self._invalidate_cache()
        return True

    # ---------- 查询 ----------

    def get(self, capability_id: str) -> Optional[CapabilityDescriptor]:
        with self._lock:
            return self._capabilities.get(capability_id)

    def capabilities(self) -> Tuple[CapabilityDescriptor, ...]:
        with self._lock:
            return tuple(sorted(self._capabilities.values(), key=lambda c: c.capability_id))

    def __len__(self) -> int:
        with self._lock:
            return len(self._capabilities)

    def snapshot(self) -> RegistrySnapshot:
        with self._lock:
            capabilities = tuple(
                sorted(self._capabilities.values(), key=lambda c: c.capability_id)
            )
        version = _snapshot_version(capabilities)
        content_version = _content_version(capabilities)
        return RegistrySnapshot(
            version=version, capabilities=capabilities, content_version=content_version
        )

    # ---------- 内部 ----------

    def _validate(self, descriptor: CapabilityDescriptor) -> None:
        if not descriptor.capability_id or not descriptor.capability_id.strip():
            raise RegistryError("capability_id 不能为空")
        if not descriptor.version:
            raise RegistryError("capability version 不能为空")
        for op in descriptor.operations:
            expected_prefix = f"{descriptor.capability_id}/"
            if not op.operation_id.startswith(expected_prefix):
                raise RegistryError(
                    f"operation_id 必须是 'capability_id/name' 形式: {op.operation_id}"
                )
            if op.name != op.operation_id[len(expected_prefix):]:
                raise RegistryError(
                    f"operation_id 与操作名不一致: {op.operation_id} vs {op.name}"
                )
            validate_operation_schema(op.input_schema)

    def _check_alias_conflicts(
        self,
        descriptor: CapabilityDescriptor,
        ignore: set[str],
    ) -> None:
        """别名是显式路由的钥匙，跨能力冲突必须 fail-closed。"""
        seen: Dict[str, str] = {}
        for existing_id, existing in self._capabilities.items():
            if existing_id in ignore:
                continue
            for alias in self._aliases_of(existing):
                seen.setdefault(alias, existing_id)
        for alias in self._aliases_of(descriptor):
            owner = seen.get(alias)
            if owner and owner != descriptor.capability_id:
                raise RegistryError(
                    f"别名冲突: '{alias}' 已属于能力 {owner}"
                )

    @staticmethod
    def _aliases_of(descriptor: CapabilityDescriptor) -> Tuple[str, ...]:
        aliases = {alias.strip().casefold() for alias in descriptor.aliases if alias.strip()}
        aliases.add(descriptor.capability_id.casefold())
        aliases.add(descriptor.name.strip().casefold())
        return tuple(aliases)

    @staticmethod
    def _version_of(capabilities: Tuple[CapabilityDescriptor, ...]) -> str:
        payload = json.dumps(
            [c.content_fingerprint() for c in capabilities],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # ---------- 快照缓存 ----------

    def _invalidate_cache(self) -> None:
        # snapshot() 每次重新计算版本；缓存位预留，保证以后加缓存不漏失效
        self._snapshot_cache = None


def build_registry_snapshot(registry: CapabilityRegistry) -> RegistrySnapshot:
    return registry.snapshot()


# Descriptor 的 content_fingerprint 不包含 provider_health（健康是运行时状态，
# 不是能力内容）。健康变化必须反映到快照版本，因此快照版本叠加健康状态。
def _snapshot_version(capabilities) -> str:
    payload = json.dumps(
        [[c.content_fingerprint(), c.provider_health.value, c.enabled] for c in capabilities],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# 索引失效 key：仅内容（指纹 + enabled）。健康是运行时状态，不改变语料——
# 健康翻转推进 version（契约失效）但不推进 content_version（索引不重建）。
def _content_version(capabilities) -> str:
    payload = json.dumps(
        [[c.content_fingerprint(), c.enabled] for c in capabilities],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
