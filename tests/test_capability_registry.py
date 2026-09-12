# -*- coding: utf-8 -*-
"""统一 Capability Registry 契约测试。

所有 fixture 使用通用合成能力，不引用任何现有安装项。
"""
from __future__ import annotations

import unittest

from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    ProviderHealth,
    RiskLevel,
)
from fluxagent.capability.registry import (
    CapabilityRegistry,
    RegistryError,
    validate_operation_schema,
)


def make_capability(
    capability_id: str = "cap.alpha",
    name: str = "Alpha",
    aliases: tuple = (),
    kind: str = "mcp",
    enabled: bool = True,
    health: ProviderHealth = ProviderHealth.HEALTHY,
    operations: tuple = (),
) -> CapabilityDescriptor:
    if not operations:
        operations = (
            OperationDescriptor(
                capability_id=capability_id,
                name=f"{capability_id}.op",
                operation_id=f"{capability_id}/{capability_id}.op",
                description="合成操作",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                required_slots=("query",),
                side_effect_level=RiskLevel.READ,
            ),
        )
    return CapabilityDescriptor(
        capability_id=capability_id,
        version="1",
        kind=kind,
        name=name,
        description="合成能力",
        aliases=aliases,
        operations=operations,
        provider_health=health,
        enabled=enabled,
    )


class TestDescriptorValidation(unittest.TestCase):
    def test_prompt_skill_cannot_have_operations(self):
        op = OperationDescriptor(
            capability_id="cap.doc",
            name="run",
            operation_id="cap.doc/run",
        )
        with self.assertRaises(ValueError):
            CapabilityDescriptor(
                capability_id="cap.doc",
                version="1",
                kind="prompt_skill",
                name="Doc",
                operations=(op,),
            )

    def test_executable_kind_requires_operations(self):
        with self.assertRaises(ValueError):
            CapabilityDescriptor(
                capability_id="cap.empty",
                version="1",
                kind="mcp",
                name="Empty",
                operations=(),
            )

    def test_operation_must_belong_to_capability(self):
        op = OperationDescriptor(
            capability_id="cap.other",
            name="run",
            operation_id="cap.other/run",
        )
        with self.assertRaises(ValueError):
            CapabilityDescriptor(
                capability_id="cap.main",
                version="1",
                kind="mcp",
                name="Main",
                operations=(op,),
            )

    def test_bad_schema_fails_validation(self):
        with self.assertRaises(RegistryError):
            validate_operation_schema({"type": "not-a-type"})


class TestRegistryContract(unittest.TestCase):
    def test_register_and_snapshot_version_changes(self):
        registry = CapabilityRegistry()
        snapshot1 = registry.snapshot()
        registry.register(make_capability())
        snapshot2 = registry.snapshot()
        self.assertEqual(len(snapshot1.capabilities), 0)
        self.assertEqual(len(snapshot2.capabilities), 1)
        self.assertNotEqual(snapshot1.version, snapshot2.version)
        # 内容不变则版本稳定
        self.assertEqual(snapshot2.version, registry.snapshot().version)

    def test_duplicate_capability_id_fails_closed(self):
        registry = CapabilityRegistry()
        registry.register(make_capability())
        with self.assertRaises(RegistryError):
            registry.register(make_capability())

    def test_refresh_replaces_same_id(self):
        registry = CapabilityRegistry()
        registry.register(make_capability(version_id=None) if False else make_capability())
        updated = make_capability()
        registry.refresh(updated)
        self.assertEqual(len(registry), 1)

    def test_refresh_missing_id_fails(self):
        registry = CapabilityRegistry()
        with self.assertRaises(RegistryError):
            registry.refresh(make_capability())

    def test_alias_conflict_fails_closed(self):
        registry = CapabilityRegistry()
        registry.register(make_capability(capability_id="cap.one", name="One", aliases=("共享别名",)))
        with self.assertRaises(RegistryError):
            registry.register(make_capability(capability_id="cap.two", name="Two", aliases=("共享别名",)))

    def test_name_conflict_fails_closed(self):
        registry = CapabilityRegistry()
        registry.register(make_capability(capability_id="cap.one", name="Same"))
        with self.assertRaises(RegistryError):
            registry.register(make_capability(capability_id="cap.two", name="Same"))

    def test_unregister_and_disable(self):
        registry = CapabilityRegistry()
        registry.register(make_capability())
        self.assertTrue(registry.unregister("cap.alpha"))
        self.assertEqual(len(registry), 0)
        registry.register(make_capability())
        self.assertTrue(registry.set_enabled("cap.alpha", False))
        snapshot = registry.snapshot()
        self.assertEqual(snapshot.enabled_capabilities(), ())
        self.assertEqual(len(snapshot.capabilities), 1)

    def test_find_by_operation(self):
        registry = CapabilityRegistry()
        registry.register(make_capability())
        snapshot = registry.snapshot()
        op = snapshot.find_by_operation("cap.alpha.op")
        self.assertIsNotNone(op)
        self.assertEqual(op.capability_id, "cap.alpha")
        self.assertIsNone(snapshot.find_by_operation("missing.tool"))

    def test_health_update_changes_version(self):
        registry = CapabilityRegistry()
        registry.register(make_capability())
        v1 = registry.snapshot().version
        registry.update_health("cap.alpha", ProviderHealth.UNAVAILABLE)
        v2 = registry.snapshot().version
        self.assertNotEqual(v1, v2)
        self.assertEqual(
            registry.get("cap.alpha").provider_health, ProviderHealth.UNAVAILABLE
        )

    def test_operation_id_prefix_enforced(self):
        registry = CapabilityRegistry()
        bad_op = OperationDescriptor(
            capability_id="cap.alpha",
            name="op",
            operation_id="cap.other/op",
        )
        with self.assertRaises(RegistryError):
            registry.register(make_capability(operations=(bad_op,)))


if __name__ == "__main__":
    unittest.main()
