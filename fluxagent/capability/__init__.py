# -*- coding: utf-8 -*-
"""统一能力控制面。

任何能力来源（内置工具、外部 MCP、Prompt Skill、可执行 Skill）都必须转换成
本包定义的统一描述结构后进入 Registry；Router、Context Compiler 和
Execution Gateway 只消费这些结构，不感知也不特判任何具体服务。
"""
from .descriptors import (
    CAPABILITY_KINDS,
    CapabilityDescriptor,
    OperationDescriptor,
    ProviderHealth,
    RiskLevel,
)
from .registry import CapabilityRegistry, RegistrySnapshot

__all__ = [
    "CAPABILITY_KINDS",
    "CapabilityDescriptor",
    "OperationDescriptor",
    "ProviderHealth",
    "RiskLevel",
    "CapabilityRegistry",
    "RegistrySnapshot",
]
