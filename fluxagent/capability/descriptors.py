# -*- coding: utf-8 -*-
"""统一能力描述模型。

设计要点：
- 来源无关：builtin / mcp / prompt_skill / executable_skill 都输出同一结构；
- 不可变：descriptor 定义后内容指纹固定，Registry 版本由指纹推导；
- 双通道：可执行能力暴露 OperationDescriptor（含真实 JSON Schema），
  纯文档能力只暴露 context_provider（受限上下文片段），不会被虚构为工具。

本模块不知道也不允许出现任何具体能力、服务或工具名称。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Tuple


CAPABILITY_KINDS = frozenset({"builtin", "mcp", "prompt_skill", "executable_skill"})


class RiskLevel(str, Enum):
    """与 policy.RiskLevel 对齐的操作副作用等级（字符串可序列化）。"""

    READ = "read"
    WRITE = "write"
    MUTATE = "mutate"
    EXEC = "exec"
    CRITICAL = "critical"


class ProviderHealth(str, Enum):
    """能力来源进程/索引的健康状态；路由与网关都必须检查。"""

    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class OperationDescriptor:
    """一个可执行操作的统一描述。

    operation_id 全局唯一（capability_id + "/" + name）；input_schema 是
    真实参数 schema，Execution Gateway 用它做执行前校验，禁止与实现漂移。
    """

    capability_id: str
    name: str
    operation_id: str
    description: str = ""
    input_schema: Mapping[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    required_slots: Tuple[str, ...] = ()
    side_effect_level: RiskLevel = RiskLevel.READ
    requires_approval: bool = False
    examples: Tuple[str, ...] = ()
    # 承载实际执行的 LangChain BaseTool（或等价可调用宿主）；纯上下文能力为 None
    tool: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def content_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "capability_id": self.capability_id,
                "name": self.name,
                "operation_id": self.operation_id,
                "description": self.description,
                "input_schema": self.input_schema,
                "required_slots": list(self.required_slots),
                "side_effect_level": self.side_effect_level.value,
                "requires_approval": self.requires_approval,
                "examples": list(self.examples),
                "metadata": dict(self.metadata),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CapabilityDescriptor:
    """一个能力的统一描述：检索、路由、上下文和执行都以此为唯一事实源。"""

    capability_id: str
    version: str
    kind: str
    name: str
    description: str = ""
    aliases: Tuple[str, ...] = ()
    intents: Tuple[str, ...] = ()
    examples: Tuple[str, ...] = ()
    negative_examples: Tuple[str, ...] = ()
    operations: Tuple[OperationDescriptor, ...] = ()
    provider_health: ProviderHealth = ProviderHealth.UNKNOWN
    enabled: bool = True
    # Prompt Skill 的受限上下文提供器：返回该 skill 的文档片段（供 Compiler 裁剪）
    context_provider: Optional[Callable[[], str]] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in CAPABILITY_KINDS:
            raise ValueError(f"未知能力类型: {self.kind}")
        if self.kind == "prompt_skill" and self.operations:
            raise ValueError(
                f"prompt_skill 不允许携带可执行操作: {self.capability_id}"
            )
        for op in self.operations:
            if op.capability_id != self.capability_id:
                raise ValueError(
                    f"operation {op.operation_id} 不属于能力 {self.capability_id}"
                )
        if self.kind in {"builtin", "mcp", "executable_skill"} and not self.operations:
            raise ValueError(f"可执行能力必须至少有一个操作: {self.capability_id}")

    def content_fingerprint(self) -> str:
        payload = json.dumps(
            {
                "capability_id": self.capability_id,
                "version": self.version,
                "kind": self.kind,
                "name": self.name,
                "description": self.description,
                "aliases": list(self.aliases),
                "intents": list(self.intents),
                "examples": list(self.examples),
                "negative_examples": list(self.negative_examples),
                "operations": [op.content_fingerprint() for op in self.operations],
                "enabled": self.enabled,
                "metadata": dict(self.metadata),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def searchable_text(self) -> str:
        """路由索引的检索文本：卡片正文 + 操作卡正文，由 Router 统一分词。"""
        parts = [
            self.capability_id,
            self.name,
            self.description,
            " ".join(self.aliases),
            " ".join(self.intents),
            " ".join(self.examples),
            " ".join(self.negative_examples),
        ]
        for op in self.operations:
            parts.append(op.name)
            parts.append(op.description)
            parts.append(" ".join(op.examples))
            parts.append(" ".join(op.required_slots))
        return "\n".join(part for part in parts if part)
