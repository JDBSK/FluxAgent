# -*- coding: utf-8 -*-
"""Provider 适配层：把三类现有发现器输出转换为统一 CapabilityDescriptor。

边界：
- 本模块只做结构转换，不包含任何具体能力/服务/工具名称；
- Builtin  -> 静态 LangChain 工具列表；
- MCP      -> installer/index.json 的发现结果（record + remote_tools）；
- Skill    -> skill_loader 的扫描结果（有脚本=executable_skill，
              无脚本=prompt_skill，只提供上下文，不产生虚假执行操作）；
- 未来来源只需产出 CapabilityDescriptor，不需要改 Router/Registry。
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from langchain_core.tools import BaseTool

from ..core.policy import TOOL_BASE_LEVEL
from .descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    ProviderHealth,
    RiskLevel,
)


_RISK_BY_LEVEL = {
    "read": RiskLevel.READ,
    "write": RiskLevel.WRITE,
    "mutate": RiskLevel.MUTATE,
    "exec": RiskLevel.EXEC,
    "critical": RiskLevel.CRITICAL,
}

# 与 mcp.client._exposed_name 保持一致的暴露名清理规则（避免循环导入）
_MCP_NAME_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MCP_NAME_DIGEST_RE = re.compile(r"[0-9a-f]{8}")


def _sanitize_mcp_name(raw: str) -> str:
    import hashlib as _hashlib

    cleaned = _MCP_NAME_SANITIZE_RE.sub("_", raw).strip("_") or "mcp_tool"
    if len(cleaned) <= 64:
        return cleaned
    suffix = _hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:55]}_{suffix}"


def _schema_of_tool(tool: BaseTool) -> Mapping[str, Any]:
    schema_obj = getattr(tool, "args_schema", None)
    if schema_obj is None:
        return {"type": "object", "properties": {}}
    try:
        raw = schema_obj.model_json_schema()
    except Exception:  # noqa: BLE001 - 坏 schema 在注册表 fail-closed，不在适配层吞掉
        raise
    raw.pop("title", None)
    return raw


def _risk_of_tool(tool: BaseTool) -> RiskLevel:
    level = TOOL_BASE_LEVEL.get(str(tool.name))
    if level is not None:
        return _RISK_BY_LEVEL.get(getattr(level, "value", str(level)), RiskLevel.WRITE)
    metadata = getattr(tool, "metadata", None) or {}
    declared = str(metadata.get("risk", "")).casefold()
    if declared in _RISK_BY_LEVEL:
        return _RISK_BY_LEVEL[declared]
    return RiskLevel.WRITE


def _slug(text: str) -> str:
    """规范化字符串为能力ID片段,避免与 skill/mcp 冲突."""
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", str(text)).strip("_")
    return slug or "capability"


def builtin_descriptors(tools: Iterable[BaseTool]) -> list[CapabilityDescriptor]:
    """静态内置工具 -> builtin 描述。每个工具一个能力（内置工具即最小能力单元）。"""
    descriptors: list[CapabilityDescriptor] = []
    for tool in tools:
        name = str(tool.name)
        capability_id = f"builtin.{_slug(name)}"  # builtin 工具按名称分组，避免与 skill/mcp 冲突
        op = OperationDescriptor(
            capability_id=capability_id,
            name=name,
            operation_id=f"{capability_id}/{name}",
            description=str(getattr(tool, "description", "") or ""),
            input_schema=_schema_of_tool(tool),
            side_effect_level=_risk_of_tool(tool),
            tool=tool,
        )
        descriptors.append(
            CapabilityDescriptor(
                capability_id=capability_id,
                version="1",
                kind="builtin",
                name=name,
                description=str(getattr(tool, "description", "") or ""),
                intents=(),
                operations=(op,),
                provider_health=ProviderHealth.HEALTHY,
                metadata={"aliases": list((getattr(tool, "metadata", None) or {}).get("aliases", []))},
            )
        )
    return descriptors


def mcp_descriptors(
    records_with_tools: Iterable[tuple[Mapping[str, Any], Iterable[Any]]],
) -> list[CapabilityDescriptor]:
    """(index.json record, 发现的 remote_tools) -> mcp 描述。

    remote_tool 需要具备 name/description/inputSchema；与 mcp.client 的
    发现输出一致。read_only_tools 声明映射为 READ 副作用等级。
    """
    descriptors: list[CapabilityDescriptor] = []
    for record, remote_tools in records_with_tools:
        server_name = str(record.get("name") or "unnamed")
        capability_id = f"mcp.{_slug(server_name)}"
        read_only = {str(n) for n in (record.get("read_only_tools") or [])}
        operations: list[OperationDescriptor] = []
        for remote in remote_tools:
            remote_name = str(getattr(remote, "name", "") or "")
            if not remote_name:
                continue
            exposed = _sanitize_mcp_name(f"mcp_{server_name}_{remote_name}")
            schema = (
                getattr(remote, "input_schema", None)
                or getattr(remote, "inputSchema", None)
                or {"type": "object", "properties": {}}
            )
            required = tuple(schema.get("required", []) or []) if isinstance(schema, Mapping) else ()
            operations.append(
                OperationDescriptor(
                    capability_id=capability_id,
                    name=exposed,
                    operation_id=f"{capability_id}/{exposed}",
                    description=str(getattr(remote, "description", "") or ""),
                    input_schema=dict(schema),
                    required_slots=required,
                    side_effect_level=RiskLevel.READ if remote_name in read_only else RiskLevel.WRITE,
                    tool=None,  # 执行工具由 mcp.client 的 _make_tool 提供，通过 tool_bindings 注入
                    metadata={"server": server_name, "remote_name": remote_name},
                )
            )
        descriptors.append(
            CapabilityDescriptor(
                capability_id=capability_id,
                version="1",
                kind="mcp",
                name=server_name,
                description=f"MCP server {server_name}",
                aliases=(server_name,),
                operations=tuple(operations),
                provider_health=ProviderHealth.HEALTHY if operations else ProviderHealth.UNAVAILABLE,
                metadata={"server": server_name},
            )
        )
    return descriptors


def skill_descriptors(
    skill_infos: Iterable[Mapping[str, Any]],
    *,
    content_loader: Optional[Any] = None,
    tool_loader: Optional[Any] = None,
) -> list[CapabilityDescriptor]:
    """skill_loader._scan_skills() 输出 -> 统一描述。

    - 无脚本：prompt_skill，只有 context_provider（懒读说明书）；
    - 有脚本：executable_skill，操作 = loader 生成的 help/run StructuredTool。
    content_loader/tool_loader 注入避免本模块依赖 loader 单例。
    """
    descriptors: list[CapabilityDescriptor] = []
    for info in skill_infos:
        folder = str(info.get("folder") or "")
        raw_name = str(info.get("raw_name") or folder)
        capability_id = f"skill.{_slug(str(info.get('name') or raw_name))}"
        scripts = list(info.get("scripts") or [])
        aliases = tuple(
            dict.fromkeys(
                str(a) for a in (info.get("aliases") or []) if str(a).strip()
            )
        )
        description = str(info.get("description") or "")

        context_provider = None
        if content_loader is not None:
            md_path = info.get("md_path")

            def _read_context(path=md_path):
                try:
                    return str(content_loader(path))
                except Exception as exc:  # noqa: BLE001 - 上下文读取失败降级为空片段
                    return f"[skill 上下文不可用: {exc}]"

            context_provider = _read_context

        operations: tuple[OperationDescriptor, ...] = ()
        health = ProviderHealth.HEALTHY
        if scripts:
            tool = tool_loader(info) if tool_loader is not None else None
            if tool is None:
                health = ProviderHealth.UNAVAILABLE
            else:
                schema = _schema_of_tool(tool)
                operations = (
                    OperationDescriptor(
                        capability_id=capability_id,
                        name=str(tool.name),
                        operation_id=f"{capability_id}/{tool.name}",
                        description=description,
                        input_schema=schema,
                        required_slots=tuple(schema.get("required", []) or []),
                        side_effect_level=_risk_of_tool(tool),
                        tool=tool,
                        metadata={"folder": folder},
                    ),
                )
        else:
            operations = ()

        descriptors.append(
            CapabilityDescriptor(
                capability_id=capability_id,
                version="1",
                kind="prompt_skill" if not scripts else "executable_skill",
                name=raw_name,
                description=description,
                aliases=aliases,
                operations=operations,
                provider_health=health,
                context_provider=context_provider,
                metadata={"folder": folder},
            )
        )
    return descriptors


def bind_operation_tools(
    descriptor: CapabilityDescriptor,
    tools_by_name: Mapping[str, BaseTool],
) -> CapabilityDescriptor:
    """把实际执行工具绑定到 descriptor 的 operation 上（MCP 场景：工具由
    client._make_tool 构造，与 descriptor 分开产生）。"""
    rebound: list[OperationDescriptor] = []
    for op in descriptor.operations:
        tool = tools_by_name.get(op.name)
        rebound.append(
            OperationDescriptor(
                capability_id=op.capability_id,
                name=op.name,
                operation_id=op.operation_id,
                description=op.description,
                input_schema=op.input_schema,
                required_slots=op.required_slots,
                side_effect_level=op.side_effect_level,
                requires_approval=op.requires_approval,
                examples=op.examples,
                tool=tool if tool is not None else op.tool,
                metadata=op.metadata,
            )
        )
    return CapabilityDescriptor(
        capability_id=descriptor.capability_id,
        version=descriptor.version,
        kind=descriptor.kind,
        name=descriptor.name,
        description=descriptor.description,
        aliases=descriptor.aliases,
        intents=descriptor.intents,
        examples=descriptor.examples,
        negative_examples=descriptor.negative_examples,
        operations=tuple(rebound),
        provider_health=descriptor.provider_health,
        enabled=descriptor.enabled,
        context_provider=descriptor.context_provider,
        metadata=descriptor.metadata,
    )


def descriptors_from_tools(tools: Iterable[BaseTool]) -> list[CapabilityDescriptor]:
    """无显式 Provider 时的 fallback：按工具 metadata 派生统一描述。

    仅用于测试注入/旧调用方兼容；CLI 主路径必须走
    builtin_descriptors / mcp_descriptors / skill_descriptors 的显式适配。
    capability metadata 缺失的工具视为独立 builtin 能力；skill/mcp 工具按
    metadata.capability 分组。skill 工具的 prompt/executable 区分在 fallback
    中无法从工具对象还原，统一登记为 executable_skill 以保持既有执行行为。
    """
    groups: dict[str, dict[str, Any]] = {}
    for tool in tools:
        metadata = getattr(tool, "metadata", None) or {}
        capability = str(metadata.get("capability") or "").strip()
        kind_md = str(metadata.get("kind") or "").strip()
        name = str(tool.name)
        if not capability and name.startswith("mcp_"):
            # 与历史路由的降级规则一致：mcp_<server>_<tool> 命名即可路由
            capability = name.split("_", 2)[1]
            kind_md = "mcp"
        if not capability:
            group_key = f"builtin.{_slug(name)}"
            kind = "builtin"
            display = name
            group_aliases: list[str] = []
        elif kind_md == "builtin":
            group_key = f"builtin.{_slug(capability)}"
            kind = "builtin"
            display = capability
            group_aliases = [str(a) for a in (metadata.get("aliases") or []) if str(a).strip()]
        else:
            kind = "mcp" if kind_md == "mcp" else "executable_skill"
            group_key = f"{kind}.{_slug(capability)}"
            display = capability
            group_aliases = [str(a) for a in (metadata.get("aliases") or []) if str(a).strip()]
        group = groups.setdefault(
            group_key,
            {
                "capability_id": group_key,
                "kind": kind,
                "display": display,
                "aliases": group_aliases,
                "description": str(getattr(tool, "description", "") or ""),
                "operations": [],
            },
        )
        schema = _schema_of_tool(tool)
        group["operations"].append(
            OperationDescriptor(
                capability_id=group_key,
                name=name,
                operation_id=f"{group_key}/{name}",
                description=str(getattr(tool, "description", "") or ""),
                input_schema=schema,
                required_slots=tuple(schema.get("required", []) or []),
                side_effect_level=_risk_of_tool(tool),
                tool=tool,
            )
        )

    descriptors: list[CapabilityDescriptor] = []
    for group in groups.values():
        descriptors.append(
            CapabilityDescriptor(
                capability_id=group["capability_id"],
                version="1",
                kind=group["kind"],
                name=group["display"],
                description=group["description"],
                aliases=tuple(group["aliases"]),
                operations=tuple(group["operations"]),
                provider_health=ProviderHealth.HEALTHY,
            )
        )
    return descriptors
