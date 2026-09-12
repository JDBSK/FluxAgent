"""Load registered stdio MCP servers as LangChain tools for FluxAgent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..core.config import MCP_SERVERS_INDEX
from ..core.installer import InstallError, list_installed_mcp_servers
from ..core.logger import audit_logger
from ..core.policy import RiskLevel


_TOOL_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _exception_detail(exc: BaseException) -> str:
    # 返回 TaskGroup/ExceptionGroup 失败的原因。
    nested = getattr(exc, "exceptions", None)
    if nested:
        details = [_exception_detail(item) for item in nested]
        return "; ".join(dict.fromkeys(detail for detail in details if detail))
    return str(exc).strip() or exc.__class__.__name__


@dataclass(frozen=True)
class MCPToolLoadResult:
    # MCP 工具加载结果，包括成功加载的工具、风险覆盖和错误信息。
    tools: list[BaseTool]
    risk_overrides: dict[str, RiskLevel]
    errors: list[str]
    # 与 tools 同批发现产生的统一能力描述（capability registry 用）
    descriptors: list[Any]


def _make_descriptors(record: dict[str, Any], remote_tools: list[Any], tools_by_name: dict[str, BaseTool]) -> Any:
    """由单个 server 的 record + 发现结果生成绑定工具的统一 CapabilityDescriptor。"""
    from ..capability.providers import bind_operation_tools, mcp_descriptors

    candidates = mcp_descriptors([(record, remote_tools)])
    if not candidates:
        return None
    return bind_operation_tools(candidates[0], tools_by_name)


def _exposed_name(server_name: str, tool_name: str) -> str:
    # 生成一个安全的工具名称，用于在 FluxAgent 中暴露 MCP 工具。
    raw = f"mcp_{server_name}_{tool_name}"
    cleaned = _TOOL_NAME_PATTERN.sub("_", raw).strip("_") or "mcp_tool"
    if len(cleaned) <= 64:
        return cleaned
    suffix = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:55]}_{suffix}"


def _server_params(record: dict[str, Any]) -> StdioServerParameters:
    # 从 MCP 服务器记录中提取 stdio 连接参数。
    command = str(record.get("command") or "").strip()
    if not command:
        raise ValueError("缺少启动 command")
    cwd = Path(str(record.get("installed_path") or "")).resolve()
    if not cwd.is_dir():
        raise ValueError(f"源码目录不存在: {cwd}")

    raw_env = record.get("env")
    env = None
    if isinstance(raw_env, dict):
        env = {**os.environ, **{str(k): str(v) for k, v in raw_env.items()}}

    return StdioServerParameters(
        command=command,
        args=[str(item) for item in (record.get("args") or [])],
        cwd=cwd,
        env=env,
        encoding="utf-8",
        encoding_error_handler="replace",
    )


async def _discover_server(record: dict[str, Any]) -> list[Any]:
    # 发现 MCP 服务器的可用工具。
    params = _server_params(record)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()  # 初始化 MCP 客户端会话。
            result = await session.list_tools()
            return list(result.tools)


def _render_tool_result(server_name: str, tool_name: str, result: Any) -> str:
    # 渲染 MCP 工具调用的结果为字符串。
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
        elif hasattr(block, "model_dump"):
            parts.append(json.dumps(block.model_dump(), ensure_ascii=False))
        else:
            parts.append(str(block))
    output = "\n".join(part for part in parts if part).strip()

    if not output:
        output = "MCP 工具调用完成，但没有返回文本内容。"
    if bool(getattr(result, "isError", False) or getattr(result, "is_error", False)):
        return f"❌ MCP {server_name}/{tool_name} 调用失败: {output}"
    return output


async def _call_server_tool(
    record: dict[str, Any], tool_name: str, arguments: dict[str, Any]
) -> str:
    # 调用远程 MCP 工具，并返回渲染后的结果。
    params = _server_params(record)
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return _render_tool_result(str(record.get("name") or "?"), tool_name, result)


def _make_tool(record: dict[str, Any], remote_tool: Any, exposed_name: str) -> BaseTool:
    # 创建一个本地工具实例，用于调用远程 MCP 工具。
    server_name = str(record.get("name") or "unknown")  # MCP 服务器的名称,unkonwn是兜底
    remote_name = str(remote_tool.name)

    async def invoke_remote(**kwargs: Any) -> str:
        # 异步调用远程 MCP 工具，并返回结果。
        return await asyncio.wait_for(
            _call_server_tool(record, remote_name, kwargs), timeout=60
        )

    def invoke_remote_sync(**kwargs: Any) -> str:
        # 同步调用远程 MCP 工具，并返回结果。
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(invoke_remote(**kwargs))
        raise RuntimeError(
            "MCP 同步工具不能直接运行在事件循环线程中，请使用 ainvoke。"
        )

    description = str(getattr(remote_tool, "description", "") or "").strip()
    description = f"[MCP: {server_name}/{remote_name}] {description}".strip()  # 添加 MCP 服务器和工具的名称信息
    schema = (
        getattr(remote_tool, "input_schema", None)
        or getattr(remote_tool, "inputSchema", None)
        or {
            "type": "object",
            "properties": {},
        }
    )  # 获取远程工具的输入参数 schema，如果没有则使用空对象 schema

    return StructuredTool.from_function(
        func=invoke_remote_sync,
        coroutine=invoke_remote,
        name=exposed_name,
        description=description,
        args_schema=schema,
        metadata={
            "capability": server_name,
            "aliases": [server_name, remote_name, *[str(x) for x in (record.get("aliases") or [])]],
            "kind": "mcp",
        },  # metadata的作用是提供工具的额外信息，包括能力、别名和类型等，便于在 FluxAgent 中进行管理和调用。
    )


def _is_declared_read_only(record: dict[str, Any], remote_tool: Any) -> bool:
    # 检查工具是否在 MCP 服务器的 read_only_tools 列表中声明为只读。
    declared = {str(name) for name in (record.get("read_only_tools") or [])}
    return str(remote_tool.name) in declared


async def load_installed_mcp_tools(
    index_path: Path | None = None,
    *,
    discovery_timeout: float = 15.0,
) -> MCPToolLoadResult:
    # 寻找已启用的注册服务器，而不会导致启动失败封闭。
    tools: list[BaseTool] = []
    risk_overrides: dict[str, RiskLevel] = {}
    descriptors: list[Any] = []
    errors: list[str] = []
    try:
        records = list_installed_mcp_servers(index_path or Path(MCP_SERVERS_INDEX))
    except InstallError as exc:
        return MCPToolLoadResult([], {}, [str(exc)], [])

    # 遍历所有已安装的 MCP 服务器记录，尝试发现工具并加载到 FluxAgent 中。
    used_names: set[str] = set()
    for record in records:
        if record.get("enabled", True) is False:
            continue
        server_name = str(record.get("name") or "unnamed")
        try:
            remote_tools = await asyncio.wait_for(
                _discover_server(record), timeout=discovery_timeout
            )
            server_tools_by_name: dict[str, BaseTool] = {}
            for remote_tool in remote_tools:
                exposed = _exposed_name(server_name, str(remote_tool.name))
                if exposed in used_names:
                    raise ValueError(f"工具名称冲突: {exposed}")  # 如果工具名称冲突，抛出异常以避免覆盖。
                used_names.add(exposed)
                server_tools_by_name[exposed] = _make_tool(record, remote_tool, exposed)
                risk_overrides[exposed] = (
                    RiskLevel.READ
                    if _is_declared_read_only(record, remote_tool)
                    else RiskLevel.WRITE
                )  # 根据 MCP 服务器的声明设置工具的风险级别。
            tools.extend(server_tools_by_name.values())
            descriptor = _make_descriptors(record, remote_tools, server_tools_by_name)
            if descriptor is not None:
                descriptors.append(descriptor)
            audit_logger.log_event(
                thread_id="system_default",
                event="system_action",
                action="mcp_client_connected",
                server=server_name,
                tools=[str(tool.name) for tool in remote_tools],
            )
        except Exception as exc:
            detail = _exception_detail(exc)
            message = f"{server_name}: {detail}"
            errors.append(message)
            audit_logger.log_event(
                thread_id="system_default",
                event="system_action",
                action="mcp_client_connect_failed",
                server=server_name,
                error=detail[:300],
            )

    return MCPToolLoadResult(tools, risk_overrides, errors, descriptors)
