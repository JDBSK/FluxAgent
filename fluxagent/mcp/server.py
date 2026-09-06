"""MCP stdio server。

跑法：
    conda activate cyber
    python -m fluxagent.mcp.server                  # 只暴露安全工具
    python -m fluxagent.mcp.server --allow-side-effects   # 放开副作用工具（自负风险）

设计取舍：
- 用 stdio 而不是 HTTP：MCP 客户端（Claude Desktop / IDE）默认拉子进程，
  stdio 没有端口、没有鉴权面，暴露面最小。要 HTTP 再加，注册表可复用。
- handler 里不做业务判断，全部委托 ToolRegistry，保证"Agent 内调用"和
  "MCP 外部调用"走同一套安全判断。
"""
import argparse
import asyncio
import sys

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .registry import build_registry


def build_server(allow_side_effects: bool = False, include_skills: bool = True) -> Server:
    """构建绑定工具注册表的 MCP 服务实例。"""
    registry = build_registry(
        allow_side_effects=allow_side_effects, include_skills=include_skills
    )

    async def on_list_tools(ctx, params) -> types.ListToolsResult:
        """返回当前注册表中的 MCP 工具描述。"""
        return types.ListToolsResult(
            tools=[types.Tool(**spec) for spec in registry.describe()]
        )

    async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        """在线程池中执行指定工具并包装 MCP 调用结果。"""
        # 同步工具放线程池：沙盒 shell 有 60s 超时，直接 await 会卡住事件循环
        text, is_error = await asyncio.to_thread(
            registry.call, params.name, params.arguments
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)], isError=is_error
        )

    return Server(
        name="fluxagent",
        version="1.0.0",
        instructions=(
            "FluxAgent 工具服务。所有文件与 shell 操作被限制在 office 沙盒内。"
            "带 [需审批] 标记的工具默认不可调用。"
        ),
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )


async def run(allow_side_effects: bool = False) -> None:
    """通过标准输入输出运行 MCP 服务。"""
    server = build_server(allow_side_effects=allow_side_effects)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


def main() -> None:
    """解析命令行参数并启动 MCP 服务。"""
    parser = argparse.ArgumentParser(description="FluxAgent MCP server (stdio)")
    parser.add_argument(
        "--allow-side-effects",
        action="store_true",
        help="放开写文件/执行 shell 等副作用工具（默认拒绝）",
    )
    args = parser.parse_args()
    # stdio 传输里 stdout 是协议通道，任何 print 都会污染帧，日志必须走 stderr
    print(
        f"[MCP] fluxagent server starting (side_effects={args.allow_side_effects})",
        file=sys.stderr,
    )
    asyncio.run(run(allow_side_effects=args.allow_side_effects))


if __name__ == "__main__":
    main()
