import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fluxagent.core.policy import RiskLevel
from fluxagent.mcp.client import _exception_detail, load_installed_mcp_tools


class TestMCPClient(unittest.IsolatedAsyncioTestCase):
    def test_renders_nested_task_group_cause(self):
        # 测试 _exception_detail 函数是否正确渲染嵌套的 TaskGroup 异常原因。
        error = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [RuntimeError("No module named server")],
        )

        self.assertEqual(_exception_detail(error), "No module named server")

    @patch("fluxagent.mcp.client._call_server_tool", new_callable=AsyncMock)
    @patch("fluxagent.mcp.client._discover_server", new_callable=AsyncMock)
    @patch("fluxagent.mcp.client.list_installed_mcp_servers")
    async def test_loads_namespaced_tool_and_calls_remote(
        self, load_registry, discover, call_remote
    ):
        record = {
            "name": "weather-mcp",
            "command": "node",
            "args": ["dist/index.js", "--mcp"],
            "installed_path": str(Path.cwd()),
            "read_only_tools": ["getWeather"],
        }
        load_registry.return_value = [record]
        discover.return_value = [
            SimpleNamespace(
                name="getWeather",
                description="查询天气",
                input_schema={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                annotations=None,
            )
        ]
        call_remote.return_value = "南京：晴"

        result = await load_installed_mcp_tools()

        self.assertEqual([tool.name for tool in result.tools], ["mcp_weather-mcp_getWeather"])
        self.assertEqual(result.risk_overrides[result.tools[0].name], RiskLevel.READ)
        self.assertEqual(await result.tools[0].ainvoke({"city": "南京"}), "南京：晴")
        self.assertEqual(
            await asyncio.to_thread(result.tools[0].invoke, {"city": "南京"}),
            "南京：晴",
        )
        self.assertEqual(call_remote.await_count, 2)
        call_remote.assert_awaited_with(record, "getWeather", {"city": "南京"})

    @patch("fluxagent.mcp.client._discover_server", new_callable=AsyncMock)
    @patch("fluxagent.mcp.client.list_installed_mcp_servers")
    async def test_broken_server_does_not_block_other_startup(self, load_registry, discover):
        load_registry.return_value = [{"name": "broken", "enabled": True}]
        discover.side_effect = ValueError("缺少启动 command")

        result = await load_installed_mcp_tools()

        self.assertEqual(result.tools, [])
        self.assertIn("broken", result.errors[0])

    @patch("fluxagent.mcp.client._discover_server", new_callable=AsyncMock)
    @patch("fluxagent.mcp.client.list_installed_mcp_servers")
    async def test_reports_nested_connection_cause(self, load_registry, discover):
        load_registry.return_value = [{"name": "stock-sdk-mcp", "enabled": True}]
        discover.side_effect = ExceptionGroup(
            "unhandled errors in a TaskGroup",
            [RuntimeError("No module named server")],
        )

        result = await load_installed_mcp_tools()

        self.assertEqual(result.errors, ["stock-sdk-mcp: No module named server"])


if __name__ == "__main__":
    unittest.main()
