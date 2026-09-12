"""工具注册表：协议层与执行层之间的唯一接缝。

为什么单独抽一层，不让 server.py 直接调 LangChain 工具：
1. 可测：注册表是纯 Python，不需要起 stdio 进程就能验证 list/call 语义
2. 可换协议：以后要加 HTTP/gRPC 入口，复用同一个注册表即可
3. 危险工具标记只做一次，两个入口（Agent 内 / MCP 外）共享同一份判断

安全立场（面试会问）：MCP 客户端是外部进程，Agent 内的审批闸门管不到它。
所以这里默认拒绝副作用工具（allow_side_effects=False），要放开必须显式传参。
沙盒层（路径前缀比对 + cwd 限定）仍然生效，这是最后一道硬约束。
"""
from typing import Any, Optional

from langchain_core.tools import BaseTool

from ..core.policy import is_side_effect_tool
from ..core.logger import audit_logger
from ..core.skill_loader import load_dynamic_skills
from ..core.tools.builtins import BUILTIN_TOOLS


class ToolRegistry:
    """把 LangChain 工具映射成 MCP 可用的 name/schema/调用。"""

    def __init__(
        self,
        tools: Optional[list[BaseTool]] = None,
        allow_side_effects: bool = False,
        thread_id: str = "mcp_server",
        extra_side_effect_tools: Optional[set[str]] = None,
    ):
        self._tools: dict[str, BaseTool] = {t.name: t for t in (tools or [])}
        self.allow_side_effects = allow_side_effects
        self.thread_id = thread_id
        self._extra_side_effect_tools = set(extra_side_effect_tools or set())

    def names(self) -> list[str]:
        """返回按名称排序的已注册工具列表。"""
        return sorted(self._tools)

    def is_blocked(self, name: str) -> bool:
        """副作用工具在未放开时被拒。外部调用方绕不过 Agent 的审批闸门，
        所以这里用白名单思路兜底，而不是信任调用方。"""
        return (not self.allow_side_effects) and (
            is_side_effect_tool(name) or name in self._extra_side_effect_tools
        )

    def describe(self) -> list[dict[str, Any]]:
        """list_tools 的数据源。schema 直接取工具自带的 args_schema，
        不手写一份——手写必然和实现漂移。"""
        out = []
        for name in self.names():
            tool = self._tools[name]
            schema: dict[str, Any] = {"type": "object", "properties": {}}
            try:
                if tool.args_schema is not None:
                    raw = tool.args_schema.model_json_schema()  # 生成 JSON Schema
                    schema = {
                        "type": "object",
                        "properties": raw.get("properties", {}),
                        "required": raw.get("required", []),
                    }
            except Exception as exc:
                audit_logger.log_event(
                    thread_id=self.thread_id,
                    event="system_action",
                    action="mcp_schema_failed",
                    tool=name,
                    error=str(exc)[:300],
                )
                # Do not advertise a schema that cannot validate the real
                # function signature. Other healthy tools remain available.
                continue

            desc = (tool.description or "").strip()
            if self.is_blocked(name):
                desc = f"[需审批·当前不可通过 MCP 调用] {desc}"
            out.append({"name": name, "description": desc[:1000], "inputSchema": schema})
        return out

    def call(self, name: str, arguments: Optional[dict[str, Any]] = None) -> tuple[str, bool]:
        """执行工具，返回 (文本结果, is_error)。

        错误不抛异常而是当结果返回——MCP 客户端通常也是个 LLM，
        让它读到错误文本自己纠正，比连接断开更有用。
        """
        args = arguments or {}
        if name not in self._tools:
            return f"未知工具: {name}。可用工具: {', '.join(self.names())}", True

        if self.is_blocked(name):
            audit_logger.log_event(
                thread_id=self.thread_id, event="system_action",
                action="mcp_call_blocked", tool=name,
            )
            return (
                f"拒绝执行: {name} 属于副作用工具，需要人工审批，"
                f"不允许通过 MCP 外部调用。",
                True,
            )

        audit_logger.log_event(
            thread_id=self.thread_id, event="tool_call", tool=name, args=args,
        )
        try:
            result = self._tools[name].invoke(args)
            text = result if isinstance(result, str) else str(result)
            audit_logger.log_event(
                thread_id=self.thread_id, event="tool_result",
                tool=name, result_summary=text[:200],
            )
            return text, False
        except Exception as exc:
            audit_logger.log_event(
                thread_id=self.thread_id, event="tool_result",
                tool=name, result_summary=f"ERROR: {exc}"[:200],
            )
            return f"工具执行异常: {exc}", True


def build_registry(
    allow_side_effects: bool = False,
    include_skills: bool = True,
    tools: Optional[list[BaseTool]] = None,
) -> ToolRegistry:
    """默认把内置工具 + 动态技能一起暴露。

    技能走的还是 skill_loader 的懒加载，MCP 只是多一个入口，
    不复制一份加载逻辑——热更新能力自动继承。
    """
    if tools is None:
        tools = list(BUILTIN_TOOLS)
        dynamic_skill_names: set[str] = set()
        if include_skills:
            try:
                dynamic_skills = load_dynamic_skills()
                tools += dynamic_skills
                dynamic_skill_names = {tool.name for tool in dynamic_skills}
            except Exception as exc:
                audit_logger.log_event(
                    thread_id="mcp_server",
                    event="system_action",
                    action="dynamic_skills_load_failed",
                    error=str(exc)[:300],
                )
        return ToolRegistry(
            tools=tools,
            allow_side_effects=allow_side_effects,
            extra_side_effect_tools=dynamic_skill_names,
        )
    return ToolRegistry(tools=tools, allow_side_effects=allow_side_effects)
