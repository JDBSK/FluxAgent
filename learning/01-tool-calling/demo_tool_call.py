# -*- coding: utf-8 -*-
"""第 1 课 Demo：不依赖大模型，先看清工具调用的完整流程。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    function: Callable[..., str]


def get_current_time(attempt: int = 0) -> str:
    return datetime.now().strftime("当前时间：%Y-%m-%d %H:%M:%S")


def calculator(expression: str, attempt: int = 0) -> str:
    if any(word in expression for word in ("import", "open", "__")):
        return "策略拒绝：计算器只允许算术表达式"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return "参数错误：不是合法算术表达式"
    return f"计算结果是：{result}"


def fetch_report(path: str, attempt: int = 0) -> str:
    if path == "retry-demo.txt" and attempt == 0:
        return "Error: 网络暂时不可用"
    if not path.startswith("workspace/"):
        return "越权拦截：只能读取 workspace/ 内的文件"
    return f"读取成功：{path}（第 {attempt + 1} 次尝试成功）"


TOOLS = {
    "get_current_time": Tool(
        name="get_current_time",
        description="用户询问现在时间、日期、星期几时调用。无需参数。",
        schema={"type": "object", "properties": {}, "required": []},
        function=get_current_time,
    ),
    "calculator": Tool(
        name="calculator",
        description="用户要求计算加减乘除表达式时调用。传入 expression 字符串。",
        schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        function=calculator,
    ),
    "fetch_report": Tool(
        name="fetch_report",
        description="用户要求读取报告文件时调用。只能读取 workspace/ 内的文件。",
        schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        function=fetch_report,
    ),
}


def validate_args(tool: Tool, args: dict[str, Any]) -> str | None:
    for field in tool.schema.get("required", []):
        if field not in args:
            return f"参数错误：缺少 {field}"
    return None


def call_tool(name: str, args: dict[str, Any], retry_count: int = 0) -> str:
    tool = TOOLS.get(name)
    if tool is None:
        return "fatal：工具不存在，应终止或重新规划"

    error = validate_args(tool, args)
    if error:
        return error

    result = tool.function(**args, attempt=retry_count)

    if result.startswith("Error:"):
        if retry_count < 2:
            print(f"    观察到临时失败，第 {retry_count + 1} 次重试")
            return call_tool(name, args, retry_count + 1)
        return "exhausted：重试用尽，应重新规划"

    if result.startswith("越权拦截"):
        return result + "；策略失败，禁止盲目重试，应终止或换方案"

    return result


def show_case(title: str, user_request: str, tool_name: str, args: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print(f"用户任务：{user_request}")
    print(f"模型决策：调用 {tool_name}")
    print(f"生成参数：{args}")
    print(f"执行结果：{call_tool(tool_name, args)}")


if __name__ == "__main__":
    print("第 1 课：工具调用 Demo")
    print("=" * 48)

    show_case("案例 1：无参数工具", "现在几点了？", "get_current_time", {})
    show_case("案例 2：带参数工具", "(128+64)/4 等于多少？", "calculator", {"expression": "(128+64)/4"})
    show_case("案例 3：临时失败后重试", "读取 workspace/retry-demo.txt", "fetch_report", {"path": "workspace/retry-demo.txt"})
    show_case("案例 4：策略失败不重试", "读取 C:/Windows/system.ini", "fetch_report", {"path": "C:/Windows/system.ini"})

    print("\n思考题：")
    print("1. 为什么案例 3 可以重试，案例 4 不可以？")
    print("2. 如果模型选错工具，应该在工具层修，还是在提示词/工具描述层修？")


