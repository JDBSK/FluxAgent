"""可编程的假 LLM + 假工具。

为什么不用真模型跑评测：
1. 100+ case × 多轮 = 上千次调用，钱和时间都不划算
2. 真模型不确定，回归基线会天天飘，无法判断"是我改坏了还是模型抖了"
3. 要专门构造"工具连续失败"这类分支，真模型很难稳定复现

所以：图/状态/闸门这些确定性逻辑用 stub 测；模型质量单独用小样本 judge 抽测。
"""
from typing import Any, Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ScriptedLLM(BaseChatModel):
    """按脚本依次返回预设回复。

    script 里每项可以是：
      - str                      -> 纯文本回复
      - list[dict]               -> tool_calls，如 [{"name":"t","args":{}}]
      - callable(messages)->两者  -> 需要看上下文再决定时用
    脚本耗尽后返回兜底文本，避免测试因越界而抛不相关的错。
    """

    script: list[Any] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    cursor: int = 0
    fallback: str = "完成"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _next_item(self, messages: list[BaseMessage]) -> Any:
        if self.cursor >= len(self.script):
            return self.fallback
        item = self.script[self.cursor]
        self.cursor += 1
        if callable(item):
            return item(messages)
        return item

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        item = self._next_item(messages)

        if isinstance(item, list):
            tool_calls = [
                {
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                    "id": tc.get("id", f"call_{self.cursor}_{i}"),
                }
                for i, tc in enumerate(item)
            ]
            msg = AIMessage(content=item[0].get("content", ""), tool_calls=tool_calls)
        else:
            msg = AIMessage(content=str(item))

        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        # 真模型 bind_tools 会返回带 schema 的新对象；stub 直接复用自身，
        # 这样 planner 用的裸 llm 和 agent 用的 llm_with_tools 共享同一份脚本游标
        return self

    def reset(self):
        self.cursor = 0
        self.calls = []


class _Args(BaseModel):
    value: str = Field(default="", description="任意参数")


def make_tool(name: str, result: str = "ok", fail: bool = False) -> StructuredTool:
    """造一个假工具。fail=True 时返回 ❌ 开头，触发 progress_node 的失败判定。"""

    def _run(value: str = "") -> str:
        return f"❌ 执行失败：{name} transient" if fail else f"{result}:{value}"

    return StructuredTool.from_function(
        func=_run, name=name, description=f"测试工具 {name}", args_schema=_Args
    )


def counting_tool(name: str, counter: dict, result: str = "ok") -> StructuredTool:
    """记录被调次数的工具，用来验证"闸门拒绝后工具确实没跑"。"""

    def _run(value: str = "") -> str:
        counter[name] = counter.get(name, 0) + 1
        return f"{result}:{value}"

    return StructuredTool.from_function(
        func=_run, name=name, description=f"计数工具 {name}", args_schema=_Args
    )
