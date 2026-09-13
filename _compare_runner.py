# -*- coding: utf-8 -*-
"""修改前/后对比运行器：独立于 capability 包，可在基线树与当前树上运行。

用法（由编排方设置 PYTHONPATH 指向目标代码树）：
  set PYTHONPATH=G:\\FluxAgent-v2\\.baseline-wt
  python _compare_runner.py --mode old --output artifacts\\compare_old.json

  set PYTHONPATH=G:\\FluxAgent-v2
  python _compare_runner.py --mode new --output artifacts\\compare_new.json

old 模式：旧生产配置（builtin + 全部外部工具整表绑定，旧 fallback 路由）。
new 模式：新生产配置（统一 Registry + Hybrid Router + Context Compiler）。
两侧使用同一批合成能力、同一批用例、同一个真实模型；指标全部确定性。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

MIN_INTERVAL = 9.0  # 免费渠道 8 次/分钟限流余量
_last = {"t": 0.0}


def _pace() -> None:
    elapsed = time.monotonic() - _last["t"]
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last["t"] = time.monotonic()


def _invoke_retry(fn, attempts: int = 4):
    for attempt in range(attempts):
        _pace()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            import re as _re

            text = str(exc)
            status = getattr(exc, "status_code", None)
            retryable = (
                (isinstance(status, int) and (status == 429 or 500 <= status < 600))
                or bool(_re.search(r"\b(429|5\d{2})\b", text))
                or "rate limit" in text.casefold()
                or "ratelimit" in text.casefold()
            )
            if not retryable or attempt == attempts - 1:
                raise
            time.sleep(min(20.0 * (attempt + 1), 60.0))
    raise RuntimeError("unreachable")


# ---------- 合成能力（与 post_change_eval 同一套，plain 工具形态） ----------

class CookingArgs(BaseModel):
    peopleCount: int = Field(description="用餐人数")
    allergies: list[str] = Field(default_factory=list, description="过敏原列表")


class MarketArgs(BaseModel):
    queries: list[str] = Field(description="股票名称或代码列表")


class WeatherArgs(BaseModel):
    city: str = Field(description="城市名称")


class TranslateArgs(BaseModel):
    text: str = Field(description="要翻译的文本")


RECORDED: list[str] = []


def _recorder(name: str, strict: bool = False):
    def _run(peopleCount: int = 0, allergies: list | None = None,
             queries: list | None = None, city: str = "", text: str = "", **_kw):
        if strict:
            if not isinstance(queries, list):
                return f"❌ MCP 调用失败: invalid_type expected array at queries"
        RECORDED.append(name)
        return f"{name} 模拟结果: OK"

    return _run


TOOLS_SPEC = [
    ("mcp_cooking_recommendMeals", "根据人数忌口过敏原推荐中餐菜谱和一周膳食计划", CookingArgs, _recorder("mcp_cooking_recommendMeals")),
    ("mcp_market_getQuotes", "查询股票实时行情价格涨跌幅和证券市场数据", MarketArgs, _recorder("mcp_market_getQuotes", strict=True)),
    ("mcp_weather_getWeather", "查询城市当前天气温度降雨和空气质量", WeatherArgs, _recorder("mcp_weather_getWeather")),
    ("mcp_translate_text", "把中文文本翻译成英文并返回翻译结果", TranslateArgs, _recorder("mcp_translate_text")),
]

# (用例, 期望执行的外部工具集合, 禁止执行的外部工具集合)
CASES = [
    ("帮我推荐中餐菜谱，两个人吃", {"mcp_cooking_recommendMeals"}, {"mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"}),
    ("查询贵州茅台的股票行情", {"mcp_market_getQuotes"}, {"mcp_cooking_recommendMeals", "mcp_weather_getWeather"}),
    ("1加1等于几？直接回答。", set(), {"mcp_cooking_recommendMeals", "mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"}),
    ("你觉得人生的意义是什么", set(), {"mcp_cooking_recommendMeals", "mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"}),
    ("查一下南京今天的天气", {"mcp_weather_getWeather"}, {"mcp_cooking_recommendMeals", "mcp_market_getQuotes"}),
    ("把'今天天气很好'翻译成英文", {"mcp_translate_text"}, {"mcp_cooking_recommendMeals", "mcp_market_getQuotes"}),
]


def build_plain_tools() -> list:
    tools = []
    for name, desc, args_model, fn in TOOLS_SPEC:
        tools.append(
            StructuredTool.from_function(func=fn, name=name, description=desc, args_schema=args_model)
        )
    return tools


def build_capability_registry():
    from fluxagent.capability.descriptors import CapabilityDescriptor, OperationDescriptor, RiskLevel
    from fluxagent.capability.providers import builtin_descriptors
    from fluxagent.capability.registry import CapabilityRegistry
    from fluxagent.core.tools.builtins import BUILTIN_TOOLS

    registry = CapabilityRegistry()
    for descriptor in builtin_descriptors(BUILTIN_TOOLS):
        registry.register(descriptor)
    for name, desc, args_model, _fn in TOOLS_SPEC:
        tool = next(t for t in build_plain_tools() if t.name == name)
        capability_id = f"mcp.{name.split('_')[1]}"
        schema = args_model.model_json_schema()
        schema.pop("title", None)
        for props in schema.get("properties", {}).values():
            props.pop("title", None)
        negatives = ()
        if capability_id == "mcp.market":
            # 能力边界声明：股票行情能力不覆盖加密货币与贵金属
            negatives = ("比特币", "加密货币", "黄金", "白银")
        registry.register(
            CapabilityDescriptor(
                capability_id=capability_id,
                version="1",
                kind="mcp",
                name=capability_id.split(".", 1)[1],
                description=desc,
                aliases=(name.split("_")[1],),
                intents=(),
                examples=(),
                negative_examples=negatives,
                operations=(
                    OperationDescriptor(
                        capability_id=capability_id,
                        name=name,
                        operation_id=f"{capability_id}/{name}",
                        description=desc,
                        input_schema=schema,
                        required_slots=tuple(schema.get("required", [])),
                        side_effect_level=RiskLevel.READ,
                        tool=tool,
                    ),
                ),
            )
        )
    return registry


def run(mode: str) -> dict:
    from fluxagent.core.agent import create_agent_app
    from fluxagent.core.provider import get_provider

    provider_name = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model_name = os.getenv("DEFAULT_MODEL", "glm-5")

    if mode == "new":
        registry = build_capability_registry()
        app = create_agent_app(
            provider_name=provider_name,
            model_name=model_name,
            capability_registry=registry,
            enable_planner=False,
            require_approval=False,
        )
    else:
        tools = build_plain_tools()
        from fluxagent.core.tools.builtins import BUILTIN_TOOLS
        app = create_agent_app(
            provider_name=provider_name,
            model_name=model_name,
            tools=list(BUILTIN_TOOLS) + tools,
            enable_planner=False,
            require_approval=False,
        )

    from langchain_core.messages import HumanMessage

    case_results = []
    for user, expect, forbid in CASES:
        RECORDED.clear()
        error = ""
        final_text = ""
        try:
            state = _invoke_retry(
                lambda: app.invoke(
                    {"messages": [HumanMessage(content=user)]},
                    {"configurable": {"thread_id": f"compare-{mode}-{abs(hash(user)) % 99999}"}},
                )
            )
            for message in reversed(state.get("messages", [])):
                if getattr(message, "type", None) == "ai" and str(message.content).strip():
                    final_text = str(message.content)
                    break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {str(exc)[:200]}"

        executed = set(RECORDED)
        correct = executed == expect
        wrong_hits = sorted(executed & forbid)
        missing = sorted(expect - executed)
        case_results.append(
            {
                "user": user,
                "executed": sorted(executed),
                "expected": sorted(expect),
                "wrong_tool_hits": wrong_hits,
                "missing": missing,
                "answer_present": bool(final_text.strip()),
                "error": error,
            }
        )

    total = len(case_results)
    fully_correct = sum(1 for c in case_results if c["executed"] == c["expected"] and not c["error"] and c["answer_present"])
    wrong_tool_cases = sum(1 for c in case_results if c["wrong_tool_hits"])
    return {
        "mode": mode,
        "model": model_name,
        "provider": provider_name,
        "cases": case_results,
        "aggregate": {
            "fully_correct": fully_correct,
            "total": total,
            "routing_accuracy": round(fully_correct / total, 4),
            "cases_with_wrong_tool_calls": wrong_tool_cases,
        },
    }


# 多轮混合域会话（扩容版）：2 个会话 × 6 轮，覆盖股票/菜谱/情感/天气/翻译
# 交叉与污染探针。会话 1 复现原始事故模式；会话 2 加入跨域切换与恢复探针。
MULTI_SESSIONS = [
    {
        "session": "s1_incident_pattern",
        "turns": [
            ("查询贵州茅台的股票行情", {"mcp_market_getQuotes"}, {"mcp_cooking_recommendMeals"}),
            ("帮我推荐中餐菜谱，两个人吃", {"mcp_cooking_recommendMeals"}, {"mcp_market_getQuotes"}),
            ("我喜欢一个女孩，她对我时好时坏，不知道她喜不喜欢我，帮我分析一下", set(), {"mcp_cooking_recommendMeals", "mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"}),
            ("回到刚才的话题，帮我推荐明天的菜谱", {"mcp_cooking_recommendMeals"}, {"mcp_market_getQuotes"}),
            ("帮我看看五粮液今天多少钱", {"mcp_market_getQuotes"}, {"mcp_cooking_recommendMeals"}),
            ("替我给女朋友写一句晚安短信翻译成英文", {"mcp_translate_text"}, {"mcp_market_getQuotes", "mcp_cooking_recommendMeals"}),
        ],
    },
    {
        "session": "s2_cross_domain_switch",
        "turns": [
            ("今天南京天气怎么样", {"mcp_weather_getWeather"}, {"mcp_market_getQuotes"}),
            ("用一句话安慰我，项目上线失败了", set(), {"mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"}),
            ("明天北京会下雨吗", {"mcp_weather_getWeather"}, {"mcp_cooking_recommendMeals"}),
            ("推荐三个适合下雨天的家常菜", {"mcp_cooking_recommendMeals"}, {"mcp_market_getQuotes", "mcp_weather_getWeather"}),
            ("股票被套了怎么办", set(), {"mcp_market_getQuotes", "mcp_weather_getWeather"}),
            ("把'恭喜发财'翻译成英文", {"mcp_translate_text"}, {"mcp_cooking_recommendMeals"}),
        ],
    },
    {
        # 能力缺失场景：语料中没有订票/黄金/加密货币能力。
        # 正确行为：不调用任何无关工具，直接回答或澄清。
        "session": "s3_missing_capability",
        "turns": [
            ("帮我订明天去上海的高铁票", set(), {"mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_cooking_recommendMeals", "mcp_translate_text"}),
            ("查一下现在黄金多少钱一克", set(), {"mcp_market_getQuotes", "mcp_weather_getWeather"}),
            ("使用订票服务MCP帮我订一张去北京的车票", set(), {"mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_cooking_recommendMeals", "mcp_translate_text"}),
            ("帮我看看比特币现在的价格", set(), {"mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"}),
        ],
    },
]


def run_multi(mode: str) -> dict:
    from fluxagent.core.agent import create_agent_app
    from fluxagent.core.tools.builtins import BUILTIN_TOOLS
    from langchain_core.messages import HumanMessage

    provider_name = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model_name = os.getenv("DEFAULT_MODEL", "glm-5")

    if mode == "new":
        registry = build_capability_registry()
        app = create_agent_app(
            provider_name=provider_name,
            model_name=model_name,
            capability_registry=registry,
            enable_planner=False,
            require_approval=False,
        )
    else:
        tools = build_plain_tools()
        app = create_agent_app(
            provider_name=provider_name,
            model_name=model_name,
            tools=list(BUILTIN_TOOLS) + tools,
            enable_planner=False,
            require_approval=False,
        )

    all_turns = []
    for session in MULTI_SESSIONS:
        config = {"configurable": {"thread_id": f"compare-multi-{mode}-{session['session']}"}}
        for index, (user, expect, forbid) in enumerate(session["turns"], start=1):
            RECORDED.clear()
            error = ""
            final_text = ""
            try:
                state = _invoke_retry(
                    lambda: app.invoke({"messages": [HumanMessage(content=user)]}, config)
                )
                for message in reversed(state.get("messages", [])):
                    if getattr(message, "type", None) == "ai" and str(message.content).strip():
                        final_text = str(message.content)
                        break
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {str(exc)[:200]}"

            executed = set(RECORDED)
            all_turns.append(
                {
                    "session": session["session"],
                    "turn": index,
                    "user": user,
                    "executed": sorted(executed),
                    "expected": sorted(expect),
                    "wrong_tool_hits": sorted(executed & forbid),
                    "answer_present": bool(final_text.strip()),
                    "error": error,
                }
            )

    correct_turns = sum(
        1 for t in all_turns if t["executed"] == t["expected"] and not t["error"] and t["answer_present"]
    )
    return {
        "mode": mode,
        "suite": "multi_turn_mixed_domain_scaled",
        "model": model_name,
        "turns": all_turns,
        "aggregate": {
            "correct_turns": correct_turns,
            "total_turns": len(all_turns),
            "contamination_turns": sum(1 for t in all_turns if t["wrong_tool_hits"]),
            "error_turns": sum(1 for t in all_turns if t["error"]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["old", "new"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--suite", choices=["single", "multi"], default="single")
    args = parser.parse_args()
    report = run_multi(args.mode) if args.suite == "multi" else run(args.mode)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    agg = report["aggregate"]
    print(f"[compare:{args.mode}:{args.suite}] {json.dumps(agg, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
