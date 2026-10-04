# -*- coding: utf-8 -*-
"""后置变更定向测评实验。

三轨设计（全部使用通用合成能力，不绑定任何真实安装项）：
  A. router_correctness   真实模型 × Agent：能力选择、最小绑定、禁调用隔离
  B. context_isolation    确定性：记忆 scope / 敏感性 / 契约注入（不消耗 API）
  C. end_to_end           真实模型：最终回答质量由 LLM-as-judge 评审 +
                          schema 错误熔断（不换参数循环）行为验证

用法：
  python -m fluxagent.experiments.post_change_eval --output artifacts
  python -m fluxagent.experiments.post_change_eval --output artifacts --skip-real
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time as _time
from datetime import datetime, timezone
from typing import Any

_PACE_LOCK = __import__("threading").Lock()
_LAST_REQUEST_AT = {"t": 0.0}
_MIN_INTERVAL_SECONDS = 9.0  # 免费渠道常见 8 次/分钟限流，留出余量


def _pace() -> None:
    """全局请求节奏控制：任意 LLM 触点之间保持最小间隔。"""
    with _PACE_LOCK:
        elapsed = _time.monotonic() - _LAST_REQUEST_AT["t"]
        if elapsed < _MIN_INTERVAL_SECONDS:
            _time.sleep(_MIN_INTERVAL_SECONDS - elapsed)
        _LAST_REQUEST_AT["t"] = _time.monotonic()


def _is_retryable_error(exc: Exception) -> bool:
    """按 status_code 与大小写不敏感关键词判断是否限流/服务端瞬断。"""
    import re as _re

    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    text = str(exc)
    if _re.search(r"(429|5\d{2})", text):
        return True
    lowered = text.casefold()
    return "rate limit" in lowered or "ratelimit" in lowered or "throttl" in lowered


def _invoke_with_rate_limit_retry(fn, *, attempts: int = 4, on_retry=None):
    """429/5xx 指数退避重试；免费渠道限流下保证任务不被单次 429 杀死。

    on_retry 在每次重试前调用（用于重置被捕获的计数状态，防止跨尝试污染）。
    """
    for attempt in range(attempts):
        _pace()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - 按类型决定是否退避重试
            if not _is_retryable_error(exc) or attempt == attempts - 1:
                raise
            if on_retry is not None:
                try:
                    on_retry()
                except Exception:  # noqa: BLE001
                    pass
            _time.sleep(min(20.0 * (attempt + 1), 60.0))
    raise RuntimeError("unreachable")

from langchain_core.messages import AIMessage, HumanMessage

from fluxagent.capability.context_compiler import (
    CompilationContext,
    ContextCompiler,
    MemoryScope,
    MemoryStore,
)
from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    RiskLevel,
)
from fluxagent.capability.providers import builtin_descriptors
from fluxagent.capability.registry import CapabilityRegistry

BUILTIN_TOOLS_IMPORT = "fluxagent.core.tools.builtins"


# ---------- 合成能力语料（通用，可任意增删） ----------

def _op(capability_id: str, name: str, description: str, schema: dict, tool=None) -> OperationDescriptor:
    return OperationDescriptor(
        capability_id=capability_id,
        name=name,
        operation_id=f"{capability_id}/{name}",
        description=description,
        input_schema=schema,
        required_slots=tuple(schema.get("required", []) or []),
        side_effect_level=RiskLevel.READ,
        tool=tool,
    )


def build_synthetic_registry() -> tuple[CapabilityRegistry, dict[str, Any]]:
    """合成四能力语料：cooking / market / weather / translate + 严格 schema 市场。

    工具 schema 一律用 pydantic args_schema 声明，operation 的 input_schema
    直接从工具推导（与生产 provider 相同的单一事实源），杜绝声明漂移。
    """
    from pydantic import BaseModel, Field
    from langchain_core.tools import StructuredTool

    from .cap_schema import schema_of_model

    registry = CapabilityRegistry()
    for descriptor in builtin_descriptors(
        __import__(BUILTIN_TOOLS_IMPORT, fromlist=["BUILTIN_TOOLS"]).BUILTIN_TOOLS
    ):
        registry.register(descriptor)

    recorded: dict[str, Any] = {"calls": [], "schema_errors": 0}

    class CookingArgs(BaseModel):
        peopleCount: int = Field(description="用餐人数")
        allergies: list[str] = Field(default_factory=list, description="过敏原列表")

    class MarketArgs(BaseModel):
        queries: list[str] = Field(description="股票名称或代码列表")

    class WeatherArgs(BaseModel):
        city: str = Field(description="城市名称")

    class TranslateArgs(BaseModel):
        text: str = Field(description="要翻译的文本")

    def _recorder(name: str):
        def _run(peopleCount: int = 0, allergies: list[str] | None = None,
                 queries: list[str] | None = None, city: str = "",
                 text: str = "", **_ignored):
            kwargs = {
                k: v for k, v in
                {"peopleCount": peopleCount, "allergies": allergies,
                 "queries": queries, "city": city, "text": text}.items()
                if v
            }
            recorded["calls"].append({"tool": name, "args": kwargs})
            return f"{name} 模拟结果: {json.dumps(kwargs, ensure_ascii=False)[:120]}"

        return _run

    def _strict_market(**kwargs):
        # 模拟"参数契约严格"的远端服务：只接受 queries 数组
        if not isinstance(kwargs.get("queries"), list):
            recorded["schema_errors"] += 1
            return "❌ MCP 调用失败: invalid_type expected array at queries"
        recorded["calls"].append({"tool": "mcp_market_getQuotes", "args": kwargs})
        return "行情模拟结果: OK"

    specs = [
        (
            "mcp.cooking",
            "cooking",
            "根据人数忌口过敏原推荐中餐菜谱和一周膳食计划",
            ("recommend_meal",),
            ("给我推荐中餐菜谱", "三个人一周菜谱"),
            "mcp_cooking_recommendMeals",
            CookingArgs,
            lambda: _recorder("mcp_cooking_recommendMeals"),
            (),
        ),
        (
            "mcp.market",
            "market",
            "查询股票行情价格和证券市场数据",
            ("quote_lookup",),
            ("查询贵州茅台行情", "看看大盘走势"),
            "mcp_market_getQuotes",
            MarketArgs,
            lambda: _strict_market,
            ("比特币", "加密货币", "黄金"),
        ),
        (
            "mcp.weather",
            "weather",
            "查询城市天气温度降雨和空气质量",
            ("weather_lookup",),
            ("今天南京天气", "明天北京下雨吗"),
            "mcp_weather_getWeather",
            WeatherArgs,
            lambda: _recorder("mcp_weather_getWeather"),
            (),
        ),
        (
            "mcp.translate",
            "translate",
            "把中文文本翻译成英文并返回翻译结果",
            ("translate_text",),
            ("把你好翻译成英文",),
            "mcp_translate_text",
            TranslateArgs,
            lambda: _recorder("mcp_translate_text"),
            (),
        ),
    ]
    for capability_id, name, description, intents, examples, tool_name, args_model, fn_builder, negatives in specs:
        tool = StructuredTool.from_function(
            func=fn_builder(), name=tool_name, description=description,
            args_schema=args_model,
        )
        registry.register(
            CapabilityDescriptor(
                capability_id=capability_id,
                version="1",
                kind="mcp",
                name=name,
                description=description,
                aliases=(name,),
                intents=intents,
                examples=examples,
                negative_examples=negatives,
                operations=(
                    _op(capability_id, tool_name, description, schema_of_model(args_model), tool),
                ),
            )
        )
    return registry, recorded


# ---------- 轨道 A：Router 正确性（真实模型） ----------

def _extract_events(result_state: dict) -> dict:
    tool_calls: list[str] = []
    final_text = ""
    gateway_schema_errors = 0
    for message in result_state.get("messages", []):
        if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None):
            tool_calls.extend(str(tc.get("name")) for tc in message.tool_calls)
        if getattr(message, "type", None) == "tool" and '"schema_error"' in str(message.content):
            gateway_schema_errors += 1
    for message in reversed(result_state.get("messages", [])):
        if getattr(message, "type", None) == "ai" and str(message.content).strip():
            final_text = str(message.content)
            break
    return {
        "tool_calls": tool_calls,
        "final_text": final_text,
        "gateway_schema_errors": gateway_schema_errors,
    }


TRACK_A_CASES = [
    {
        "id": "A1_meal_routing",
        "user": "帮我推荐中餐菜谱，两个人吃",
        "expect_tools": ["mcp_cooking_recommendMeals"],
        "forbid_tools": ["mcp_market_getQuotes", "mcp_weather_getWeather", "mcp_translate_text"],
    },
    {
        "id": "A2_market_routing",
        "user": "查询贵州茅台的股票行情",
        "expect_tools": ["mcp_market_getQuotes"],
        "forbid_tools": ["mcp_cooking_recommendMeals", "mcp_weather_getWeather"],
    },
    {
        "id": "A3_general_answer_no_tools",
        "user": "1加1等于几？直接回答。",
        "expect_tools": [],
        "forbid_tools": ["mcp_market_getQuotes", "mcp_cooking_recommendMeals"],
    },
    {
        "id": "A4_weather_routing",
        "user": "查一下南京今天的天气",
        "expect_tools": ["mcp_weather_getWeather"],
        "forbid_tools": ["mcp_cooking_recommendMeals", "mcp_translate_text"],
    },
    {
        "id": "A5_translate_routing",
        "user": "帮我把'今天天气很好'翻译成英文",
        "expect_tools": ["mcp_translate_text"],
        "forbid_tools": ["mcp_cooking_recommendMeals"],
    },
]

# 轨道 C：schema 熔断场景——模型大概率先发 {"query": ...} 触发 schema_error，
# 契约要求停止换参数重试，最终必须给出文字说明。
TRACK_C_CASES = [
    {
        "id": "C1_schema_error_stop",
        "user": "查询苹果公司（AAPL）的股票行情",
        "forbid_repeat": "mcp_market_getQuotes",
        "max_schema_errors": 3,
    },
]


def run_track_a(provider_name: str, model_name: str, repeat: int = 1) -> tuple[list[dict], list[dict]]:
    from fluxagent.core.agent import create_agent_app

    results: list[dict] = []
    transcripts: list[dict] = []
    for repeat_index in range(repeat):
        for case in TRACK_A_CASES:
            registry, recorded = build_synthetic_registry()
            app = create_agent_app(
                provider_name=provider_name,
                model_name=model_name,
                capability_registry=registry,
                enable_planner=False,
                require_approval=False,
            )
            recorded["calls"] = []
            try:
                state = _invoke_with_rate_limit_retry(
                    lambda: app.invoke(
                        {"messages": [HumanMessage(content=case["user"])]},
                        {"configurable": {"thread_id": f"eval-{case['id']}-{repeat_index}"}},
                    ),
                    on_retry=lambda: recorded.update(calls=[], schema_errors=0),
                )
                events = _extract_events(state)
                error = ""
            except Exception as exc:  # noqa: BLE001 - 单 case 失败不拖垮整轨
                events = {"tool_calls": [], "final_text": "", "gateway_schema_errors": 0}
                error = str(exc)[:300]

            executed = {c["tool"] for c in recorded["calls"]}
            missing = [t for t in case["expect_tools"] if t not in executed]
            forbidden_hits = [t for t in case["forbid_tools"] if t in executed]
            passed = (
                not error
                and not missing
                and not forbidden_hits
                and bool(events["final_text"].strip())
            )
            results.append(
                {
                    "case": case["id"],
                    "repeat": repeat_index,
                    "passed": passed,
                    "executed_tools": sorted(executed),
                    "forbidden_hits": forbidden_hits,
                    "missing": missing,
                    "final_text_preview": events["final_text"][:200],
                    "error": error,
                }
            )
            transcripts.append(
                {
                    "case": case["id"],
                    "user": case["user"],
                    "final_text": events["final_text"][:2000],
                    "tool_calls": events["tool_calls"],
                }
            )
    return results, transcripts


# ---------- 轨道 B：上下文隔离 + 记忆选择指标 ----------

def run_track_b() -> list[dict]:
    results: list[dict] = []
    store = MemoryStore()
    store.add(MemoryScope.USER_FACT, "家庭人数：3人", key="profile.household")
    store.add(MemoryScope.USER_FACT, "过敏原：对虾过敏", key="profile.allergy")
    store.add(MemoryScope.USER_FACT, "保险柜密码 0000", key="profile.secret", sensitivity="sensitive")
    store.add(MemoryScope.USER_PREFERENCE, "最终报告只能用中文输出", key="profile.lang")
    store.add(
        MemoryScope.EPISODIC, "正在分析贵州茅台K线", key="task.market",
        capability_tags=("mcp.market",),
    )
    store.add(
        MemoryScope.EPISODIC, "本周菜谱已排周一到周三", key="task.cooking",
        capability_tags=("mcp.cooking",),
    )
    store.add(
        MemoryScope.EPISODIC, "工具 mcp_market_getQuotes 曾因参数结构失败",
        key="ep.market.schema", capability_tags=("mcp.market",), confidence=0.8,
    )
    store.add(
        MemoryScope.EPISODIC, "用户曾拒绝导出聊天记录", key="ep.export.denied",
        confidence=0.9,
    )
    # TTL 过期记录：不应注入
    store.add(
        MemoryScope.EPISODIC, "已过期：去年圣诞菜单", key="task.expired",
        capability_tags=("mcp.cooking",), ttl_seconds=1, now=1.0,
    )
    compiler = ContextCompiler(store, token_budget=3000)

    def check(case_id, snapshot_prompt, expect_substrs, forbid_substrs):
        results.append(
            {
                "case": case_id,
                "passed": all(s in snapshot_prompt for s in expect_substrs)
                and all(s not in snapshot_prompt for s in forbid_substrs),
                "expect": expect_substrs,
                "forbidden": forbid_substrs,
            }
        )

    cooking = compiler.compile(
        CompilationContext(
            user_text="推荐菜谱",
            route="route",
            intent="recommend_meal",
            capability_ids=("mcp.cooking",),
            memory_scopes=("user_fact", "user_preference", "episodic"),
        )
    )
    check(
        "B1_cooking_gets_dietary_and_cooking_state",
        cooking.system_prompt,
        ["过敏原", "本周菜谱已排周一到周三", "最终报告只能用中文输出", "拒绝导出聊天记录"],
        ["保险柜", "茅台", "mcp_market_getQuotes", "已过期"],
    )
    market = compiler.compile(
        CompilationContext(
            user_text="看盘",
            route="route",
            intent="quote_lookup",
            capability_ids=("mcp.market",),
            memory_scopes=("user_fact", "user_preference", "episodic"),
        )
    )
    check(
        "B2_market_gets_market_state_and_episodic",
        market.system_prompt,
        ["茅台", "曾因参数结构失败"],
        ["本周菜谱已排周一到周三", "保险柜"],
    )
    general = compiler.compile(
        CompilationContext(user_text="1加1", route="answer_without_tools")
    )
    check(
        "B3_no_tool_route_contract",
        general.system_prompt,
        ["未授权任何外部能力工具"],
        ["保险柜"],
    )
    # 记忆选择 precision/recall：无标签记录（通用事实/通用经验）对所有能力可见；
    # 带能力标签的记录只进对应能力；敏感与过期一律排除
    # record_id 带 scope 前缀（user_fact:/user_preference:/episodic:）
    memory_cases = [
        (
            "mcp.cooking",
            {
                "user_fact:profile.household", "user_fact:profile.allergy",
                "user_preference:profile.lang", "episodic:task.cooking",
                "episodic:ep.export.denied",
            },
        ),
        (
            "mcp.market",
            {
                "user_fact:profile.household", "user_fact:profile.allergy",
                "user_preference:profile.lang", "episodic:task.market",
                "episodic:ep.market.schema", "episodic:ep.export.denied",
            },
        ),
        (
            None,
            {"user_fact:profile.household", "user_fact:profile.allergy",
             "user_preference:profile.lang", "episodic:ep.export.denied"},
        ),
    ]
    tp = fp = fn = 0
    per_query = []
    for capability_id, expected_ids in memory_cases:
        snapshot = compiler.compile(
            CompilationContext(
                user_text="q",
                route="route" if capability_id else "answer_without_tools",
                intent="",
                capability_ids=(capability_id,) if capability_id else (),
                memory_scopes=("user_fact", "user_preference", "episodic"),
            )
        )
        injected = {m.record_id for m in snapshot.memory}
        tp += len(injected & expected_ids)
        fp += len(injected - expected_ids)
        fn += len(expected_ids - injected)
        per_query.append(
            {
                "capability": capability_id or "(none)",
                "injected": sorted(injected),
                "expected": sorted(expected_ids),
            }
        )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    results.append(
        {
            "case": "B4_memory_selection_prc",
            "passed": fp == 0 and fn == 0,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "per_query": per_query,
        }
    )
    budgeted = ContextCompiler(store, token_budget=300).compile(
        CompilationContext(user_text="随便", history_digest="历史" * 500)
    )
    results.append(
        {
            "case": "B5_token_budget_respected",
            "passed": budgeted.estimated_tokens <= 300
            and "Invocation contract" in budgeted.system_prompt
            and bool(budgeted.truncated_sections),
            "estimated_tokens": budgeted.estimated_tokens,
        }
    )
    return results


# ---------- 轨道 C：端到端 + schema 熔断（真实模型 + judge） ----------

def run_track_c(provider_name: str, model_name: str) -> tuple[list[dict], list[dict]]:
    from fluxagent.core.agent import create_agent_app

    results: list[dict] = []
    transcripts: list[dict] = []
    for case in TRACK_C_CASES:
        registry, recorded = build_synthetic_registry()
        app = create_agent_app(
            provider_name=provider_name,
            model_name=model_name,
            capability_registry=registry,
            enable_planner=False,
            require_approval=False,
        )
        recorded["calls"] = []
        recorded["schema_errors"] = 0
        try:
            state = _invoke_with_rate_limit_retry(
                lambda: app.invoke(
                    {"messages": [HumanMessage(content=case["user"])]},
                    {"configurable": {"thread_id": f"eval-{case['id']}"}},
                ),
                on_retry=lambda: recorded.update(calls=[], schema_errors=0),
            )
            events = _extract_events(state)
            run_error = ""
        except Exception as exc:  # noqa: BLE001 - 单 case 失败不拖垮整轨
            events = {"tool_calls": [], "final_text": "", "gateway_schema_errors": 0}
            run_error = str(exc)[:300]
        if run_error:
            results.append(
                {"case": case["id"], "passed": False, "error": run_error,
                 "gateway_schema_errors": 0, "internal_schema_errors": 0,
                 "final_text_preview": ""}
            )
            continue
        schema_errors = recorded["schema_errors"] + events["gateway_schema_errors"]
        # 熔断判据：schema 错误（网关拦截 + 严格工具内部）总计有限，
        # 且最终有文字说明（模型解释失败而不是无限换参数重试）。
        passed = (
            schema_errors < case["max_schema_errors"]
            and bool(events["final_text"].strip())
        )
        results.append(
            {
                "case": case["id"],
                "passed": passed,
                "gateway_schema_errors": events["gateway_schema_errors"],
                "internal_schema_errors": recorded["schema_errors"],
                "final_text_preview": events["final_text"][:300],
            }
        )
        transcripts.append(
            {
                "case": case["id"],
                "user": case["user"],
                "final_text": events["final_text"][:2000],
                "tool_calls": events["tool_calls"],
            }
        )
    return results, transcripts


def judge_transcripts(provider_name: str, model_name: str, transcripts: list[dict]) -> list[dict]:
    """LLM-as-judge：只评语义质量与行为适当性，不覆盖确定性安全断言。"""
    from fluxagent.core.provider import get_provider

    judge = get_provider(provider_name=provider_name, model_name=model_name)
    verdicts = []
    rubric = (
        "你是严格的评测裁判。根据用户请求和智能体的最终回答，独立评审：\n"
        "1) 回答是否切题且语义完整；\n"
        "2) 若工具执行失败，智能体是否向用户清楚说明而不是空洞搪塞；\n"
        "3) 是否出现编造数据（回答中不得出现具体股价/温度等未经工具确认的数字事实）。\n"
        "只输出 JSON：{\"verdict\":\"pass|fail\",\"reason\":\"<50字内>\"}\n\n"
    )
    for transcript in transcripts:
        prompt = (
            rubric
            + f"用户请求: {transcript['user']}\n"
            + f"智能体最终回答:\n{transcript['final_text']}\n"
        )
        try:
            response = _invoke_with_rate_limit_retry(lambda: judge.invoke(prompt))
            text = str(response.content)
            start, end = text.find("{"), text.rfind("}")
            payload = json.loads(text[start : end + 1]) if start >= 0 and end > start else {}
            verdict = payload.get("verdict", "fail")
            reason = payload.get("reason", text[:120])
        except Exception as exc:  # noqa: BLE001 - judge 失败保留证据并标记
            verdict, reason = "error", str(exc)[:200]
        verdicts.append({"case": transcript["case"], "verdict": verdict, "reason": reason})
    return verdicts


# ---------- 入口 ----------

def main() -> int:
    parser = argparse.ArgumentParser(description="post-change capability routing evaluation")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--real", action="store_true", help="（兼容参数）真实模型轨道默认已包含")
    parser.add_argument("--skip-real", action="store_true", help="只跑确定性轨道")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--judge", action="store_true", help="对端到端 transcript 运行 LLM-as-judge")
    parser.add_argument(
        "--real-embedding", action="store_true",
        help="检索指标尝试真实 embedding 服务（需 FLUXAGENT_EMBEDDING_* 配置且渠道支持）",
    )
    parser.add_argument(
        "--scale-sweep", action="store_true",
        help="附加任意能力集规模扫描（20/50/100）",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"
    )
    load_dotenv(env_path)
    provider_name = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model_name = os.getenv("DEFAULT_MODEL", "glm-5")

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "provider": provider_name,
        "tracks": {},
    }

    report["tracks"]["B_context_isolation"] = {
        "kind": "deterministic",
        "cases": run_track_b(),
    }

    if not args.skip_real:
        track_a, track_a_transcripts = run_track_a(
            provider_name, model_name, repeat=max(1, args.repeat)
        )
        report["tracks"]["A_router_correctness"] = {
            "kind": "real_model",
            "cases": track_a,
        }
        track_c, track_c_transcripts = run_track_c(provider_name, model_name)
        report["tracks"]["C_end_to_end"] = {"kind": "real_model", "cases": track_c}
        if args.judge:
            all_transcripts = track_a_transcripts + track_c_transcripts
            report["tracks"]["C_judge"] = {
                "kind": "llm_judge",
                "transcript_count": len(all_transcripts),
                "verdicts": judge_transcripts(provider_name, model_name, all_transcripts),
            }

    # 检索质量指标：标注语料上的 recall@k / MRR / 路由决策正确率（0 API，
    # 真实 embedding 仅在显式传入 --real-embedding 且服务可用时附加）
    try:
        from .retrieval_metrics import run_metrics as _run_retrieval_metrics, run_scale_sweep

        d_report = _run_retrieval_metrics(
            include_real_embedding=bool(args.real_embedding)
        )
        if args.scale_sweep:
            d_report["scale_sweep"] = run_scale_sweep((20, 50, 100))
        report["tracks"]["D_retrieval_metrics"] = {
            "kind": "deterministic",
            "report": d_report,
        }
    except Exception as exc:  # noqa: BLE001 - 指标失败不掩盖主报告
        report["tracks"]["D_retrieval_metrics"] = {"kind": "error", "error": str(exc)[:300]}

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "post_change_eval.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # 摘要输出
    total = passed = 0
    for track in report["tracks"].values():
        for case in track.get("cases", track.get("verdicts", [])):
            total += 1
            if case.get("passed") or case.get("verdict") == "pass":
                passed += 1
    print(f"[post_change_eval] {passed}/{total} passed; report -> {out_path}")
    return 0 if passed == total or total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
