"""Paid, real-model engineering evaluation with safe deterministic tools.

This suite complements ScriptedLLM regression tests. The model makes every
decision, while scoring uses exact tool names, schemas, arguments, plan
constraints, and ToolMessage IDs instead of an LLM judge.

Run from the project root:
    python -m tests.eval.live_engineering_metrics --output logs/live_engineering_metrics.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from fluxagent.core.agent import _build_planner_prompt, _parse_plan  # noqa: E402
from fluxagent.core.provider import get_provider  # noqa: E402


class CityArgs(BaseModel):
    city: str = Field(description="Exact city name from the request")


class RateArgs(BaseModel):
    base: str = Field(description="Exact source currency code")
    quote: str = Field(description="Exact target currency code")


class OrderArgs(BaseModel):
    order_id: str = Field(description="Exact order ID")


class InventoryArgs(BaseModel):
    sku: str = Field(description="Exact SKU")


class ExpressionArgs(BaseModel):
    expression: str = Field(description="Exact arithmetic expression")


class TimeArgs(BaseModel):
    timezone: str = Field(description="Exact IANA timezone")


class LogArgs(BaseModel):
    service: str = Field(description="Exact service name")
    keyword: str = Field(description="Exact log keyword")


class ServiceArgs(BaseModel):
    service: str = Field(description="Exact service name")


class UserArgs(BaseModel):
    user_id: str = Field(description="Exact user ID")


class DeliveryArgs(BaseModel):
    tracking_id: str = Field(description="Exact tracking ID")


TOOL_DEFINITIONS: tuple[tuple[str, str, type[BaseModel]], ...] = (
    ("lookup_weather", "Read-only weather lookup for a city.", CityArgs),
    ("lookup_exchange_rate", "Read-only currency exchange-rate lookup.", RateArgs),
    ("lookup_order", "Read-only commerce order status lookup.", OrderArgs),
    ("lookup_inventory", "Read-only warehouse stock lookup by SKU.", InventoryArgs),
    ("calculate_expression", "Evaluate one arithmetic expression.", ExpressionArgs),
    ("lookup_time", "Read-only current time lookup for an IANA timezone.", TimeArgs),
    ("search_logs", "Read-only service log search by service and keyword.", LogArgs),
    ("get_service_status", "Read-only runtime health lookup for one service.", ServiceArgs),
    ("query_user_profile", "Read-only user profile lookup by user ID.", UserArgs),
    ("lookup_delivery", "Read-only parcel delivery lookup by tracking ID.", DeliveryArgs),
)

SCHEMA_BY_TOOL = {name: schema for name, _, schema in TOOL_DEFINITIONS}


def _route_cases() -> list[dict[str, Any]]:
    groups = (
        ("lookup_weather", (("杭州现在天气如何？", {"city": "杭州"}), ("查一下成都的天气", {"city": "成都"}), ("我想知道深圳当前天气", {"city": "深圳"}), ("请查询西安天气", {"city": "西安"}), ("苏州今天什么天气？", {"city": "苏州"}))),
        ("lookup_exchange_rate", (("查询 USD 兑换 CNY 的汇率", {"base": "USD", "quote": "CNY"}), ("EUR 换 JPY 当前汇率是多少", {"base": "EUR", "quote": "JPY"}), ("查 GBP 到 USD 的汇率", {"base": "GBP", "quote": "USD"}), ("查询 CNY 兑换 HKD", {"base": "CNY", "quote": "HKD"}), ("AUD 对 NZD 的当前汇率", {"base": "AUD", "quote": "NZD"}))),
        ("lookup_order", (("查询订单 ORD-1042 的状态", {"order_id": "ORD-1042"}), ("订单 ORD-8871 现在到哪一步了", {"order_id": "ORD-8871"}), ("帮我查 ORD-2309 订单", {"order_id": "ORD-2309"}), ("读取订单 ORD-5510 的当前状态", {"order_id": "ORD-5510"}), ("ORD-9923 这个订单处理完了吗", {"order_id": "ORD-9923"}))),
        ("lookup_inventory", (("SKU-A17 还有多少库存", {"sku": "SKU-A17"}), ("查询 SKU-B204 的仓库库存", {"sku": "SKU-B204"}), ("SKU-C88 是否有货", {"sku": "SKU-C88"}), ("读取 SKU-D310 当前库存", {"sku": "SKU-D310"}), ("帮我查 SKU-E55 的余量", {"sku": "SKU-E55"}))),
        ("calculate_expression", (("计算 17*23", {"expression": "17*23"}), ("算一下 (48+12)/5", {"expression": "(48+12)/5"}), ("请计算 2**10", {"expression": "2**10"}), ("求 999-347", {"expression": "999-347"}), ("计算 81/9+6", {"expression": "81/9+6"}))),
        ("lookup_time", (("查询 Asia/Shanghai 当前时间", {"timezone": "Asia/Shanghai"}), ("America/New_York 现在几点", {"timezone": "America/New_York"}), ("读取 Europe/London 当前时间", {"timezone": "Europe/London"}), ("Asia/Tokyo 现在的时间", {"timezone": "Asia/Tokyo"}), ("查 Australia/Sydney 当前时间", {"timezone": "Australia/Sydney"}))),
        ("search_logs", (("在 payment 服务日志中搜索 timeout", {"service": "payment", "keyword": "timeout"}), ("查询 gateway 日志里的 502", {"service": "gateway", "keyword": "502"}), ("在 auth 服务日志检索 invalid_token", {"service": "auth", "keyword": "invalid_token"}), ("搜索 inventory 日志中的 deadlock", {"service": "inventory", "keyword": "deadlock"}), ("从 checkout 日志找 retry_exhausted", {"service": "checkout", "keyword": "retry_exhausted"}))),
        ("get_service_status", (("查询 payment 服务健康状态", {"service": "payment"}), ("gateway 服务现在正常吗", {"service": "gateway"}), ("读取 auth 服务运行状态", {"service": "auth"}), ("检查 inventory 服务是否健康", {"service": "inventory"}), ("checkout 服务当前状态如何", {"service": "checkout"}))),
        ("query_user_profile", (("查询用户 U-102 的画像", {"user_id": "U-102"}), ("读取 U-887 用户资料", {"user_id": "U-887"}), ("查一下用户 U-309", {"user_id": "U-309"}), ("获取 U-551 的用户画像", {"user_id": "U-551"}), ("用户 U-923 的资料是什么", {"user_id": "U-923"}))),
        ("lookup_delivery", (("查询快递 TRK-1042 到哪了", {"tracking_id": "TRK-1042"}), ("读取 TRK-8871 的物流状态", {"tracking_id": "TRK-8871"}), ("包裹 TRK-2309 送到了吗", {"tracking_id": "TRK-2309"}), ("查一下 TRK-5510 的配送进度", {"tracking_id": "TRK-5510"}), ("TRK-9923 当前在哪里", {"tracking_id": "TRK-9923"}))),
    )
    cases: list[dict[str, Any]] = []
    for tool_name, rows in groups:
        for index, (query, args) in enumerate(rows, 1):
            cases.append({"id": f"route_{tool_name}_{index}", "query": query, "tool": tool_name, "args": args})
    return cases


def _plan_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, suffix in enumerate(("alpha", "bravo", "charlie", "delta", "echo"), 1):
        specs = (
            (f"读取 orders_{suffix}.csv，筛选 status=paid 的记录，再把结果写入 paid_{suffix}.json", (f"orders_{suffix}.csv", "status=paid", f"paid_{suffix}.json")),
            (f"检查 gateway_{suffix}.log，统计 timeout_code_{index} 出现次数，最后生成 incident_{suffix}.md", (f"gateway_{suffix}.log", f"timeout_code_{index}", f"incident_{suffix}.md")),
            (f"读取 service_{suffix}.yaml，将 replicas_{index} 改为 {index + 2}，然后验证 config_{suffix}_valid", (f"service_{suffix}.yaml", f"replicas_{index}", f"config_{suffix}_valid")),
            (f"列出 assets_{suffix} 目录，找出 .png_{index} 文件，再把清单保存到 images_{suffix}.txt", (f"assets_{suffix}", f".png_{index}", f"images_{suffix}.txt")),
            (f"查询 api_{suffix} 状态，检索 alert_{index}，最后输出 summary_{suffix}.txt", (f"api_{suffix}", f"alert_{index}", f"summary_{suffix}.txt")),
        )
        for kind, (goal, anchors) in enumerate(specs, 1):
            cases.append({"id": f"plan_{kind}_{suffix}", "goal": goal, "anchors": list(anchors)})
    return cases


def _loop_cases() -> list[dict[str, Any]]:
    selected = _route_cases()[0:5] + _route_cases()[10:15] + _route_cases()[25:30] + _route_cases()[30:35] + _route_cases()[45:50]
    cases = []
    for index, case in enumerate(selected, 1):
        copied = dict(case)
        copied["id"] = f"loop_{index:02d}"
        copied["query"] = f"{case['query']}。必须调用工具，并在最终回答中原样带上工具返回的验证令牌。"
        copied["token"] = f"CC_LIVE_TOKEN_{index:02d}"
        cases.append(copied)
    return cases


ROUTE_CASES = _route_cases()
PLAN_CASES = _plan_cases()
LOOP_CASES = _loop_cases()


def build_safe_tools(tokens: dict[str, str] | None = None) -> list[StructuredTool]:
    tokens = tokens or {}
    tools = []
    for name, description, schema in TOOL_DEFINITIONS:
        def run(_tool_name: str = name, **kwargs: Any) -> str:
            case_id = str(kwargs.pop("_case_id", ""))
            token = tokens.get(case_id, "CC_SAFE_RESULT")
            payload = json.dumps(kwargs, ensure_ascii=False, sort_keys=True)
            return f"RESULT={payload}; VERIFICATION_TOKEN={token}. Include the exact token in the final answer."

        tools.append(StructuredTool.from_function(func=run, name=name, description=description, args_schema=schema))
    return tools


def _normalise(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {key: _normalise(item) for key, item in value.items()}
    return value


def score_route_message(message: AIMessage, case: dict[str, Any]) -> dict[str, Any]:
    calls = list(message.tool_calls or [])
    actual_name = calls[0].get("name", "") if len(calls) == 1 else ""
    actual_args = calls[0].get("args", {}) if len(calls) == 1 else {}
    schema_valid = False
    if len(calls) == 1 and actual_name in SCHEMA_BY_TOOL:
        try:
            SCHEMA_BY_TOOL[actual_name].model_validate(actual_args)
            schema_valid = True
        except Exception:
            schema_valid = False
    tool_correct = len(calls) == 1 and actual_name == case["tool"]
    args_correct = tool_correct and _normalise(actual_args) == _normalise(case["args"])
    return {
        "passed": bool(tool_correct and schema_valid and args_correct),
        "tool_correct": tool_correct,
        "schema_valid": schema_valid,
        "args_correct": args_correct,
        "tool_call_count": len(calls),
        "actual_tool": actual_name,
        "actual_args": actual_args,
    }


def score_plan_text(raw: str, case: dict[str, Any]) -> dict[str, Any]:
    strict_steps: list[str] = []
    strict_format = False
    try:
        parsed = json.loads(raw.strip())
        if isinstance(parsed, list) and 2 <= len(parsed) <= 6 and all(isinstance(step, str) and step.strip() for step in parsed):
            strict_steps = [step.strip() for step in parsed]
            strict_format = True
    except (json.JSONDecodeError, TypeError):
        pass
    runtime_steps = [step["action"] for step in _parse_plan(raw)]
    steps = strict_steps if strict_format else runtime_steps
    def anchor_key(text: str) -> str:
        text = re.sub(r"\s+", "", text.lower())
        return text.replace("等于", "=").replace("＝", "=")

    anchor_indexes = []
    for anchor in case["anchors"]:
        expected = anchor_key(anchor)
        anchor_indexes.append(next((i for i, step in enumerate(steps) if expected in anchor_key(step)), -1))
    coverage = all(index >= 0 for index in anchor_indexes)
    ordered_distinct_steps = coverage and anchor_indexes == sorted(anchor_indexes) and len(set(anchor_indexes)) == len(anchor_indexes)
    passed = strict_format and coverage and ordered_distinct_steps
    return {
        "passed": passed,
        "strict_format": strict_format,
        "anchor_coverage": coverage,
        "ordered_distinct_steps": ordered_distinct_steps,
        "step_count": len(steps),
        "steps": steps,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _usage(message: AIMessage) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


async def _with_retries(factory: Callable[[], Awaitable[AIMessage]], retries: int) -> tuple[AIMessage | None, str, int, float]:
    started = time.perf_counter()
    last_error = ""
    for attempt in range(1, retries + 2):
        try:
            return await factory(), "", attempt, time.perf_counter() - started
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:300]
            if attempt <= retries:
                await asyncio.sleep(min(2.0, 0.4 * attempt))
    return None, last_error, retries + 1, time.perf_counter() - started


class StartRateLimiter:
    """Space API request starts globally while allowing responses to overlap."""

    def __init__(self, requests_per_minute: float):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._interval = 60.0 / requests_per_minute
        self._next_start = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            loop = asyncio.get_running_loop()
            delay = self._next_start - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start = loop.time() + self._interval


def _metric_summary(name: str, details: list[dict[str, Any]], llm_calls: int) -> dict[str, Any]:
    total = len(details)
    successes = sum(bool(item.get("passed")) for item in details)
    latencies = [float(item["latency_seconds"]) for item in details if item.get("completed")]
    token_rows = [item.get("usage", {}) for item in details]
    return {
        "name": name,
        "sample_count": total,
        "success_count": successes,
        "failure_count": total - successes,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "wilson_95_ci": wilson_interval(successes, total),
        "api_error_count": sum(bool(item.get("error")) for item in details),
        "p50_latency_seconds": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "mean_llm_calls": round(llm_calls / total, 3) if total else 0.0,
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in token_rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in token_rows),
        "details": details,
    }


async def run_live_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.provider or os.getenv("DEFAULT_PROVIDER", "openai")
    model_name = args.model or os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    llm = get_provider(
        provider_name=provider,
        model_name=model_name,
        temperature=0.0,
        max_retries=0,
        timeout=args.request_timeout,
    )
    loop_tokens = {case["id"]: case["token"] for case in LOOP_CASES}  # 构建循环用例的令牌字典,用途：用于循环用例的令牌验证
    tools = build_safe_tools(loop_tokens)  # 构建安全的工具列表,为什么安全是因为这些工具不会执行真实的操作，而是返回一个格式化的结果字符串，包含请求参数和验证令牌。
    bound = llm.bind_tools(tools)  # 将工具绑定到模型中
    semaphore = asyncio.Semaphore(args.concurrency)  # 限制并发请求数
    rate_limiter = StartRateLimiter(args.requests_per_minute)
    progress_lock = asyncio.Lock()
    progress: dict[str, int] = {}

    reusable: dict[tuple[str, str], dict[str, Any]] = {}
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            previous = json.loads(resume_path.read_text(encoding="utf-8"))
            for metric_key, metric in previous.get("metrics", {}).items():
                for detail in metric.get("details", []):
                    if detail.get("completed") and not detail.get("error"):
                        reusable[(metric_key, detail["id"])] = detail

    def reused_detail(metric_key: str, case_id: str) -> dict[str, Any] | None:
        detail = reusable.get((metric_key, case_id))
        if detail is None:
            return None
        return {**detail, "reused": True}

    async def mark_progress(label: str, total: int, reused: bool = False) -> None:
        async with progress_lock:
            progress[label] = progress.get(label, 0) + 1
            suffix = " reused" if reused else ""
            print(f"[{label}] {progress[label]}/{total}{suffix}", flush=True)

    async def limited(call: Callable[[], Awaitable[AIMessage]]):
        async with semaphore:
            async def throttled() -> AIMessage:
                await rate_limiter.acquire()
                return await call()
            return await _with_retries(throttled, args.retries)

    route_cases = ROUTE_CASES[:args.limit] if args.limit else ROUTE_CASES
    plan_cases = PLAN_CASES[:args.limit] if args.limit else PLAN_CASES
    loop_cases = LOOP_CASES[:args.limit] if args.limit else LOOP_CASES
    metrics: dict[str, Any] = {}

    if args.suite in {"all", "route"}:
        async def route_one(case: dict[str, Any]) -> dict[str, Any]:
            reused = reused_detail("tool_routing", case["id"])
            if reused is not None:
                await mark_progress("route", len(route_cases), reused=True)
                return reused
            async def call() -> AIMessage:
                return await bound.ainvoke([
                    SystemMessage(content="You are being evaluated for tool routing. Call exactly one best matching tool. Copy every requested argument exactly. Do not answer directly."),
                    HumanMessage(content=case["query"]),
                ])
            message, error, attempts, latency = await limited(call)
            detail = {"id": case["id"], "completed": message is not None, "error": error, "attempts": attempts, "latency_seconds": round(latency, 3)}
            if message is not None:
                detail.update(score_route_message(message, case))
                detail["usage"] = _usage(message)
            else:
                detail.update({"passed": False, "usage": {}})
            await mark_progress("route", len(route_cases))
            return detail

        route_details = await asyncio.gather(*(route_one(case) for case in route_cases))
        metrics["tool_routing"] = _metric_summary("工具路由及参数准确率", route_details, len(route_details))

    if args.suite in {"all", "plan"}:
        async def plan_one(case: dict[str, Any]) -> dict[str, Any]:
            reused = reused_detail("planning", case["id"])
            if reused is not None:
                # Re-score old valid responses after deterministic scorer fixes.
                if reused.get("raw_excerpt"):
                    reused.update(score_plan_text(str(reused["raw_excerpt"]), case))
                await mark_progress("plan", len(plan_cases), reused=True)
                return reused
            async def call() -> AIMessage:
                return await llm.ainvoke([HumanMessage(content=_build_planner_prompt(case["goal"]))])
            message, error, attempts, latency = await limited(call)
            detail = {"id": case["id"], "completed": message is not None, "error": error, "attempts": attempts, "latency_seconds": round(latency, 3)}
            if message is not None:
                raw = str(message.content)
                detail.update(score_plan_text(raw, case))
                detail["usage"] = _usage(message)
                detail["raw_excerpt"] = raw[:500]
            else:
                detail.update({"passed": False, "usage": {}})
            await mark_progress("plan", len(plan_cases))
            return detail

        plan_details = await asyncio.gather(*(plan_one(case) for case in plan_cases))
        metrics["planning"] = _metric_summary("复杂任务计划约束通过率", plan_details, len(plan_details))

    if args.suite in {"all", "loop"}:
        async def loop_one(case: dict[str, Any]) -> dict[str, Any]:
            reused = reused_detail("tool_loop", case["id"])
            if reused is not None:
                await mark_progress("loop", len(loop_cases), reused=True)
                return reused
            conversation = [
                SystemMessage(content="Complete the request with exactly one tool call. After receiving the ToolMessage, answer the user and include the exact VERIFICATION_TOKEN. Do not call another tool."),
                HumanMessage(content=case["query"]),
            ]
            async def first_call() -> AIMessage:
                return await bound.ainvoke(conversation)
            first, first_error, first_attempts, first_latency = await limited(first_call)
            detail = {"id": case["id"], "completed": False, "error": first_error, "attempts": first_attempts, "latency_seconds": round(first_latency, 3), "usage": {}}
            if first is None:
                detail["passed"] = False
                await mark_progress("loop", len(loop_cases))
                return detail
            route_score = score_route_message(first, case)
            calls = list(first.tool_calls or [])
            if not route_score["passed"] or len(calls) != 1:
                detail.update(route_score)
                detail["passed"] = False
                detail["completed"] = True
                detail["usage"] = _usage(first)
                await mark_progress("loop", len(loop_cases))
                return detail
            call = calls[0]
            token_result = f"VERIFICATION_TOKEN={case['token']}. Include the exact token in the final answer."
            tool_message = ToolMessage(content=token_result, tool_call_id=call["id"], name=call["name"])
            async def second_call() -> AIMessage:
                return await bound.ainvoke(conversation + [first, tool_message])
            final, second_error, second_attempts, second_latency = await limited(second_call)
            first_usage = _usage(first)
            detail["attempts"] += second_attempts
            detail["latency_seconds"] = round(first_latency + second_latency, 3)
            detail["error"] = second_error
            if final is None:
                detail["passed"] = False
                detail["usage"] = first_usage
                await mark_progress("loop", len(loop_cases))
                return detail
            final_usage = _usage(final)
            final_text = str(final.content)
            no_extra_call = not final.tool_calls
            token_grounded = case["token"] in final_text
            detail.update({
                "passed": bool(no_extra_call and token_grounded),
                "completed": True,
                "route_correct": True,
                "tool_call_id_paired": tool_message.tool_call_id == call["id"],
                "final_without_tool_call": no_extra_call,
                "observation_token_grounded": token_grounded,
                "final_excerpt": final_text[:500],
                "usage": {key: first_usage[key] + final_usage[key] for key in first_usage},
            })
            await mark_progress("loop", len(loop_cases))
            return detail

        loop_details = await asyncio.gather(*(loop_one(case) for case in loop_cases))
        llm_calls = sum(2 if item.get("route_correct") else 1 for item in loop_details)
        metrics["tool_loop"] = _metric_summary("ToolMessage工具闭环完成率", loop_details, llm_calls)

    all_details = [detail for metric in metrics.values() for detail in metric["details"]]
    report = {
        "evaluation": "FluxAgent real-model engineering metrics",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "provider": provider,
        "model": model_name,
        "temperature": 0.0,
        "concurrency": args.concurrency,
        "requests_per_minute": args.requests_per_minute,
        "request_timeout_seconds": args.request_timeout,
        "retries": args.retries,
        "scoring": "deterministic rules; no LLM-as-judge",
        "dataset": {
            "tool_routing_cases": len(route_cases) if args.suite in {"all", "route"} else 0,
            "planning_cases": len(plan_cases) if args.suite in {"all", "plan"} else 0,
            "tool_loop_cases": len(loop_cases) if args.suite in {"all", "loop"} else 0,
            "total_cases": len(all_details),
        },
        "metrics": metrics,
    }
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print(f"provider={report['provider']} model={report['model']} temperature=0")
    print(f"cases={report['dataset']['total_cases']}")
    for metric in report["metrics"].values():
        low, high = metric["wilson_95_ci"]
        print(
            f"[{metric['name']}] {metric['success_count']}/{metric['sample_count']} "
            f"rate={metric['success_rate']:.1%} ci95=[{low:.1%},{high:.1%}] "
            f"p50={metric['p50_latency_seconds']}s p95={metric['p95_latency_seconds']}s "
            f"api_errors={metric['api_error_count']}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paid real-model FluxAgent engineering metrics")
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--suite", choices=("all", "route", "plan", "loop"), default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit cases per selected suite; 0 runs all")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests-per-minute", type=float, default=9.0)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", help="Reuse completed cases from an earlier JSON report")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "logs" / "live_engineering_metrics.json"))
    args = parser.parse_args()
    report = asyncio.run(run_live_evaluation(args))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(report)
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
