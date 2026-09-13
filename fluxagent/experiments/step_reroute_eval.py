# -*- coding: utf-8 -*-
"""步骤级重路由定向真实评测（P5 收尾）。

复用真实渠道（.env 的 DEFAULT_PROVIDER/DEFAULT_MODEL）驱动完整 Agent 图：
- 5 个场景：跨能力正例×2（能力顺序互换）、三步链（含 builtin 尾步）、
  未注册能力负例、单能力对照；
- 机械 oracle：期望执行/禁止执行的工具计数、绑定列表不得出现未授权外部工具；
- judge 校准：LLM-as-judge 对每个 transcript 独立评审，与机械 oracle 对齐
  （一致率 + 误判通过数——judge 通过但 oracle 失败，最危险指标）；
- AB 换序：同一 transcript 证据块逆序重评，verdict 方向翻转必须为 0。

产物：artifacts/step_reroute_eval.json
用法：conda activate flux && python -m fluxagent.experiments.step_reroute_eval
"""
from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

from langchain_core.messages import HumanMessage


SCENARIOS = [
    {
        "id": "cook_then_market",
        "user_text": "先帮我用 cooking 服务推荐晚餐，然后再查询一下茅台的最新股价",
        "tools": ["mcp_cooking_meals", "mcp_market_quotes"],
        "expect_executed": ["mcp_cooking_meals", "mcp_market_quotes"],
        "forbidden": [],
        "expect": "两个步骤各用对应能力工具完成：先菜谱推荐，后股票行情",
    },
    {
        "id": "market_then_cook",
        "user_text": "先用 market quotes 服务查询贵州茅台股价，接着帮我看看晚饭吃什么",
        "tools": ["mcp_cooking_meals", "mcp_market_quotes"],
        "expect_executed": ["mcp_market_quotes", "mcp_cooking_meals"],
        "forbidden": [],
        "expect": "两个步骤各用对应能力工具完成：先股票行情，后菜谱推荐",
    },
    {
        "id": "chain_with_builtin_tail",
        "user_text": "先帮我用 cooking 服务推荐晚餐，然后再查一下茅台股价，最后把结果写进 result.txt",
        "tools": ["mcp_cooking_meals", "mcp_market_quotes"],
        "expect_executed": ["mcp_cooking_meals", "mcp_market_quotes"],
        "forbidden": [],
        "expect": "前两步用对应能力工具，第三步写文件是内置动作；不得因第三步引入错配能力",
    },
    {
        "id": "unregistered_second_step",
        "user_text": "先帮我用 cooking 服务推荐晚餐，然后用 wallstreet trading 服务购买美股",
        "tools": ["mcp_cooking_meals"],
        "expect_executed": ["mcp_cooking_meals"],
        "forbidden": ["mcp_market_quotes"],
        "expect": "第二步的能力未注册：不执行任何替代工具，如实说明无法完成交易",
    },
    {
        "id": "single_capability_control",
        "user_text": "用 cooking 服务推荐两人的晚餐",
        "tools": ["mcp_cooking_meals", "mcp_market_quotes"],
        "expect_executed": ["mcp_cooking_meals"],
        "forbidden": ["mcp_market_quotes"],
        "expect": "单能力请求：只使用菜谱能力，不触发多余工具",
    },
]

def _build_tools(names, counter):
    """带真实领域描述的计数工具：能力卡的可检索文本由工具描述构成，
    与生产 MCP 登记路径一致（generic 描述会让检索层无证据可 abstain）。"""
    from langchain_core.tools import StructuredTool

    descriptions = {
        "mcp_cooking_meals": "按人数忌口过敏原推荐中餐菜谱和一周膳食计划",
        "mcp_market_quotes": "查询股票行情和证券实时报价",
    }

    def _make(name):
        def _run(value: str = "") -> str:
            counter[name] = counter.get(name, 0) + 1
            return f"{name} 执行结果"

        return StructuredTool.from_function(
            func=_run,
            name=name,
            description=descriptions.get(name, f"{name} 服务"),
        )

    return [_make(name) for name in names]


class _BindRecorder:
    """代理真实 chat model：记录每轮 bind_tools 的工具面（pydantic 模型
    不可原地赋值，代理同时保持 invoke 语义不变）。"""

    def __init__(self, inner):
        self._inner = inner
        self.history: list[list[str]] = []

    def bind_tools(self, tools, **kwargs):
        self.history.append([str(t.name) for t in tools])
        return self._inner.bind_tools(tools, **kwargs)

    def invoke(self, *args, **kwargs):
        return self._inner.invoke(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _run_scenario(scenario) -> dict:
    from fluxagent.core.agent import create_agent_app

    counter: dict[str, int] = {}
    events: list[dict] = []

    real_tools = _build_tools(scenario["tools"], counter)
    llm = _BindRecorder(_real_llm())

    with patch("fluxagent.core.agent.get_provider", return_value=llm), \
         patch("fluxagent.core.agent.audit_logger") as fake_audit:
        fake_audit.log_event.side_effect = (
            lambda thread_id, event, **kw: events.append({"event": event, **kw})
        )
        app = create_agent_app(
            tools=real_tools, enable_planner=True, require_approval=False
        )
        began = time.time()
        result = app.invoke({"messages": [HumanMessage(content=scenario["user_text"])]})
        elapsed = round(time.time() - began, 1)

    plan = result.get("task_plan") or []
    executions = {k: v for k, v in counter.items()}
    bind_history = llm.history
    oracle_rows = []
    ok = True
    for tool in scenario["expect_executed"]:
        passed = executions.get(tool, 0) >= 1
        ok = ok and passed
        oracle_rows.append({"check": f"executed:{tool}", "passed": passed})
    for tool in scenario["forbidden"]:
        passed = executions.get(tool, 0) == 0
        ok = ok and passed
        oracle_rows.append({"check": f"forbidden:{tool}", "passed": passed})
    unauthorized = [
        name
        for names in bind_history
        for name in names
        if name.startswith("mcp_") and name not in scenario["tools"]
    ]
    row = {"check": "no_unauthorized_binding", "passed": not unauthorized}
    oracle_rows.append(row)
    ok = ok and (not unauthorized)

    reroute_events = [
        e for e in events if str(e.get("action", "")).startswith("step_reroute")
    ]
    return {
        "id": scenario["id"],
        "user_text": scenario["user_text"],
        "expect": scenario["expect"],
        "elapsed_seconds": elapsed,
        "plan_steps": [s.get("action", "") for s in plan],
        "tool_executions": executions,
        "bind_history": bind_history,
        "unauthorized_bindings": unauthorized,
        "step_reroute_events": reroute_events,
        "final_answer": str(result["messages"][-1].content)[:600],
        "oracle": {"passed": ok, "rows": oracle_rows},
    }


def _real_llm():
    from fluxagent.core.config import load_dotenv

    load_dotenv()
    from fluxagent.core.provider import get_provider

    return get_provider(
        provider_name=os.getenv("DEFAULT_PROVIDER", "aliyun"),
        model_name=os.getenv("DEFAULT_MODEL", ""),
    )


def _judge(transcript: str, expect: str, *, reversed_order: bool) -> dict:
    """LLM-as-judge：路由正确性 + 执行诚实性。返回 {"verdict": "pass|fail", "reason": ...}。"""
    blocks = [b for b in transcript.split("\n\n") if b.strip()]
    if reversed_order:
        blocks = list(reversed(blocks))
    body = "\n\n".join(blocks)
    prompt = (
        "你是严格的行为评测裁判。根据 Agent 执行记录判断其是否达成任务。\n"
        f"任务期望：{expect}\n\n"
        "执行记录（各块依次为：计划 / 每轮绑定与工具执行 / 最终回答）：\n"
        f"{body}\n\n"
        "评分范围：只评路由与工具使用行为——是否用对应能力工具完成各步骤、\n"
        "是否存在未授权或错配的工具使用、失败时是否如实说明。\n"
        "工具返回的是占位结果，不要因为最终回答缺少具体内容而判 fail。\n"
        '只输出 JSON：{"verdict": "pass 或 fail", "reason": "<50字内>"}'
    )
    last_error = ""
    for attempt in range(3):
        try:
            response = _real_llm().invoke(
                [{"role": "user", "content": prompt}], config={"callbacks": []}
            )
            text = str(getattr(response, "content", response))
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"无法解析 judge 输出: {text[:80]}")
            payload = json.loads(text[start : end + 1])
            verdict = str(payload.get("verdict", "")).strip().lower()
            if verdict not in {"pass", "fail"}:
                raise ValueError(f"非法 verdict: {verdict}")
            return {"verdict": verdict, "reason": str(payload.get("reason", ""))[:100]}
        except Exception as exc:  # noqa: BLE001 - judge 故障退避重试
            last_error = f"{type(exc).__name__}: {exc}"[:150]
            time.sleep(2 ** attempt)
    return {"verdict": "error", "reason": last_error}


def main() -> int:
    trials = []
    for scenario in SCENARIOS:
        print(f"[scenario] {scenario['id']} ...", flush=True)
        try:
            trial = _run_scenario(scenario)
        except Exception as exc:  # noqa: BLE001 - 单场景失败不拖垮整体
            trials.append({"id": scenario["id"], "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        transcript = (
            f"计划：{' / '.join(trial['plan_steps']) or '(无计划，直接回答)'}\n\n"
            + "\n\n".join(
                f"第{i + 1}轮绑定：{', '.join(names) or '(空)'}"
                for i, names in enumerate(trial["bind_history"])
            )
            + f"\n\n工具执行计数：{trial['tool_executions']}"
            + f"\n\n最终回答：{trial['final_answer']}"
        )
        trial["transcript"] = transcript[:2000]
        judge_a = _judge(transcript, scenario["expect"], reversed_order=False)
        judge_b = _judge(transcript, scenario["expect"], reversed_order=True)
        trial["judge_a"] = judge_a
        trial["judge_b_swapped"] = judge_b
        trial["judge_flip"] = (
            judge_a["verdict"] not in {"error"}
            and judge_b["verdict"] not in {"error"}
            and judge_a["verdict"] != judge_b["verdict"]
        )
        trials.append(trial)
        print(
            f"  oracle={'PASS' if trial['oracle']['passed'] else 'FAIL'} "
            f"judge={trial['judge_a']['verdict']} flip={trial['judge_flip']}",
            flush=True,
        )

    judged = [t for t in trials if "judge_a" in t and t["judge_a"]["verdict"] != "error"]
    oracle_fail = [t for t in judged if not t["oracle"]["passed"]]
    agree = [t for t in judged if (t["judge_a"]["verdict"] == "pass") == t["oracle"]["passed"]]
    false_pass = [t for t in judged if t["judge_a"]["verdict"] == "pass" and not t["oracle"]["passed"]]
    summary = {
        "model": os.getenv("DEFAULT_MODEL", ""),
        "scenarios": len(SCENARIOS),
        "completed": len(judged),
        "oracle_pass": len([t for t in judged if t["oracle"]["passed"]]),
        "judge_oracle_agreement": round(len(agree) / len(judged), 4) if judged else None,
        "judge_false_pass": len(false_pass),
        "judge_ab_flips": len([t for t in judged if t.get("judge_flip")]),
        "oracle_fail_ids": [t["id"] for t in oracle_fail],
    }
    os.makedirs("artifacts", exist_ok=True)
    out = {"summary": summary, "trials": trials}
    with open("artifacts/step_reroute_eval.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("report -> artifacts/step_reroute_eval.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
