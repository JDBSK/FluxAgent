"""评测入口：跑全部套件，算指标，比基线，出报告。

    conda activate cyber
    python -m tests.eval.run_suite                 # 全跑
    python -m tests.eval.run_suite --suite hitl    # 单套件
    python -m tests.eval.run_suite --save-baseline # 更新回归基线
    python -m tests.eval.run_suite --judge         # 额外跑 LLM-as-judge 抽测（要 API Key）

回归基线的意义：单看"通过率 100%"没用，改一行代码之后还是不是 100% 才有用。
基线存 pass/fail 的逐条结果，对比时能指出"哪条从 pass 变 fail"，
而不只是给个总分下降的模糊信号。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

from .cases import all_cases
from .runners import run_case

BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baseline.json")
REPORT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs"
)

# 安全类断言不允许有任何失败——通过率 99% 意味着有一条越权没拦住
CRITICAL_KINDS = {"shell", "path", "gate", "multi_gate", "mcp_call", "mcp_blockflag", "hard_deny"}


def run_suite(name: str, cases: list[dict[str, Any]], verbose: bool = False) -> dict[str, Any]:
    results = []
    by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    started = time.time()

    for case in cases:
        t0 = time.time()
        passed, detail = run_case(case)
        elapsed_ms = (time.time() - t0) * 1000

        by_kind[case["kind"]][0] += 1 if passed else 0
        by_kind[case["kind"]][1] += 1
        results.append({
            "id": case["id"], "kind": case["kind"],
            "passed": passed, "detail": detail, "ms": round(elapsed_ms, 1),
        })
        if verbose and not passed:
            print(f"    FAIL {case['id']}: {detail[:110]}", file=sys.stderr)

    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    critical_fails = [
        r for r in results if not r["passed"] and r["kind"] in CRITICAL_KINDS
    ]
    latencies = sorted(r["ms"] for r in results)

    return {
        "suite": name,
        "total": total,
        "passed": passed_n,
        "success_rate": round(passed_n / total, 4) if total else 0.0,
        "critical_failures": len(critical_fails),
        "critical_failure_ids": [r["id"] for r in critical_fails][:10],
        "latency_p50_ms": latencies[len(latencies) // 2] if latencies else 0,
        "latency_p95_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
        "wall_seconds": round(time.time() - started, 2),
        "by_kind": {k: {"passed": v[0], "total": v[1]} for k, v in sorted(by_kind.items())},
        "results": results,
    }


def derive_metrics(suites: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """把逐条结果聚成 docs/测评指标.md 里定义的那几个指标。

    指标不是随便挑的：每个都对应一个"改坏了会怎样"的具体后果。
    """
    def rate(suite: str, kinds: set[str]) -> float | None:
        s = suites.get(suite)
        if not s:
            return None
        hit = [v for k, v in s["by_kind"].items() if k in kinds]
        if not hit:
            return None
        p = sum(v["passed"] for v in hit)
        t = sum(v["total"] for v in hit)
        return round(p / t, 4) if t else None

    total = sum(s["total"] for s in suites.values())
    passed = sum(s["passed"] for s in suites.values())

    return {
        # 总体通过率
        "success_rate": round(passed / total, 4) if total else 0.0,
        # 该拦的有没有拦住 + 该放的有没有放过（审批闸门是否可信）
        "interrupt_compliance": rate("hitl", {"gate", "multi_gate"}),
        # 计划与游标状态是否一致（任务状态是否可信）
        "plan_consistency": rate("task_state", {"plan", "advance"}),
        # 沙盒越权拦截率（安全底线）
        "sandbox_block_rate": rate("tools", {"shell", "path"}),
        # 工具/技能行为正确率
        "tool_accuracy": rate("tools", {"calc", "twophase", "lazy"}),
        # MCP 协议正确性
        "mcp_protocol_ok": rate("mcp", {"mcp_call", "mcp_schema", "mcp_meta", "mcp_blockflag"}),
        # 安全类失败总数，必须为 0
        "policy_compliance": rate("runtime", {"policy_assess", "hard_deny"}),
        "recovery_ok": rate("runtime", {"classify", "replan_flow"}),
        "budget_ok": rate("runtime", {"budget_check", "budget_flow"}),
        "critical_failures": sum(s["critical_failures"] for s in suites.values()),
    }


def compare_baseline(report: dict[str, Any]) -> dict[str, Any]:
    """逐条对比基线，指出回归和新修复。"""
    if not os.path.exists(BASELINE_PATH):
        return {"status": "no_baseline"}

    with open(BASELINE_PATH, encoding="utf-8") as fh:
        baseline = json.load(fh)

    base_map = baseline.get("case_results", {})
    now_map = {
        r["id"]: r["passed"]
        for s in report["suites"].values()
        for r in s["results"]
    }

    regressions = [cid for cid, ok in now_map.items() if base_map.get(cid) is True and not ok]
    fixed = [cid for cid, ok in now_map.items() if base_map.get(cid) is False and ok]
    new_cases = [cid for cid in now_map if cid not in base_map]
    # 只统计"本次跑过但结果消失"的 case。用 --suite 跑子集时，没跑到的 case
    # 不是被删除，不该算进 removed，否则子集运行会报一堆假的 removed
    removed = [cid for cid in base_map if cid not in now_map] if not report.get("partial") else []

    return {
        "status": "compared",
        "baseline_success_rate": baseline.get("metrics", {}).get("success_rate"),
        "current_success_rate": report["metrics"]["success_rate"],
        "regression_delta": round(
            report["metrics"]["success_rate"]
            - (baseline.get("metrics", {}).get("success_rate") or 0), 4
        ),
        "regressions": regressions[:20],
        "regression_count": len(regressions),
        "fixed": fixed[:20],
        "fixed_count": len(fixed),
        "new_cases": len(new_cases),
        "removed_cases": len(removed),
    }


def save_baseline(report: dict[str, Any]) -> None:
    payload = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metrics": report["metrics"],
        "case_results": {
            r["id"]: r["passed"]
            for s in report["suites"].values()
            for r in s["results"]
        },
    }
    with open(BASELINE_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"[baseline] saved {len(payload['case_results'])} cases -> {BASELINE_PATH}")


def print_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 62)
    print("FluxAgent 评测报告")
    print("=" * 62)
    for name, s in report["suites"].items():
        flag = "PASS" if s["passed"] == s["total"] else "FAIL"
        print(f"[{flag}] {name:12s} {s['passed']:4d}/{s['total']:<4d} "
              f"{s['success_rate'] * 100:6.2f}%  p95={s['latency_p95_ms']:.0f}ms  "
              f"{s['wall_seconds']}s")
        for kind, v in s["by_kind"].items():
            mark = "" if v["passed"] == v["total"] else "   <-- FAIL"
            print(f"         {kind:16s} {v['passed']:4d}/{v['total']:<4d}{mark}")

    print("-" * 62)
    print("指标:")
    for key, val in report["metrics"].items():
        if val is None:
            continue
        shown = f"{val * 100:.2f}%" if isinstance(val, float) else val
        print(f"  {key:24s} {shown}")

    cmp_ = report["baseline"]
    print("-" * 62)
    if cmp_["status"] == "no_baseline":
        print("回归: 无基线（用 --save-baseline 建立）")
    else:
        print(f"回归: delta={cmp_['regression_delta']:+.4f} "
              f"regressions={cmp_['regression_count']} fixed={cmp_['fixed_count']} "
              f"new={cmp_['new_cases']}")
        for cid in cmp_["regressions"][:5]:
            print(f"  REGRESSION: {cid}")

    total = sum(s["total"] for s in report["suites"].values())
    passed = sum(s["passed"] for s in report["suites"].values())
    crit = report["metrics"]["critical_failures"]
    print("=" * 62)
    print(f"总计 {passed}/{total} ({passed / total * 100:.2f}%) | 安全类失败 {crit}")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(description="FluxAgent 评测")
    parser.add_argument("--suite", help="只跑指定套件")
    parser.add_argument("--save-baseline", action="store_true", help="把本次结果存为基线")
    parser.add_argument("--judge", action="store_true", help="额外跑 LLM-as-judge 抽测")
    parser.add_argument("--verbose", action="store_true", help="打印失败明细")
    args = parser.parse_args()

    suites_cases = all_cases()
    if args.suite:
        if args.suite not in suites_cases:
            print(f"未知套件 {args.suite}，可选: {list(suites_cases)}", file=sys.stderr)
            return 2
        suites_cases = {args.suite: suites_cases[args.suite]}

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "partial": bool(args.suite),  # 子集运行：不能覆盖基线，也不算 removed
        "suites": {},
    }
    for name, cases in suites_cases.items():
        print(f"running {name} ({len(cases)} cases)...", file=sys.stderr)
        report["suites"][name] = run_suite(name, cases, verbose=args.verbose)

    report["metrics"] = derive_metrics(report["suites"])
    report["baseline"] = compare_baseline(report)

    if args.judge:
        from .judge import run_judge

        report["judge"] = run_judge()

    os.makedirs(REPORT_DIR, exist_ok=True)
    out_path = os.path.join(REPORT_DIR, "eval_report.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print_summary(report)
    if args.judge and report.get("judge"):
        j = report["judge"]
        print(f"LLM-as-judge: {j.get('status')} "
              f"avg={j.get('avg_score')} n={j.get('n')}")
    print(f"\n报告已写入 {out_path}")

    if args.save_baseline:
        save_baseline(report)

    # 安全类失败或出现回归 -> 非零退出码，方便挂 CI
    failed = report["metrics"]["critical_failures"] > 0
    if report["baseline"].get("regression_count", 0) > 0:
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
