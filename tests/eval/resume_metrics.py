"""Resume-facing deterministic metrics for FluxAgent.

This module intentionally separates two kinds of evidence:

1. Rule/workflow metrics: run with ScriptedLLM and deterministic fake tools.
2. Memory structure metrics: compare watermark policies without claiming
   semantic recall or real-token savings.

No network model is called by this module.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from statistics import mean
from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage

from fluxagent.core.context import measure_context, trim_context_messages
from fluxagent.core.logger import audit_logger

from .cases import all_cases
from .runners import run_case


MetricSelector = Callable[[str, dict[str, Any]], bool]


METRIC_SELECTORS: dict[str, MetricSelector] = {
    "workflow_correctness": lambda suite, case: suite == "task_state",
    "planner_route_accuracy": lambda suite, case: suite == "task_state" and case["kind"] == "route",
    "approval_compliance": lambda suite, case: suite == "hitl",
    "sandbox_policy_compliance": lambda suite, case: suite == "tools" and case["kind"] in {"shell", "path"},
    "mcp_boundary_compliance": lambda suite, case: suite == "mcp" and case["kind"] in {
        "mcp_call", "mcp_blockflag"
    },
    "recovery_logic": lambda suite, case: suite == "runtime" and case["kind"] in {
        "classify", "replan_flow", "replan_validate"
    },
    "budget_guard": lambda suite, case: suite == "runtime" and case["kind"] in {
        "budget_check", "budget_flow"
    },
    "session_management": lambda suite, case: suite == "runtime" and case["kind"] in {
        "session_store", "session_isolation"
    },
}

CRITICAL_KINDS = {
    "shell", "path", "gate", "multi_gate", "mcp_call", "mcp_blockflag", "hard_deny"
}


def _ratio(passed: int, total: int) -> dict[str, Any]:
    return {
        "passed": passed,
        "total": total,
        "rate": round(passed / total, 4) if total else None,
    }


def run_rule_metrics() -> dict[str, Any]:
    """Run deterministic cases and aggregate metrics useful on a resume."""
    counters = defaultdict(lambda: [0, 0])
    suite_counts = defaultdict(lambda: [0, 0])
    critical = [0, 0]
    failures: list[dict[str, str]] = []

    # These cases validate graph/policy behavior, not the asynchronous writer.
    # Disable audit disk output so the evaluator is hermetic and low-noise.
    original_log_event = audit_logger.log_event
    audit_logger.log_event = lambda *args, **kwargs: None
    try:
        for suite, cases in all_cases().items():
            for case in cases:
                with contextlib.redirect_stdout(io.StringIO()):
                    passed, detail = run_case(case)
                suite_counts[suite][1] += 1
                suite_counts[suite][0] += int(passed)
                if not passed:
                    failures.append({"suite": suite, "id": case["id"], "detail": detail})

                for metric, selector in METRIC_SELECTORS.items():
                    if selector(suite, case):
                        counters[metric][1] += 1
                        counters[metric][0] += int(passed)

                if case["kind"] in CRITICAL_KINDS:
                    critical[1] += 1
                    critical[0] += int(passed)
    finally:
        audit_logger.log_event = original_log_event

    total_passed = sum(value[0] for value in suite_counts.values())
    total = sum(value[1] for value in suite_counts.values())
    return {
        "scope": "deterministic_workflow_and_policy_cases",
        "uses_real_llm": False,
        "overall": _ratio(total_passed, total),
        "suites": {name: _ratio(*values) for name, values in sorted(suite_counts.items())},
        "metrics": {name: _ratio(*values) for name, values in counters.items()},
        "safety_critical_cases": _ratio(*critical),
        "critical_failures": critical[1] - critical[0],
        "failures": failures[:20],
    }


def benchmark_watermark(total_turns: int, trigger_turns: int, keep_turns: int) -> dict[str, Any]:
    """Replay equal-sized turns through the real trimming function."""
    messages = []
    summary_calls = 0
    raw_turns_after_trim = []
    compressed_turns_total = 0

    for turn in range(1, total_turns + 1):
        messages.extend([
            HumanMessage(content=f"user-{turn}"),
            AIMessage(content=f"assistant-{turn}"),
        ])
        kept, discarded = trim_context_messages(
            messages,
            trigger_turns=trigger_turns,
            keep_turns=keep_turns,
        )
        if discarded:
            summary_calls += 1
            compressed_turns_total += measure_context(discarded)["turns"]
            messages = kept
        raw_turns_after_trim.append(measure_context(messages)["turns"])

    return {
        "policy": f"{trigger_turns}/{keep_turns}",
        "total_turns": total_turns,
        "summary_calls": summary_calls,
        "compressed_turns_total": compressed_turns_total,
        "raw_turns_at_end": measure_context(messages)["turns"],
        "mean_raw_turns_after_trim": round(mean(raw_turns_after_trim), 3),
        "max_raw_turns_after_trim": max(raw_turns_after_trim),
    }


def run_memory_comparison(total_turns: int = 60) -> dict[str, Any]:
    baseline = benchmark_watermark(total_turns, 8, 4)
    candidate = benchmark_watermark(total_turns, 20, 10)
    reduction = 0.0
    if baseline["summary_calls"]:
        reduction = 1 - candidate["summary_calls"] / baseline["summary_calls"]
    return {
        "scope": "structural_watermark_replay",
        "uses_real_llm": False,
        "does_not_measure": ["semantic_recall", "summary_faithfulness", "real_token_cost"],
        "baseline": baseline,
        "candidate": candidate,
        "summary_call_reduction_vs_8_4": round(reduction, 4),
    }


def build_report(total_turns: int = 60) -> dict[str, Any]:
    return {
        "rule_metrics": run_rule_metrics(),
        "memory_comparison": run_memory_comparison(total_turns),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate resume-safe deterministic metrics")
    parser.add_argument("--turns", type=int, default=60, help="turns used in watermark replay")
    parser.add_argument("--output", help="optional JSON output path")
    args = parser.parse_args()

    report = build_report(args.turns)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0 if report["rule_metrics"]["critical_failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
