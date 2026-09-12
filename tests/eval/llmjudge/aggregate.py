# -*- coding: utf-8 -*-
"""聚合与报告（prompt.md 07 的落地）：任务通过率、已决通过率、门禁、
12 维质量分（evidence_coverage 同步呈现）、Wilson 区间、分族表现。
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .prompts import DEFAULT_WEIGHTS


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson 95% 置信区间；total=0 返回 None。"""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    return (max(0.0, (center - margin) / denom), min(1.0, (center + margin) / denom))


def _quality(verdicts: list[dict], applicable: dict, weights: dict) -> dict:
    """observed_quality 与 evidence_coverage 必须同时呈现，防止重归一化幻觉。"""
    weight_sum_applicable = sum(
        weights.get(d, 0) for d, meta in (applicable or {}).items()
        if meta.get("applicable", True)
    ) or sum(weights.values())
    scored_weight = 0.0
    weighted = 0.0
    for item in verdicts:
        d = item.get("dimension_id")
        if item.get("status") == "scored" and isinstance(item.get("score"), (int, float)):
            w = weights.get(d, 0)
            scored_weight += w
            weighted += w * item["score"] / 4
    if scored_weight <= 0:
        return {"observed_quality": None, "evidence_coverage": 0.0,
                "quality_lower": 0.0, "quality_upper": None}
    observed = 100 * weighted / scored_weight
    coverage = scored_weight / weight_sum_applicable
    lower = 100 * weighted / weight_sum_applicable
    upper = lower + 100 * (weight_sum_applicable - scored_weight) / weight_sum_applicable
    return {
        "observed_quality": round(observed, 2),
        "evidence_coverage": round(coverage, 4),
        "quality_lower": round(lower, 2),
        "quality_upper": round(upper, 2),
    }


def aggregate(results: list[dict]) -> dict:
    """results: [{case, evidence, verdict}]。"""
    by_track: dict[str, list] = defaultdict(list)
    by_family: dict[str, list] = defaultdict(list)
    gates: dict[str, dict[str, int]] = defaultdict(lambda: {"pass": 0, "fail": 0, "unknown": 0})
    for item in results:
        verdict = item["verdict"]
        case = item["case"]
        by_track[case["track"]].append(verdict)
        by_family[case["family"]].append(verdict)
        for gate in verdict.get("gate_results") or []:
            gates[gate.get("gate_id", "?")][gate.get("result", "unknown")] += 1

    def summarize(verdicts: list[dict]) -> dict:
        n = len(verdicts)
        counts = {"pass": 0, "fail": 0, "inconclusive": 0, "invalid": 0}
        for v in verdicts:
            counts[v.get("overall_verdict", "inconclusive")] = counts.get(v.get("overall_verdict", "inconclusive"), 0) + 1
        decided = counts["pass"] + counts["fail"]
        task_pass = counts["pass"] / n if n else None
        ci = wilson_ci(counts["pass"], n) if n else None
        return {
            "n": n,
            "counts": counts,
            "task_pass_rate": round(task_pass, 4) if task_pass is not None else None,
            "task_pass_ci95": [round(x, 4) for x in ci] if ci else None,
            "decided_pass_rate": round(counts["pass"] / decided, 4) if decided else None,
            "inconclusive_rate": round(counts["inconclusive"] / n, 4) if n else None,
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tracks": {k: summarize(v) for k, v in sorted(by_track.items())},
        "families": {k: summarize(v) for k, v in sorted(by_family.items())},
        "gates": {k: dict(v) for k, v in sorted(gates.items())},
    }

    overall = [item["verdict"] for item in results]
    summary["overall"] = summarize(overall)

    # 质量分：只对真实 judge 的 trial（有适用维度声明）计算
    quality_items = [
        item for item in results
        if item["verdict"].get("judge_mode") == "llm-judge"
    ]
    if quality_items:
        per_family_quality = {}
        for item in quality_items:
            fam = item["case"]["family"]
            per_family_quality.setdefault(fam, []).append(item)
        summary["quality"] = {}
        for fam, items in sorted(per_family_quality.items()):
            weights = items[0]["case"].get("weights") or DEFAULT_WEIGHTS
            applicable = items[0]["case"]["judge_only"].get("applicable") or {}
            merged_scores = [
                d for item in items for d in (item["verdict"].get("dimension_scores") or [])
            ]
            summary["quality"][fam] = _quality(merged_scores, applicable, weights)
    else:
        summary["quality"] = None
    return summary


def render_markdown(summary: dict, results: list[dict], config_note: str) -> str:
    lines: list[str] = []
    lines.append("# FluxAgent LLM-as-judge 测评报告")
    lines.append("")
    lines.append(f"- 生成时间：{summary['generated_at']}")
    lines.append(f"- 配置：{config_note}")
    failed_trials = [r for r in results if r["verdict"].get("overall_verdict") == "fail"]
    lines.append(f"- 本批 trial 数：{len(results)}；失败 {len(failed_trials)}")
    lines.append("")
    lines.append("## 主指标")
    lines.append("")
    lines.append("| 轨道 | n | pass | fail | inconclusive | invalid | 任务通过率 | 95% CI | 已决通过率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for track, s in summary["tracks"].items():
        c = s["counts"]
        ci = s["task_pass_ci95"]
        lines.append(
            f"| {track} | {s['n']} | {c['pass']} | {c['fail']} | {c['inconclusive']} | {c['invalid']} "
            f"| {s['task_pass_rate']} | {ci} | {s['decided_pass_rate']} |")
    lines.append("")
    lines.append("## 门禁结果")
    lines.append("")
    if summary["gates"]:
        lines.append("| gate | pass | fail | unknown |")
        lines.append("|---|---|---|---|")
        for gate_id, counts in summary["gates"].items():
            lines.append(f"| {gate_id} | {counts['pass']} | {counts['fail']} | {counts['unknown']} |")
    else:
        lines.append("本批无门禁触发（替身试跑：门禁以程序断言形式执行）。")
    lines.append("")
    lines.append("## 场景族表现")
    lines.append("")
    lines.append("| 族 | n | pass | fail | inconclusive |")
    lines.append("|---|---|---|---|---|")
    for fam, s in summary["families"].items():
        c = s["counts"]
        lines.append(f"| {fam} | {s['n']} | {c['pass']} | {c['fail']} | {c['inconclusive']} |")
    lines.append("")
    if summary.get("quality"):
        lines.append("## 12 维质量分（仅真实 judge trial）")
        lines.append("")
        lines.append("| 族 | observed_quality | evidence_coverage | [lower, upper] |")
        lines.append("|---|---|---|---|")
        for fam, q in summary["quality"].items():
            lines.append(f"| {fam} | {q['observed_quality']} | {q['evidence_coverage']} | [{q['quality_lower']}, {q['quality_upper']}] |")
        lines.append("")
    lines.append("## 失败案例")
    lines.append("")
    if failed_trials:
        for r in failed_trials[:5]:
            v = r["verdict"]
            lines.append(f"- **{r['case']['case_id']}**：{v.get('summary', '')}")
            for c in v.get("acceptance_results") or []:
                if c.get("result") == "unmet":
                    lines.append(f"  - {c['criterion_id']}: {c.get('rationale', '')[:120]}")
    else:
        lines.append("本批无失败 trial。")
    lines.append("")
    lines.append("## 已知边界与未测项")
    lines.append("")
    lines.append("- 替身试跑（stub）只验证程序契约与框架链路，不构成模型能力结论。")
    lines.append("- 真实模型未运行的条目保留 not_run，不填通过率。")
    lines.append("- 执行防护是应用层边界，不等同 OS 沙盒。")
    return "\n".join(lines)
