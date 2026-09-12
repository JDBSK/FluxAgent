# -*- coding: utf-8 -*-
"""裁判客户端与判定合成。

dry-run（替身）：verdict 由程序断言合成，judge 字段标注 programmatic-only。
real：调用裁判 LLM（结构化 JSON 输出 + 校验 + 固定上限重试，保留原结果）。
"""
from __future__ import annotations

import json
import re
from typing import Any

from .prompts import JUDGE_SYSTEM_PROMPT, ALL_DIMENSIONS

_VERDICTS = {"pass", "fail", "inconclusive", "invalid"}
_SCORE_STATUSES = {"scored", "not_applicable", "insufficient_evidence"}


def verdict_from_checks(evidence: dict, case: dict) -> dict:
    """dry-run：程序断言全过且无门禁违例 → pass，否则 fail。

    这是替身模式下的合成判定，不是语义 judge 的结论；报告中单独标注。
    """
    checks = evidence.get("independent_checks") or []
    failed = [c["check_id"] for c in checks if c["status"] == "fail"]
    verdict = "pass" if not failed else "fail"
    dimension_scores = [
        {"dimension_id": d, "status": "not_applicable", "score": None,
         "evidence_ids": [], "rationale": "stub trial: 语义维度未评审"}
        for d in ALL_DIMENSIONS
    ]
    return {
        "case_id": case["case_id"],
        "trial_id": evidence["trial_id"],
        "track": case["track"],
        "trial_validity": "valid",
        "validity_reason": "stub trial with programmatic checks only",
        "acceptance_results": [
            {"criterion_id": c["check_id"], "required": True,
             "result": "met" if c["status"] == "pass" else "unmet",
             "evidence_ids": [c["check_id"]], "rationale": c["actual"]}
            for c in checks
        ],
        "gate_results": [],
        "dimension_scores": dimension_scores,
        "overall_verdict": verdict,
        "task_success": True if verdict == "pass" else False,
        "issues": [],
        "missing_evidence": [],
        "confidence": "high",
        "review_required": False,
        "summary": "替身试跑：程序断言 " + ("全部通过" if not failed else f"失败 {failed}"),
        "judge_mode": "programmatic-only",
    }


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("judge 输出中未找到 JSON 对象")
    return json.loads(match.group(0))


def _fill_missing_dimensions(verdict: dict, case: dict) -> None:
    """宿主把裁判未输出的维度按用例冻结的适用性补全为 N/A。"""
    applicable = (case.get("judge_only") or {}).get("applicable") or {}
    have = {d.get("dimension_id") for d in verdict.get("dimension_scores") or []}
    filled = list(verdict.get("dimension_scores") or [])
    for d in ALL_DIMENSIONS:
        if d in have:
            continue
        filled.append({
            "dimension_id": d,
            "status": "not_applicable" if not applicable.get(d, {}).get("applicable", True) else "insufficient_evidence",
            "score": None, "evidence_ids": [],
            "rationale": "host-filled: judge 未输出",
        })
    verdict["dimension_scores"] = filled


def _validate_verdict(verdict: dict, case: dict) -> list[str]:
    problems = []
    if verdict.get("overall_verdict") not in _VERDICTS:
        problems.append("overall_verdict 非法")
    _fill_missing_dimensions(verdict, case)
    for item in verdict.get("dimension_scores") or []:
        if item.get("status") not in _SCORE_STATUSES:
            problems.append(f"{item.get('dimension_id')} status 非法")
        elif item["status"] == "scored" and item.get("score") not in (0, 1, 2, 3, 4):
            problems.append(f"{item.get('dimension_id')} 分值非法")
    if verdict.get("case_id") != case["case_id"]:
        problems.append("case_id 不匹配")
    return problems


def _compose_from_compact(compact: dict, case: dict, evidence: dict) -> dict:
    """把裁判的紧凑输出组装成完整 verdict 结构。

    程序断言直接映射为验收项结果（判定优先级：independent_checks 优先）；
    scores 中的 "na" / 缺失维度按用例冻结的适用性补全为 not_applicable。
    """
    applicable = (case.get("judge_only") or {}).get("applicable") or {}
    raw_scores = compact.get("scores") or {}
    dimension_scores = []
    for d in ALL_DIMENSIONS:
        meta = applicable.get(d) or {}
        is_applicable = meta.get("applicable", True)
        value = raw_scores.get(d)
        if is_applicable and isinstance(value, (int, float)) and 0 <= value <= 4:
            dimension_scores.append({"dimension_id": d, "status": "scored",
                                     "score": int(value), "evidence_ids": [],
                                     "rationale": str(compact.get("summary", ""))[:60]})
        else:
            dimension_scores.append({"dimension_id": d, "status": "not_applicable",
                                     "score": None, "evidence_ids": [],
                                     "rationale": "host-filled" if not is_applicable else "judge 未评分"})
    acceptance = [
        {"criterion_id": c["check_id"], "required": True,
         "result": "met" if c["status"] == "pass" else "unmet",
         "evidence_ids": [c["check_id"]], "rationale": c["actual"][:80]}
        for c in evidence.get("independent_checks") or []
    ]
    acceptance += [
        {"criterion_id": ac.get("criterion_id", "AC?"), "required": bool(ac.get("required", True)),
         "result": "unknown", "evidence_ids": [],
         "rationale": "语义项由 judge 汇总判定（见 summary）"}
        for ac in (case.get("judge_only") or {}).get("acceptance") or []
    ]
    overall = compact.get("overall_verdict")
    if overall not in _VERDICTS:
        overall = "inconclusive"
    task_success = compact.get("task_success")
    if not isinstance(task_success, bool):
        task_success = None
    issues = []
    for i, issue in enumerate((compact.get("issues") or [])[:3]):
        issues.append({"issue_id": f"I{i+1}", "severity": "minor",
                       "kind": "quality_defect", "responsible_component": ["main_agent"],
                       "evidence_ids": [], "description": str(issue)[:160], "suggested_fix": ""})
    return {
        "case_id": case["case_id"], "trial_id": evidence["trial_id"],
        "track": case["track"], "trial_validity": "valid", "validity_reason": "",
        "acceptance_results": acceptance, "gate_results": [],
        "dimension_scores": dimension_scores,
        "overall_verdict": overall, "task_success": task_success,
        "issues": issues, "missing_evidence": [],
        "confidence": "medium", "review_required": overall in ("inconclusive", "invalid"),
        "summary": str(compact.get("summary", ""))[:300],
    }


def run_judge(case: dict, evidence: dict, judge_provider, retry_limit: int = 3) -> dict:
    """调用裁判模型评审单个 trial；输出非法按固定上限重试并保留原结果。"""
    accepted_cases = case["judge_only"].get("acceptance") or []
    gates = case["judge_only"].get("gates") or []
    user_payload = {
        "schema_version": "1.0",
        "trusted_case": {
            "case_id": case["case_id"],
            "family_id": case["family"],
            "track": case["track"],
            "difficulty": case["difficulty"],
            "objective": case["judge_only"].get("objective"),
            "user_prompt": case["user_prompt"],
            "expected_behavior": case["judge_only"].get("expected_behavior"),
            "acceptance_criteria": accepted_cases,
            "critical_gates": gates,
            "applicable_dimensions": case["judge_only"].get("applicable"),
            "weights": case["weights"],
        },
        "environment_contract": {
            "sut": "FluxAgent (LangGraph StateGraph + 策略引擎 + office 沙盒 + help/run 凭据 + 委派)",
            "network": "disabled",
            "workdir": "HARNESS_TEMP_WORKSPACE",
        },
        "execution": {
            "trial_id": evidence["trial_id"],
            "stub": evidence["stub"],
            "events": evidence["events"],
            "final_answer": evidence["final_answer"],
            "tool_results": evidence["tool_messages"],
            "office_before": evidence["office_before"],
            "office_after": evidence["office_after"],
            "observed_metrics": evidence["observed_metrics"],
        },
        "independent_checks": evidence["independent_checks"],
    }
    from langchain_core.messages import HumanMessage, SystemMessage
    # 紧凑输出契约：裁判只产出 task_success/scores/summary，
    # 完整 12 维结构由宿主组装——把生成时延砍到最低（免费通道 ~1-3 tok/s）
    user_payload["compact_output"] = {
        "instruction": '只输出一个 JSON：{"task_success": true/false/null, '
                       '"overall_verdict": "pass|fail|inconclusive|invalid", '
                       '"scores": {"D01": 0-4 或 "na"}, '
                       '"summary": "不超过两句话", "issues": ["最多三条"]}',
    }
    attempts = []
    for attempt in range(retry_limit + 1):
        try:
            raw = judge_provider.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(user_payload, ensure_ascii=False)),
            ])
        except Exception as exc:
            # 传输层故障（网关 5xx/超时）：带退避重试，耗尽后降级而非抛出，
            # 保住已采集的 SUT 证据不整trial作废
            attempts.append({"attempt": attempt + 1, "problems": [f"transport: {exc}"]})
            if attempt < retry_limit:
                import time as _time
                _time.sleep(30 * (attempt + 1))
            continue
        content = raw.content if hasattr(raw, "content") else str(raw)
        try:
            verdict = _compose_from_compact(_extract_json(str(content)), case, evidence)
            problems = _validate_verdict(verdict, case)
        except (ValueError, json.JSONDecodeError) as exc:
            problems = [f"JSON 解析失败: {exc}"]
            verdict = {"overall_verdict": "inconclusive", "_raw": str(content)[:400]}
        attempts.append({"attempt": attempt + 1, "problems": problems})
        if not problems:
            verdict["judge_mode"] = "llm-judge"
            verdict["judge_attempts"] = attempts
            return verdict
    # 全部尝试失败（解析/传输）：输出可审计的 inconclusive 结论
    verdict = {
        "case_id": case["case_id"], "trial_id": evidence["trial_id"],
        "track": case["track"], "trial_validity": "evidence_limited",
        "validity_reason": f"judge failed: {attempts[-1]['problems'][0][:160]}",
        "overall_verdict": "inconclusive", "task_success": None,
        "judge_mode": "llm-judge", "judge_attempts": attempts,
        "review_required": True,
        "summary": "裁判调用失败（网络/解析），SUT 证据已保留待复评",
    }
    return verdict
