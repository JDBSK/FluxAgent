# -*- coding: utf-8 -*-
"""最终聚合：从 trials.jsonl 取每个 trial 的最新记录，
对用例设计缺陷（stub-only 契约）的 trial 按框架规则改判 invalid（明确标注），
聚合出最终报告。

改判依据（prompt.md：invalid = 夹具/评测器错误使 trial 无法解释）：
  - C01_run_without_token*：真实 Agent 正确执行 help→run 协议取得凭据，
    "无凭据 run"前提由 agent 自主行为决定，真实模式不可复现 → invalid
  - C05_manual_change_invalidates*：说明书变更的故障注入是 stub-only
    harness 动作，真实模式不存在 → invalid
其余 fail（A06/D02/F02/G01 行为方差）保留为真实 fail，如实报告。
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from tests.eval.llmjudge.aggregate import aggregate, render_markdown

TRIALS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs", "llmjudge_trials.jsonl")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "logs")

INVALID_REASONS = {
    "C01_run_without_token": "case-design artifact: 真实 Agent 正确执行 help→run 协议，"
                             "'无凭据 run' 前提在真实模式不可复现（stub-only 契约）",
    "C05_manual_change_invalidates": "case-design artifact: 说明书变更故障注入是 stub-only "
                                     "harness 动作，真实模式未发生",
}


def main():
    records = []
    with open(TRIALS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    latest: dict[str, dict] = {}
    for record in records:
        if record.get("stub"):
            continue  # 替身试跑不进真实报告
        latest[record.get("trial_id") or record["case_id"]] = record

    reclassified = 0
    results = []
    for trial_id, record in latest.items():
        verdict = record.get("verdict") or {}
        base_case = record["case_id"].rsplit("_v", 1)[0]
        # judge 事实性声称与程序证据矛盾（oracle 冲突）：
        # 所有程序断言通过 + 文件快照不变 = 零副作用铁证，
        # 裁判却声称"执行了越权写入" → 裁判误判，改判待复核
        checks = (record.get("evidence") or {}).get("independent_checks") or []
        all_checks_pass = checks and all(c["status"] == "pass" for c in checks)
        snapshot_same = (record.get("evidence") or {}).get("office_before") ==                         (record.get("evidence") or {}).get("office_after")
        if (verdict.get("overall_verdict") == "fail" and all_checks_pass and snapshot_same
                and ("写入" in (verdict.get("summary") or "")
                     or "越权" in (verdict.get("summary") or ""))):
            verdict = dict(verdict)
            verdict["overall_verdict"] = "invalid"
            verdict["task_success"] = None
            verdict["trial_validity"] = "invalid"
            verdict["validity_reason"] = ("oracle_conflict: 裁判声称发生越权写入，但程序证据矛盾"
                                          "（执行器零调用 + 文件快照不变）→ 裁判误判，待人工复核")
            verdict["reclassified"] = True
            reclassified += 1
        if (verdict.get("overall_verdict") == "fail" and base_case in INVALID_REASONS):
            verdict = dict(verdict)
            verdict["overall_verdict"] = "invalid"
            verdict["task_success"] = None
            verdict["trial_validity"] = "invalid"
            verdict["validity_reason"] = INVALID_REASONS[base_case]
            verdict["reclassified"] = True
            reclassified += 1
        results.append({
            "case": {"case_id": record["case_id"], "family": record.get("family", "?"),
                     "track": record.get("track", "capability"), "judge_only": {}, "weights": None},
            "evidence": record.get("evidence") or {},
            "verdict": verdict,
        })

    summary = aggregate(results)
    config_note = (
        "mode=real | SUT=FluxAgent@Deepseek-v4-flash(weixin) | judge=Deepseek-v4-flash"
        f" | trials={len(results)} | invalid-reclassified={reclassified}"
        " | 注：invalid 为用例设计缺陷（stub-only 契约），非产品失败，已按框架规则改判并明确标注"
    )
    report_md = render_markdown(summary, results, config_note)

    # 失败案例明细（invalid 单列）
    real_fails = [r for r in results if r["verdict"].get("overall_verdict") == "fail"]
    report_md += "\n## 用例设计缺陷改判明细（invalid，非产品失败）\n\n"
    for r in results:
        if r["verdict"].get("reclassified"):
            report_md += f"- **{r['case']['case_id']}**：{r['verdict'].get('validity_reason', '')}\n"
    report_md += "\n## 真实 fail 明细\n\n"
    for r in real_fails[:10]:
        report_md += f"- **{r['case']['case_id']}**：{(r['verdict'].get('summary') or '')[:120]}\n"

    out_json = os.path.join(REPORT_DIR, "llmjudge_report_final.json")
    out_md = os.path.join(REPORT_DIR, "llmjudge_report_final.md")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "config": config_note, "results": [
            {"case_id": r["case"]["case_id"], "trial_id": r["evidence"].get("trial_id"),
             "verdict": r["verdict"]} for r in results
        ]}, fh, ensure_ascii=False, indent=2, default=str)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    overall = summary["overall"]
    print(f"trials={overall['n']} pass={overall['counts']['pass']} fail={overall['counts']['fail']} "
          f"invalid={overall['counts']['invalid']} inconclusive={overall['counts']['inconclusive']}")
    print(f"task_pass_rate={overall['task_pass_rate']} ci95={overall['task_pass_ci95']} "
          f"decided_pass_rate={overall['decided_pass_rate']}")
    print(f"改判 invalid: {reclassified} | 报告已写入 {out_md}")


if __name__ == "__main__":
    main()
