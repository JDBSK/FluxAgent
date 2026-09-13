# -*- coding: utf-8 -*-
"""Judge 稳定性复评与校准种子集。

1. 稳定性：对多轮对比的 16 条 transcript 各独立评审 2 次，报告一致率
   （单转录裁判没有配对换序问题，位置偏见主要出现在成对比较中；
   这里测的是同一输入的重复判定稳定性）。
2. 校准种子：按确定性结果推导临时标注（工具正确执行→期望 pass；
   污染/错配→期望 fail），标记为"待人工复核"——人工标注后即可迁移
   CyberClaw prompt.md 00-07 的校准流程。

结果写入 artifacts/judge_stability.json。
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def main() -> int:
    from fluxagent.experiments.post_change_eval import judge_transcripts

    source_path = os.path.join(PROJECT_ROOT, "artifacts", "compare_multi_new.json")
    with open(source_path, encoding="utf-8") as fh:
        report = json.load(fh)

    transcripts = [
        {
            "case": f"multi-{t['session']}-T{t['turn']}",
            "user": t["user"],
            "final_text": t.get("final_text", "") or "(空回答)",
            "expected_tools": sorted(t.get("expected", [])),
            "executed_tools": sorted(t.get("executed", [])),
            "wrong_tool_hits": sorted(t.get("wrong_tool_hits", [])),
        }
        for t in report.get("turns", [])
    ]

    provider_name = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model_name = os.getenv("DEFAULT_MODEL", "Deepseek-v4-flash")

    # 两次独立评审
    pass1 = judge_transcripts(provider_name, model_name, transcripts)
    pass2 = judge_transcripts(provider_name, model_name, transcripts)

    verdict1 = {v["case"]: v["verdict"] for v in pass1}
    verdict2 = {v["case"]: v["verdict"] for v in pass2}
    agreement = sum(1 for case in verdict1 if verdict1[case] == verdict2.get(case))
    agreement_rate = round(agreement / len(verdict1), 4) if verdict1 else None

    # 校准种子：临时标注由确定性结果推导，待人工复核
    calibration_seed = []
    for t in transcripts:
        case = t["case"]
        if t["wrong_tool_hits"]:
            provisional = "fail"
            basis = "确定性证据：调用了禁止工具（错配）"
        elif t["expected_tools"] and all(
            tool in t["executed_tools"] for tool in t["expected_tools"]
        ):
            provisional = "pass"
            basis = "确定性证据：期望工具全部正确执行"
        else:
            provisional = "pass"
            basis = "确定性证据：无工具需求且无越权调用"
        calibration_seed.append(
            {
                "case": case,
                "user": t["user"],
                "final_text": t["final_text"][:1500],
                "provisional_label": provisional,
                "label_basis": basis,
                "needs_human_review": True,
            }
        )

    out = {
        "model": model_name,
        "provider": provider_name,
        "stability": {
            "transcripts": len(transcripts),
            "agreement_rate": agreement_rate,
            "disagreements": [
                {"case": case, "pass1": verdict1[case], "pass2": verdict2.get(case)}
                for case in verdict1
                if verdict1[case] != verdict2.get(case)
            ],
        },
        "verdicts_pass1": pass1,
        "verdicts_pass2": pass2,
        "calibration_seed": {
            "note": "临时标注由确定性结果推导，需人工复核后方可作为裁判校准集",
            "items": calibration_seed,
        },
    }
    out_path = os.path.join(PROJECT_ROOT, "artifacts", "judge_stability.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(
        f"[judge-stability] agreement={agreement_rate} over {len(verdict1)} transcripts; "
        f"calibration seed: {len(calibration_seed)} items (provisional); report -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
