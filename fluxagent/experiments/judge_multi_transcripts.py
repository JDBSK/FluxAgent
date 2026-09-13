# -*- coding: utf-8 -*-
"""多轮对比 transcript 的 LLM-as-judge 扩充评审。

读取 artifacts/compare_multi_new.json 的 16 轮 transcript，逐条送 judge
（与 post_change_eval.judge_transcripts 同一 rubric），扩充 judge 样本量。
结果写入 artifacts/compare_multi_judge.json。
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


def main() -> int:
    from fluxagent.experiments.post_change_eval import judge_transcripts

    transcripts_path = os.path.join(PROJECT_ROOT, "artifacts", "compare_multi_new.json")
    with open(transcripts_path, encoding="utf-8") as fh:
        report = json.load(fh)

    transcripts = [
        {"case": f"multi-{t['session']}-T{t['turn']}", "user": t["user"], "final_text": t.get("final_text", "") or "(空回答)"}
        for t in report.get("turns", [])
    ]
    provider_name = os.getenv("DEFAULT_PROVIDER", "aliyun")
    model_name = os.getenv("DEFAULT_MODEL", "Deepseek-v4-flash")
    verdicts = judge_transcripts(provider_name, model_name, transcripts)

    out_path = os.path.join(PROJECT_ROOT, "artifacts", "compare_multi_judge.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": model_name,
                "provider": provider_name,
                "judge_count": len(verdicts),
                "verdicts": verdicts,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    passed = sum(1 for v in verdicts if v["verdict"] == "pass")
    print(f"[multi-judge] {passed}/{len(verdicts)} pass; report -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
