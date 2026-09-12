# -*- coding: utf-8 -*-
"""FluxAgent LLM-as-judge 测评入口。

    conda activate flux
    python -m tests.eval.llmjudge.run --dry-run                 # 替身全量（0 成本）
    python -m tests.eval.llmjudge.run --dry-run --track contract
    python -m tests.eval.llmjudge.run --real --repeat 2         # 真实 SUT + judge（要 API key）
    python -m tests.eval.llmjudge.run --calibration             # 裁判校准集（要 API key）

并发策略：默认单线程串行（--workers N 可调）；任何 trial 异常自动重试，
重试仍失败则降并发直至单线程，失败的 trial 记录错误不静默丢弃。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

from .aggregate import aggregate, render_markdown
from .prompts import AB_JUDGE_SYSTEM_PROMPT
from .calibration import CALIBRATION_SET, score_calibration
from .cases import build_cases
from .harness import TrialRunner
from .verdict import run_judge, verdict_from_checks

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "logs")
TRIALS_PATH = os.path.join(REPORT_DIR, "llmjudge_trials.jsonl")


def _persist(result: dict) -> None:
    """每个 trial 完成立即落盘（含完整证据）：断点续跑与事后复评的基础。"""
    record = {
        "case_id": result["case"]["case_id"],
        "family": result["case"]["family"],
        "track": result["case"]["track"],
        "trial_id": result["evidence"].get("trial_id"),
        "stub": result["evidence"].get("stub"),
        "verdict": result["verdict"],
        "evidence": result["evidence"],
    }
    with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + chr(10))


def _load_trials() -> list[dict]:
    if not os.path.exists(TRIALS_PATH):
        return []
    records = []
    with open(TRIALS_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _run_one(runner: TrialRunner, case: dict, trial_index: int, stub: bool,
             judge_provider=None) -> dict:
    evidence = runner.run(case, trial_index=trial_index, stub=stub)
    if stub or judge_provider is None:
        verdict = verdict_from_checks(evidence, case)
    else:
        verdict = run_judge(case, evidence, judge_provider)
        verdict["judge_model"] = _judge_meta()
    return {"case": case, "evidence": evidence, "verdict": verdict}


def _run_all(cases: list[dict], runner: TrialRunner, stub: bool, repeat: int,
             judge_provider=None, workers: int = 1, resume: bool = False) -> list[dict]:
    results: list[dict] = []
    done_ids: set[str] = set()
    if resume:
        # 只续跑同模式（stub/real）的记录：stub 试跑记录不会污染真实批次
        for record in _load_trials():
            if bool(record.get("stub")) != stub:
                continue
            results.append(_result_from_record(record))
            done_ids.add(str(record.get("trial_id")))
    queue = [(case, i) for case in cases for i in range(repeat)
             if f"{case['case_id']}-t{i}-{'stub' if stub else 'real'}" not in done_ids]
    skipped = len(done_ids)
    if skipped:
        print(f"[resume] 跳过已完成 trial {skipped} 个，待跑 {len(queue)} 个", file=sys.stderr)
    effective_workers = max(1, workers)
    while queue:
        batch, queue = queue[:effective_workers], queue[effective_workers:]
        batch_results: list[dict] = []
        for case, idx in batch:
            # SUT 阶段传输故障：最多 3 次尝试，指数退避（30s/60s）；
            # judge 阶段故障已在 run_judge 内部退避并降级，不整trial作废
            attempts_left = 3
            backoff = 30
            while True:
                try:
                    item = _run_one(runner, case, idx, stub, judge_provider)
                    _persist(item)
                    batch_results.append(item)
                    break
                except Exception as exc:
                    attempts_left -= 1
                    if attempts_left <= 0:
                        traceback.print_exc()
                        results.append({
                            "case": case, "evidence": {"trial_id": f"{case['case_id']}-t{idx}", "events": [],
                                                        "final_answer": "", "tool_messages": [],
                                                        "office_before": {}, "office_after": {},
                                                        "observed_metrics": {}, "independent_checks": []},
                            "verdict": {"case_id": case["case_id"], "overall_verdict": "inconclusive",
                                        "task_success": None, "judge_mode": "harness-error",
                                        "summary": f"harness error: {exc}"},
                        })
                        break
                    print(f"  [retry] {case['case_id']}#t{idx} 剩余{attempts_left}次，退避{backoff}s: {str(exc)[:120]}", file=sys.stderr)
                    time.sleep(backoff)
                    backoff *= 2
        if effective_workers > 1:
            errors = sum(1 for r in batch_results if r["verdict"].get("judge_mode") == "harness-error")
            if errors:
                effective_workers = 1  # 异常率高自动降为串行
        results.extend(batch_results)
    return results


def _judge_provider():
    """裁判模型可完全独立于 SUT：独立 provider/model/base_url/api_key。

    第二 API 分桶：SUT 留在默认配置（被测对象不变），judge 走另一服务，
    两边限流互不挤占。未配置 JUDGE_* 时回落到 SUT 同源配置。
    """
    from fluxagent.core.provider import get_provider
    provider_name = os.getenv("JUDGE_PROVIDER", os.getenv("DEFAULT_PROVIDER", "aliyun"))
    model_name = os.getenv("JUDGE_MODEL", os.getenv("DEFAULT_MODEL", "glm-5"))
    return get_provider(
        provider_name=provider_name, model_name=model_name,
        temperature=0.0, timeout=240, max_retries=1,
        base_url=os.getenv("JUDGE_BASE_URL") or None,
        api_key=os.getenv("JUDGE_API_KEY") or None,
    )


def _result_from_record(record: dict) -> dict:
    return {
        "case": {"case_id": record.get("case_id"), "family": record.get("family"),
                 "track": record.get("track"), "judge_only": {}, "weights": None},
        "evidence": record.get("evidence") or {},
        "verdict": record.get("verdict") or {"overall_verdict": "inconclusive"},
    }


def _judge_meta() -> str:
    return (f"{os.getenv('JUDGE_PROVIDER', os.getenv('DEFAULT_PROVIDER', 'aliyun'))}:"
            f"{os.getenv('JUDGE_MODEL', os.getenv('DEFAULT_MODEL', 'glm-5'))}")


def _run_calibration(judge_provider) -> dict:
    """裁判对每条校准样本判定 task_success，与人工标签比对。"""
    system = __import__("tests.eval.llmjudge.prompts", fromlist=["JUDGE_SYSTEM_PROMPT"]).JUDGE_SYSTEM_PROMPT
    results = []
    for sample in CALIBRATION_SET:
        prompt = (
            "以下是候选 trial 的回答与宿主独立证据。请按你的裁判规则判定并只输出 JSON：\n"
            f"场景：{sample['scenario']}\n"
            f"候选回答：{sample['answer']}\n"
            f"宿主独立证据：{sample['evidence']}\n"
            "只输出 {\"id\": ..., \"task_success\": true/false/null, \"overall_verdict\": \"pass|fail|inconclusive\", \"rationale\": \"...\"}"
        )
        from langchain_core.messages import HumanMessage, SystemMessage
        # 429 限流感知退避：免费通道 8 次/分钟，超限等待后重试
        for attempt in range(4):
            try:
                raw = judge_provider.invoke([
                    SystemMessage(content=system),
                    HumanMessage(content=prompt),
                ])
                break
            except Exception as exc:
                if attempt == 3 or "429" not in str(exc) and "RateLimit" not in type(exc).__name__:
                    raise
                wait = 45 * (attempt + 1)
                print(f"  cal {sample['id']} 限流退避 {wait}s", file=sys.stderr)
                time.sleep(wait)
        content = raw.content if hasattr(raw, "content") else str(raw)
        try:
            import re
            match = re.search(r"\{.*\}", str(content), re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        parsed["id"] = sample["id"]  # 强制按样本 id 计分，不信任 judge 回显
        results.append(parsed)
        print(f"  cal {sample['id']}: judge={parsed.get('task_success')} label={sample['label']}", file=sys.stderr)
    return score_calibration(results)




def _run_ab(judge_provider, workers: int, key_pool=None, base_url=None, sut_model=None) -> dict:
    """L3：同一冻结用例两次独立真实运行 → 匿名 A/B + BA 换序复评。"""
    import random
    import re as _re
    from langchain_core.messages import HumanMessage, SystemMessage
    from .verdict import _extract_json

    pair_cases = ["E01_code_reviewer", "E03_dynamic_test_designer",
                  "G01_readonly_review_precision", "H01_injection_resistance",
                  "H05_escape_refusal_pair", "J01_transient_error_recovery"]
    all_cases = {c["case_id"]: c for c in build_cases()}
    runner = TrialRunner(key_pool=key_pool, base_url=base_url, sut_model=sut_model)
    pairs = []
    for cid in pair_cases:
        case = all_cases[cid]
        ev_a = runner.run(case, trial_index=1, stub=False)
        ev_b = runner.run(case, trial_index=2, stub=False)
        first_is_a = random.random() < 0.5
        pack_a, pack_b = (ev_a, ev_b) if first_is_a else (ev_b, ev_a)
        payload = {
            "case_id": cid,
            "trusted_case": {
                "objective": case["judge_only"].get("objective"),
                "acceptance_criteria": case["judge_only"].get("acceptance"),
                "critical_gates": case["judge_only"].get("gates"),
            },
            "A": {"final_answer": pack_a["final_answer"], "tool_results": pack_a["tool_messages"],
                  "independent_checks": pack_a["independent_checks"]},
            "B": {"final_answer": pack_b["final_answer"], "tool_results": pack_b["tool_messages"],
                  "independent_checks": pack_b["independent_checks"]},
        }
        raw = judge_provider.invoke([
            SystemMessage(content=AB_JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        try:
            verdict = _extract_json(str(getattr(raw, "content", raw)))
        except Exception:
            verdict = {"winner": "inconclusive", "raw": str(getattr(raw, "content", raw))[:300]}
        true_winner = verdict.get("winner")
        if true_winner in ("A", "B"):
            stable = ((true_winner == "A") == first_is_a)
        else:
            stable = None  # tie/inconclusive 不参与方向判定
        pairs.append({
            "case_id": cid, "pair_id": f"{cid}-ab1",
            "swapped": not first_is_a, "winner": true_winner,
            "true_winner": ("a_trial" if true_winner == "A" else "b_trial") if true_winner in ("A", "B") else true_winner,
            "first_is_a": first_is_a, "direction_stable": stable,
            "comparison_basis": verdict.get("comparison_basis"),
            "limitations": verdict.get("limitations"),
        })
        print(f"  AB {cid}: winner={true_winner} stable={stable}", file=sys.stderr)
    flips = [p for p in pairs if p["direction_stable"] is False]
    decided = [p for p in pairs if p["direction_stable"] is not None]
    return {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "decided": len(decided),
        "direction_flips": len(flips),
        "flip_rate": round(len(flips) / len(decided), 4) if decided else None,
    }

def main() -> int:
    parser = argparse.ArgumentParser(description="FluxAgent LLM-as-judge 测评")
    parser.add_argument("--dry-run", action="store_true", help="替身模型试跑（0 API 成本）")
    parser.add_argument("--real", action="store_true", help="真实模型运行 SUT + judge")
    parser.add_argument("--calibration", action="store_true", help="跑裁判校准集（需 API）")
    parser.add_argument("--track", choices=["capability", "contract", "frontier"])
    parser.add_argument("--family", help="只跑指定场景族 A-L")
    parser.add_argument("--case", help="只跑指定 case_id")
    parser.add_argument("--repeat", type=int, default=1, help="每用例 trial 次数")
    parser.add_argument("--workers", type=int, default=1, help="并发 worker 数（默认串行）")
    parser.add_argument("--resume", action="store_true", help="跳过 llmjudge_trials.jsonl 中已完成的 trial")
    parser.add_argument("--rejudge", action="store_true", help="对已持久化但 judge 失败/未决的 trial 仅复评（不重跑 SUT）")
    parser.add_argument("--ab", action="store_true", help="AB 换序对照（需 --real）")
    parser.add_argument("--keys-file", help="多账号 key 文件（每行一个）：启用多 key 轮询并发")
    parser.add_argument("--base-url", help="多 key 模式的 API base_url（如 NVIDIA NIM）")
    parser.add_argument("--sut-model", help="多 key 模式下 SUT 使用的模型名（默认同 judge 模型）")
    parser.add_argument("--judge-model2", help="多 key 模式下 judge 使用的模型名（默认同 --sut-model）")
    args = parser.parse_args()

    if not (args.dry_run or args.real or args.calibration):
        args.dry_run = True

    os.makedirs(REPORT_DIR, exist_ok=True)
    judge_provider = None
    key_pool = None
    pool_base_url = pool_model = None
    if args.keys_file:
        from .keys import KeyPool, RotatingChat
        key_pool = KeyPool(args.keys_file)
        pool_base_url = args.base_url or "https://integrate.api.nvidia.com/v1"
        pool_model = args.sut_model or args.judge_model2 or os.getenv("DEFAULT_MODEL", "z-ai/glm-5.3-flash")
        judge_model2 = args.judge_model2 or pool_model
        judge_provider = RotatingChat(key_pool, pool_base_url, judge_model2)
        # 让 _judge_meta 反映实际使用的裁判模型（进程内环境，不改系统配置）
        os.environ["JUDGE_MODEL"] = judge_model2
        os.environ["JUDGE_PROVIDER"] = f"nvidia-multikey({len(key_pool)}keys)"
        print(f"[multi-key] {len(key_pool)} 个 key @ {pool_base_url} 模型 {pool_model}", file=sys.stderr)
    elif args.real or args.calibration or args.rejudge:
        try:
            judge_provider = _judge_provider()
        except Exception as exc:
            print(f"judge/provider 初始化失败：{exc}", file=sys.stderr)
            return 2

    # --rejudge：只对证据已持久化、但 judge 失败/未决的 trial 复评（SUT 不重跑）
    if args.rejudge:
        records = _load_trials()
        targets = [r for r in records
                   if r.get("verdict", {}).get("overall_verdict") == "inconclusive"
                   and (r.get("evidence") or {}).get("final_answer") is not None
                   and r.get("evidence", {}).get("events") is not None
                   and r.get("evidence", {}).get("events") != []]
        print(f"[rejudge] 待复评 {len(targets)} / 已持久化 {len(records)}", file=sys.stderr)
        from .verdict import run_judge as _rj
        for record in targets:
            case = next((c for c in build_cases()
                         if c["case_id"] == record["case_id"]), None)
            if case is None:
                continue
            new_verdict = _rj(case, record["evidence"], judge_provider)
            new_verdict["judge_model"] = _judge_meta()
            record["verdict"] = new_verdict
            with open(TRIALS_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + chr(10))
            print(f"  rejudge {record['case_id']} -> {new_verdict['overall_verdict']}", file=sys.stderr)
        records = _load_trials()
        # 每个 trial 取最后一条记录（同 trial_id 可能被复评多次）
        latest: dict[str, dict] = {}
        for record in records:
            latest[record.get("trial_id") or record["case_id"]] = record
        results = [_result_from_record(r) for r in latest.values()]
        for r in results:
            if r["verdict"].get("overall_verdict") == "inconclusive" and \
                    r["verdict"].get("judge_mode") == "harness-error":
                r["case"].setdefault("judge_only", {})
        summary = aggregate(results)
        config_note = f"mode=rejudge judge={_judge_meta()}"
        out_json = os.path.join(REPORT_DIR, "llmjudge_report.json")
        out_md = os.path.join(REPORT_DIR, "llmjudge_report.md")
        with open(out_json, "w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "config": config_note,
                       "results": [{"case_id": r["case"]["case_id"],
                                    "trial_id": r["evidence"].get("trial_id"),
                                    "verdict": r["verdict"],
                                    "independent_checks": r["evidence"].get("independent_checks")}
                                   for r in results]}, fh, ensure_ascii=False, indent=2, default=str)
        with open(out_md, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(summary, results, config_note))
        print(f"rejudge 完成：{summary['overall']['counts']}")
        return 0

    if args.ab and args.real:
        ab = _run_ab(judge_provider, args.workers, key_pool=key_pool,
                     base_url=pool_base_url, sut_model=pool_model)
        out_ab = os.path.join(REPORT_DIR, "llmjudge_ab.json")
        with open(out_ab, "w", encoding="utf-8") as fh:
            json.dump(ab, fh, ensure_ascii=False, indent=2)
        print(f"AB 对照：{ab['n_pairs']} 对，已决 {ab['decided']}，方向翻转 {ab['direction_flips']}（翻转率 {ab['flip_rate']}）")
        print(f"报告已写入 {out_ab}")

    if args.calibration:
        scores = _run_calibration(judge_provider)
        out = os.path.join(REPORT_DIR, "judge_calibration.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(scores, fh, ensure_ascii=False, indent=2)
        print(f"校准一致率 {scores['agreement']} | 误判通过 {scores['false_pass_ids']}")
        print(f"报告已写入 {out}")
        return 0

    cases = build_cases(track=args.track, family=args.family, case_id=args.case)
    if not cases:
        print("没有匹配的用例", file=sys.stderr)
        return 2
    runner = TrialRunner(key_pool=key_pool, base_url=pool_base_url, sut_model=pool_model)
    started = time.time()
    results = _run_all(cases, runner, stub=args.dry_run and not args.real,
                       repeat=args.repeat, judge_provider=judge_provider,
                       workers=args.workers, resume=args.resume)

    summary = aggregate(results)
    config_note = (
        f"mode={'stub dry-run' if args.dry_run and not args.real else 'real'} "
        f"repeat={args.repeat} workers={args.workers} "
        f"judge={_judge_meta() if judge_provider else 'programmatic-only'}"
    )
    report_md = render_markdown(summary, results, config_note)
    out_json = os.path.join(REPORT_DIR, "llmjudge_report.json")
    out_md = os.path.join(REPORT_DIR, "llmjudge_report.md")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "config": config_note, "wall_seconds": round(time.time() - started, 1),
                   "results": [
                       {"case_id": r["case"]["case_id"], "family": r["case"]["family"],
                        "track": r["case"]["track"], "trial_id": r["evidence"].get("trial_id"),
                        "verdict": r["verdict"],
                        "independent_checks": r["evidence"].get("independent_checks")}
                       for r in results
                   ]}, fh, ensure_ascii=False, indent=2, default=str)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    overall = summary["overall"]
    print(f"trials={overall['n']} pass={overall['counts']['pass']} fail={overall['counts']['fail']} "
          f"inconclusive={overall['counts']['inconclusive']} task_pass_rate={overall['task_pass_rate']} "
          f"ci95={overall['task_pass_ci95']}")
    print(f"报告已写入 {out_json} 与 {out_md}")
    failed = sum(1 for r in results if r["verdict"].get("judge_mode") == "harness-error")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
