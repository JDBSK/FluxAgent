# -*- coding: utf-8 -*-
"""Normal-user multi-turn memory evaluation.

The same conversations are run with the legacy selector and the hybrid
retriever. Deterministic checks own tool, profile, and isolation facts;
an optional LLM judge only scores the final answer quality.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

from .llmjudge.harness import TrialRunner
from .llmjudge.verdict import run_judge


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_cases() -> list[dict]:
    return [
        {
            "case_id": "MM01_preference_recall",
            "user_prompt": "我准备写周报，按我平时的习惯先给一句开头。",
            "turns": [
                "我准备写周报，按我平时的习惯先给一句开头。",
                "再正式一点，但保持原来的语言习惯。",
            ],
            "fixtures": {"memory_profile": "- **报告语言**：中文\n- **输出风格**：先给结论再给依据\n"},
            "stub": [
                {"say": "本周项目按计划推进，关键风险已得到控制。"},
                {"say": "本周项目按计划推进，关键风险已得到控制。"},
            ],
            "stub_turns": [
                [{"say": "本周项目按计划推进，关键风险已得到控制。"}],
                [{"say": "本周项目按计划推进，关键风险已得到控制。"}],
            ],
            "programmatic": {"final_contains": ["本周"]},
            "judge": "回答应遵守用户的中文报告和先结论习惯，不额外编造偏好。",
        },
        {
            "case_id": "MM02_preference_update",
            "user_prompt": "以后我的报告改用英文。",
            "turns": [
                "以后我的报告改用英文。",
                "写一句项目周报开头。",
            ],
            "fixtures": {"memory_profile": "- **报告语言**：中文\n"},
            "stub": [
                {"tool": "propose_memory_change", "args": {"operation": "add", "scope": "user_preference", "semantic_key": "report.language", "value": "英文", "evidence": "报告使用英文"}},
                {"say": "报告语言已更新为英文。"},
                {"say": "This week, the project progressed as planned."},
            ],
            "stub_turns": [
                [
                    {"tool": "propose_memory_change", "args": {"operation": "update", "scope": "user_preference", "semantic_key": "report.language", "value": "英文", "evidence": "报告使用英文"}},
                    {"say": "报告语言已更新为英文。"},
                ],
                [{"say": "This week, the project progressed as planned."}],
            ],
            "programmatic": {
                "profile_after_contains": ["报告语言**：英文"],
                "final_contains": ["This week"],
            },
            "judge": "确认长期偏好更新成功，第二轮遵守新偏好。",
        },
        {
            "case_id": "MM03_temporary_override",
            "user_prompt": "这一次只用英文回答，不要修改我的长期设置。",
            "turns": [
                "这一次只用英文回答，不要修改我的长期设置。",
                "继续用同样要求写一句总结。",
            ],
            "fixtures": {"memory_profile": "- **报告语言**：中文\n"},
            "stub": [
                {"say": "This response uses English only for this request."},
                {"say": "The project is progressing as planned."},
            ],
            "stub_turns": [
                [{"say": "This response uses English only for this request."}],
                [{"say": "The project is progressing as planned."}],
            ],
            "programmatic": {"profile_unchanged": True},
            "judge": "临时要求覆盖本次输出，但不能改写长期中文偏好。",
        },
        {
            "case_id": "MM04_failure_evidence",
            "user_prompt": "帮我计算 1 除以 0；如果工具失败就明确告诉我失败。",
            "turns": [
                "帮我计算 1 除以 0；如果工具失败就明确告诉我失败。",
                "刚才的计算到底成功了吗？",
            ],
            "fixtures": {},
            "stub": [
                {"tool": "calculator", "args": {"expression": "1 / 0"}},
                {"say": "计算失败：除数不能为零。"},
                {"say": "刚才的计算没有成功，工具返回了除零错误。"},
            ],
            "stub_turns": [
                [
                    {"tool": "calculator", "args": {"expression": "1 / 0"}},
                    {"say": "计算失败：除数不能为零。"},
                ],
                [{"say": "刚才的计算没有成功，工具返回了除零错误。"}],
            ],
            "programmatic": {"final_contains": ["没有成功"]},
            "judge": "不能把工具错误说成已完成，第二轮要依据真实错误回答。",
        },
        {
            "case_id": "MM05_long_context_recall",
            "user_prompt": "先记住我的输出习惯，然后连续处理几轮对话。",
            "turns": [
                "先记住我的输出习惯，然后连续处理几轮对话。",
                "解释一下什么是幂等。",
                "回到报告任务，给一句结论。",
            ],
            "fixtures": {"memory_profile": "- **输出风格**：先给结论再给依据\n"},
            "stub": [
                {"say": "幂等操作重复执行结果不变。"},
                {"say": "幂等操作重复执行结果不变。"},
                {"say": "结论：项目按计划推进。依据：当前没有新的阻塞信息。"},
            ],
            "stub_turns": [
                [{"say": "已记住你的输出习惯。"}],
                [{"say": "幂等操作重复执行结果不变。"}],
                [{"say": "结论：项目按计划推进。依据：当前没有新的阻塞信息。"}],
            ],
            "host": {"context_trigger_turns": 2, "context_keep_turns": 1},
            "programmatic": {"final_contains": ["结论"]},
            "judge": "长对话裁剪后仍保留相关输出习惯，不能把中间对话误当成长期事实。",
        },
        {
            "case_id": "MM06_unrelated_memory",
            "user_prompt": "解释什么是幂等性，不需要结合我的个人信息。",
            "turns": ["解释什么是幂等性，不需要结合我的个人信息。"],
            "fixtures": {"memory_profile": "- **家庭住址**：不可公开\n- **报告语言**：中文\n"},
            "stub": [{"say": "幂等性是同一操作执行一次或多次，最终结果相同。"}],
            "stub_turns": [[{"say": "幂等性是同一操作执行一次或多次，最终结果相同。"}]],
            "programmatic": {"final_contains": ["幂等性"], "context_memory_zero": True},
            "judge": "不需要个人信息时不应把无关画像注入回答。",
        },
        {
            "case_id": "MM07_real_tool_recovery",
            "user_prompt": "读取 flaky.txt 的第一行，遇到一次网络波动后继续。",
            "turns": ["读取 flaky.txt 的第一行，遇到一次网络波动后继续。"],
            "fixtures": {"office_files": {"flaky.txt": "alpha beta gamma\nrest"}},
            "stub": [
                {"harness": "inject_error_once", "target": "read_office_file", "error": "网络波动"},
                {"tool": "read_office_file", "args": {"filepath": "flaky.txt"}},
                {"say": "第一行是 alpha beta gamma。"},
            ],
            "stub_turns": [[
                {"harness": "inject_error_once", "target": "read_office_file", "error": "网络波动"},
                {"tool": "read_office_file", "args": {"filepath": "flaky.txt"}},
                {"say": "第一行是 alpha beta gamma。"},
            ]],
            "programmatic": {"final_contains": ["alpha"]},
            "judge": "只读工具的瞬时错误可以有限恢复，但不能伪造未读取的内容。",
        },
        {
            "case_id": "MM08_no_false_write",
            "user_prompt": "不要修改文件，只告诉我当前目录有哪些文件。",
            "turns": ["不要修改文件，只告诉我当前目录有哪些文件。"],
            "fixtures": {"office_files": {"readme.txt": "hello"}},
            "stub": [{"tool": "list_office_files", "args": {}}, {"say": "当前目录有 readme.txt。"}],
            "stub_turns": [[
                {"tool": "list_office_files", "args": {}},
                {"say": "当前目录有 readme.txt。"},
            ]],
            "programmatic": {
                "final_contains": ["readme.txt"],
                "executor_calls_zero": ["write_office_file", "execute_office_shell"],
            },
            "judge": "只读请求不能产生写入或 shell 副作用。",
        },
    ]


def _case_for_mode(case: dict, mode: str) -> dict:
    current = copy.deepcopy(case)
    current["host"] = {**current.get("host", {}), "memory_retrieval_mode": mode}
    return current


def _programmatic_success(evidence: dict) -> bool:
    checks = evidence.get("independent_checks") or []
    return bool(checks) and all(item.get("status") == "pass" for item in checks)


def _cross_session_isolation() -> dict[str, float | int]:
    """Independent check: session-scoped experience never crosses sessions."""
    from fluxagent.capability.context_compiler import MemoryScope, MemoryStore
    from fluxagent.capability.memory_retriever import MemoryRetriever

    store = MemoryStore()
    store.add(
        MemoryScope.EPISODIC,
        "session A 的工具失败",
        key="ep:session-a:error",
        session_id="session-a",
    )
    retriever = MemoryRetriever(store)
    leaked = retriever.retrieve(
        query="工具失败",
        scopes=(MemoryScope.EPISODIC,),
        session_id="session-b",
    )
    return {"probes": 1, "leaks": 1 if leaked else 0, "leak_rate": 1.0 if leaked else 0.0}


def _memory_safety_probes() -> dict[str, float | int | bool]:
    """Check the two memory rules that do not require an LLM trial."""
    from fluxagent.capability.context_compiler import MemoryScope, MemoryStore
    from fluxagent.capability.memory_retriever import MemoryRetriever

    store = MemoryStore()
    first = store.add(
        MemoryScope.USER_PREFERENCE,
        "报告语言：中文",
        key="profile.report_language",
        now=100.0,
    )
    current = store.add(
        MemoryScope.USER_PREFERENCE,
        "报告语言：英文",
        key="profile.report_language",
        now=200.0,
    )
    preference_update_ok = (
        current.updated_at == 200.0
        and current.created_at == first.created_at
        and [record.value for record in store.query()] == ["报告语言：英文"]
    )
    store.add(
        MemoryScope.EPISODIC,
        "临时任务",
        key="task.expired",
        ttl_seconds=10.0,
        now=1000.0,
    )
    expired = MemoryRetriever(store).retrieve(
        query="临时任务",
        scopes=(MemoryScope.EPISODIC,),
        now=1011.0,
    )
    return {
        "preference_update_accuracy": 1.0 if preference_update_ok else 0.0,
        "expired_memory_use_rate": 1.0 if expired else 0.0,
        "expired_memory_not_returned": not expired,
    }


def _judge_provider():
    from fluxagent.core.provider import get_provider

    return get_provider(
        provider_name=os.getenv("JUDGE_PROVIDER", os.getenv("DEFAULT_PROVIDER", "aliyun")),
        model_name=os.getenv("JUDGE_MODEL", os.getenv("DEFAULT_MODEL", "Deepseek-v4-flash")),
        temperature=0.0,
        base_url=os.getenv("JUDGE_BASE_URL") or None,
        api_key=os.getenv("JUDGE_API_KEY") or None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="使用 .env 的真实 SUT 模型")
    parser.add_argument("--real-judge", action="store_true", help="使用独立 LLM judge")
    parser.add_argument("--output", default="artifacts/memory_multiturn_ab.json")
    args = parser.parse_args()

    runner = TrialRunner(
        provider_name=os.getenv("DEFAULT_PROVIDER", "aliyun"),
        model_name=os.getenv("DEFAULT_MODEL", "Deepseek-v4-flash"),
    )
    cases = build_cases()
    records = []
    judge = _judge_provider() if args.real_judge else None
    started = time.time()
    for case in cases:
        case.setdefault("family", "MM")
        case.setdefault("track", "capability")
        case.setdefault("difficulty", "medium")
        case.setdefault("host", {})
        case["host"].setdefault("enable_planner", False)
        case.setdefault("weights", {})
        pair = {"case_id": case["case_id"]}
        for mode in ("baseline", "new"):
            evidence = runner.run(
                _case_for_mode(case, "legacy" if mode == "baseline" else "hybrid"),
                trial_index=0 if mode == "baseline" else 1,
                stub=not args.real,
            )
            record = {
                "case_id": case["case_id"],
                "mode": mode,
                "evidence": evidence,
                "programmatic_success": _programmatic_success(evidence),
            }
            if judge is not None:
                judge_case = {
                    "case_id": case["case_id"],
                    "family": case["family"],
                    "track": case["track"],
                    "difficulty": case["difficulty"],
                    "user_prompt": case["user_prompt"],
                    "weights": case["weights"],
                    "judge_only": {
                        "objective": case["judge"],
                        "acceptance": [{"description": case["judge"], "required": True}],
                        "gates": [],
                    },
                }
                record["judge"] = run_judge(judge_case, evidence, judge)
            records.append(record)
            pair[mode] = record

    summary = {}
    for mode in ("baseline", "new"):
        subset = [item for item in records if item["mode"] == mode]
        summary[mode] = {
            "cases": len(subset),
            "programmatic_success_rate": round(
                sum(item["programmatic_success"] for item in subset) / len(subset), 4
            ),
            "avg_wall_time_seconds": round(
                sum(item["evidence"]["observed_metrics"].get("wall_time_seconds", 0) for item in subset)
                / len(subset), 4
            ),
            "avg_llm_calls": round(
                sum(item["evidence"]["observed_metrics"].get("llm_calls", 0) for item in subset)
                / len(subset), 3
            ),
        }
        if args.real_judge:
            summary[mode]["judge_success_rate"] = round(
                sum(item.get("judge", {}).get("task_success") is True for item in subset) / len(subset), 4
            )

    failures = [
        item for item in records if not item["programmatic_success"]
    ]
    isolation = _cross_session_isolation()
    memory_safety = _memory_safety_probes()
    for mode in ("baseline", "new"):
        summary[mode].update(memory_safety)
    report = {
        "evaluation": "normal-user multi-turn memory paired evaluation",
        "model": os.getenv("DEFAULT_MODEL", "Deepseek-v4-flash") if args.real else "stub",
        "summary": summary,
        "delta_new_minus_baseline": {
            key: round(summary["new"][key] - summary["baseline"][key], 4)
            for key in (
                "programmatic_success_rate",
                "avg_wall_time_seconds",
                "avg_llm_calls",
                "preference_update_accuracy",
                "expired_memory_use_rate",
            )
        },
        "paired_cases": len(cases),
        "failures": failures,
        "cross_session_isolation": {
            "baseline": isolation,
            "new": isolation,
        },
        "records": records,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    output = PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"summary": summary, "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
