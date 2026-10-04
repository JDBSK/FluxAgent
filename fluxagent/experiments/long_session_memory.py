# -*- coding: utf-8 -*-
"""长会话记忆回归：60 轮域切换会话，逐轮断言记忆行为（0 API）。

针对"问答过长时记忆是问题高发区"的四个高发失效模式：
  F1 跨域经验泄漏：换域后其他域经验仍被注入；
  F2 经验记录堆积：同一稳定 key 不覆盖、越积越多；
  F3 token 预算失守：历史增长后规则/契约段被挤掉或总量超限；
  F4 敏感/过期复活：长会话后期敏感记录或 TTL 过期记录重新进入 prompt。

每个失效模式都有逐轮断言与违规计数，报告写入 artifacts/long_session_memory.json。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any

from fluxagent.capability.context_compiler import (
    CompilationContext,
    ContextCompiler,
    MemoryScope,
    MemoryStore,
)
from fluxagent.capability.descriptors import (
    CapabilityDescriptor,
    OperationDescriptor,
    RiskLevel,
)
from fluxagent.capability.registry import CapabilityRegistry
from fluxagent.capability.retrieval import DeterministicEmbedding
from fluxagent.capability.router import HybridRouter

TOTAL_TURNS = 60
SWITCH_EVERY = 5  # 每 5 轮切换一次域
DOMAIN_SEQUENCE = ["market", "cooking", "weather", "translate", "docs"] * 12
SESSION_ID = "long-session-A"


def build_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    for domain in ("market", "cooking", "weather", "translate", "docs"):
        capability_id = f"mcp.{domain}"
        op_name = f"mcp_{domain}_run"
        registry.register(
            CapabilityDescriptor(
                capability_id=capability_id,
                version="1",
                kind="mcp",
                name=domain,
                description=f"{domain} 域合成能力：处理{domain}相关请求",
                intents=(f"{domain}_intent",),
                operations=(
                    OperationDescriptor(
                        capability_id=capability_id,
                        name=op_name,
                        operation_id=f"{capability_id}/{op_name}",
                        description=f"{domain} 操作",
                        input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                        side_effect_level=RiskLevel.READ,
                        tool=object(),
                    ),
                ),
            )
        )
    return registry


def run(total_turns: int = TOTAL_TURNS) -> dict[str, Any]:
    registry = build_registry()
    snapshot = registry.snapshot()
    router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=128))
    router.rebuild_index(snapshot)
    store = MemoryStore()
    TOKEN_BUDGET = 2200
    compiler = ContextCompiler(store, token_budget=TOKEN_BUDGET)

    # 全局敏感记录：任何回合都不得出现
    store.add(
        MemoryScope.USER_FACT, "保险柜密码 0000", key="profile.secret",
        sensitivity="sensitive",
    )
    # TTL 过期记录（会话开始前就过期）
    store.add(
        MemoryScope.EPISODIC, "过期任务：远古清单", key="task.expired",
        ttl_seconds=1, now=1.0, session_id=SESSION_ID,
    )

    violations = {"F1_cross_domain_leak": 0, "F2_task_accumulation": 0,
                  "F3_budget_breach": 0, "F4_sensitive_or_expired_revival": 0}
    per_turn: list[dict] = []
    previous_capability: str | None = None
    history_digest = ""

    for turn in range(1, total_turns + 1):
        segment = (turn - 1) // SWITCH_EVERY
        domain = DOMAIN_SEQUENCE[segment % len(DOMAIN_SEQUENCE)]
        capability_id = f"mcp.{domain}"
        user_text = f"第{turn}轮：处理{domain}域请求"

        # 模拟真实 agent：每回合写当前域经验（同 key 覆盖）
        store.add(
            MemoryScope.EPISODIC,
            f"当前任务：{user_text}",
            key=f"task:{SESSION_ID}",
            source_id="agent_node",
            capability_tags=(capability_id,),
            ttl_seconds=3600,
            session_id=SESSION_ID,
        )
        # 模拟偶发工具失败写入 episodic（每 10 轮一次）
        if turn % 10 == 0:
            store.add(
                MemoryScope.EPISODIC,
                f"工具 mcp_{domain}_run 在第{turn}轮失败",
                key=f"episodic:{SESSION_ID}:{domain}:{turn}",
                capability_tags=(capability_id,),
                ttl_seconds=86400,
                session_id=SESSION_ID,
            )

        history_digest = "历史摘要" * (turn * 12)  # 历史快速膨胀，施压 token 预算
        compiled = compiler.compile(
            CompilationContext(
                user_text=user_text,
                route="route",
                intent=f"{domain}_intent",
                capability_ids=(capability_id,),
                memory_scopes=("user_fact", "episodic"),
                history_digest=history_digest,
                session_id=SESSION_ID,
            )
        )
        prompt = compiled.system_prompt

        turn_violations: list[str] = []
        # F1: 注入的记忆记录必须属于本域，
        # 全局记录（无 capability 标签）除外；prompt 子串作第二层
        for memory in compiled.memory:
            for other in ("market", "cooking", "weather", "translate", "docs"):
                if other != domain and f"mcp.{other}" in memory.reason:
                    turn_violations.append("F1_cross_domain_leak")
                    break
        for other in ("market", "cooking", "weather", "translate", "docs"):
            if other != domain and f"处理{other}域请求" in prompt:
                turn_violations.append("F1_cross_domain_leak")
        # F3: 预算与保底段
        if compiled.estimated_tokens > TOKEN_BUDGET or "Invocation contract" not in prompt:
            turn_violations.append("F3_budget_breach")
        # F4: 敏感/过期
        if "保险柜" in prompt or "远古清单" in prompt:
            turn_violations.append("F4_sensitive_or_expired_revival")

        for violation in turn_violations:
            violations[violation] += 1
        per_turn.append(
            {
                "turn": turn,
                "domain": domain,
                "tokens": compiled.estimated_tokens,
                "violations": turn_violations,
                "memory_records": len(compiled.memory),
            }
        )
        previous_capability = capability_id

    # F2: 记忆库有界性——task:{session} 同 key 覆盖，只能有 1 条
    # （episodic 按设计逐条累积，受 TTL 约束）
    task_records = [
        r for r in store.all_records()
        if r.scope == MemoryScope.EPISODIC and r.session_id == SESSION_ID
    ]
    if len(task_records) > 1:
        violations["F2_task_accumulation"] += len(task_records) - 1

    token_values = [t["tokens"] for t in per_turn]
    return {
        "total_turns": total_turns,
        "violations": violations,
        "all_passed": all(v == 0 for v in violations.values()),
        "task_records_in_store": len(task_records),
        "episodic_records_in_store": len([
            r for r in store.all_records() if r.scope == MemoryScope.EPISODIC
        ]),
        "token_stats": {
            "min": min(token_values),
            "max": max(token_values),
            "mean": round(statistics.mean(token_values), 1),
        },
        "per_turn": per_turn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="long-session memory regression")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    report = run()
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "long_session_memory.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(
        f"[long-session] {report['total_turns']} turns | "
        f"violations={report['violations']} | "
        f"tokens {report['token_stats']['min']}-{report['token_stats']['max']} | "
        f"task_records={report['task_records_in_store']} "
        f"episodic_records={report['episodic_records_in_store']}"
    )
    print(f"[long-session] report -> {out_path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
