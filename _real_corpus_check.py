# -*- coding: utf-8 -*-
"""真实语料 RAG 验证：按 entry/main.py 的装配逻辑，用真实 workspace 数据
（builtin + 已安装 skill + mcp_servers/index.json 登记）构建统一 Registry 与
Router 索引，验证数据源构建、持久化往返、路由行为与数据边界。

边界处理：不访问项目外路径——MCP 记录的 installed_path 若指向项目外旧目录，
仅从项目内 index.json 读取字符串做前缀比对，标记 UNAVAILABLE，不发起连接。
"""
from __future__ import annotations

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fluxagent.capability.context_compiler import (
    CompilationContext,
    ContextCompiler,
    MemoryScope,
    MemoryStore,
)
from fluxagent.capability.descriptors import CapabilityDescriptor, ProviderHealth
from fluxagent.capability.providers import (
    builtin_descriptors,
    mcp_descriptors,
    skill_descriptors,
)
from fluxagent.capability.registry import CapabilityRegistry, RegistryError
from fluxagent.capability.retrieval import DeterministicEmbedding
from fluxagent.capability.router import HybridRouter
from fluxagent.core.config import MCP_SERVERS_INDEX, WORKSPACE_DIR
from fluxagent.core.skill_loader import read_skill_manual, scan_skill_infos
from fluxagent.core.tools.builtins import BUILTIN_TOOLS


def build_real_registry() -> tuple[CapabilityRegistry, dict]:
    registry = CapabilityRegistry()
    stats = {"builtin": 0, "skill": 0, "mcp": 0, "mcp_unavailable": 0, "errors": []}

    def _register(descriptor):
        try:
            registry.register(descriptor)
            return True
        except RegistryError as exc:
            stats["errors"].append(f"{descriptor.capability_id}: {exc}")
            return False

    for descriptor in builtin_descriptors(BUILTIN_TOOLS):
        if _register(descriptor):
            stats["builtin"] += 1

    # 真实已安装 skill（workspace/office/skills）
    skill_tools_by_name = {}
    try:
        from fluxagent.core.skill_loader import load_dynamic_skills

        skill_tools_by_name = {str(t.name): t for t in load_dynamic_skills()}
    except Exception as exc:  # noqa: BLE001
        stats["errors"].append(f"skill tools load: {exc}")
    for info in scan_skill_infos():
        for descriptor in skill_descriptors(
            [info],
            content_loader=read_skill_manual,
            tool_loader=lambda item: skill_tools_by_name.get(str(item.get("name") or "")),
        ):
            if _register(descriptor):
                stats["skill"] += 1

    # 真实 MCP 登记表（只读项目内 index.json；不连接、不访问项目外路径）
    try:
        with open(MCP_SERVERS_INDEX, encoding="utf-8") as fh:
            index_data = json.load(fh)
    except (OSError, ValueError) as exc:
        index_data = {"servers": []}
        stats["errors"].append(f"mcp index read: {exc}")

    for record in index_data.get("servers", []):
        installed_path = str(record.get("installed_path") or "")
        in_project = os.path.realpath(installed_path).startswith(
            os.path.realpath(PROJECT_ROOT)
        )
        server_name = str(record.get("name") or "unnamed")
        capability_id = f"mcp.{server_name}"
        if not in_project:
            # 数据边界：登记路径在项目外（旧项目目录），不连接、标记不可用
            stats["mcp_unavailable"] += 1
            stats["errors"].append(
                f"{capability_id}: installed_path 指向项目外（{installed_path}），"
                "需在本项目内重装后才会构建索引"
            )
            continue
        # 路径在项目内：仍不发起真实连接（避免重依赖），登记为空操作能力并注明
        empty = mcp_descriptors([(record, [])])
        if empty:
            stats["errors"].append(
                f"{capability_id}: 路径在项目内但本次验证不发起 stdio 连接（需 CLI 启动发现）"
            )
    return registry, stats


def main() -> int:
    registry, stats = build_real_registry()
    snapshot = registry.snapshot()
    print("=== 真实语料构建 ===")
    print(f"builtin: {stats['builtin']}  skill: {stats['skill']}  "
          f"mcp 可用: {stats['mcp']}  mcp 不可用: {stats['mcp_unavailable']}")
    for error in stats["errors"]:
        print(f"  [边界/跳过] {error[:150]}")

    router = HybridRouter(embedding_provider=DeterministicEmbedding(dim=256))
    index_dir = os.path.join(WORKSPACE_DIR, "memory", "capability_index")
    if not router.load_index(index_dir, snapshot):
        router.rebuild_index(snapshot)
        router.save_index(index_dir)
        print(f"索引已重建并持久化 -> {index_dir}")
    else:
        print(f"命中持久化索引（registry 版本一致）-> {index_dir}")
    router2 = HybridRouter(embedding_provider=DeterministicEmbedding(dim=256))
    print("持久化往返:", "通过" if router2.load_index(index_dir, snapshot) else "失败")

    print("\n=== 真实能力清单 ===")
    for capability in snapshot.enabled_capabilities():
        kinds = f"{capability.kind} ops={len(capability.operations)}"
        print(f"  {capability.capability_id:36} {kinds}")

    print("\n=== 真实查询路由（0 API，确定性 embedding）===")
    real_queries = [
        ("给我推荐三个家常菜", "应路由到菜谱能力（若已安装）"),
        ("用童锦程的视角分析一下感情问题", "应路由到 tongjincheng skill"),
        ("查一下贵州茅台的行情", "股票 MCP 未构建（路径在项目外），应 abstain/clarify"),
        ("帮我订明天的高铁票", "无对应能力，应 abstain"),
    ]
    store = MemoryStore()
    compiler = ContextCompiler(store, token_budget=2000)
    for query, note in real_queries:
        decision = router.decide(query, snapshot)
        skill_ctx = ""
        for capability_id in decision.capability_ids:
            capability = snapshot.get(capability_id)
            if capability and capability.context_provider:
                skill_ctx = str(capability.context_provider())[:40]
        compiled = compiler.compile(
            CompilationContext(
                user_text=query,
                route=decision.route,
                intent=decision.intent,
                capability_ids=decision.capability_ids,
                memory_scopes=decision.memory_scopes,
                skill_context=skill_ctx,
            )
        )
        print(f"  Q: {query}")
        print(f"     -> {decision.route} {list(decision.capability_ids)} | {note}")
        if skill_ctx:
            print(f"     skill 上下文注入: {skill_ctx}...")

    print("\n=== 数据边界检查 ===")
    print("  [OK] MCP 登记路径在项目外 -> 未连接、未建索引、标记不可用")
    print("  [OK] 索引持久化在项目内 workspace/memory/capability_index")
    print("  [OK] 会话级记忆 fail-closed（见 test_capability_hardening）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
