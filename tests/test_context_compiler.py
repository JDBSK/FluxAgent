# -*- coding: utf-8 -*-
"""Context Compiler 与选择性记忆测试。全部使用合成数据。"""
from __future__ import annotations

import os
import tempfile
import unittest

from fluxagent.capability.context_compiler import (
    CompilationContext,
    ContextCompiler,
    MemoryScope,
    MemoryStore,
    ProfileCompatProvider,
    estimate_tokens,
)


def store_with_facts() -> MemoryStore:
    store = MemoryStore()
    store.add(
        MemoryScope.USER_FACT,
        "家庭人数：3人",
        key="profile.household",
        source_id="user_profile.md#家庭人数",
    )
    store.add(
        MemoryScope.USER_FACT,
        "过敏原：对虾过敏",
        key="profile.allergy",
        source_id="user_profile.md#过敏原",
    )
    store.add(
        MemoryScope.USER_PREFERENCE,
        "最终报告只能用中文输出",
        key="profile.language",
        source_id="user_profile.md#语言",
    )
    store.add(
        MemoryScope.USER_FACT,
        "项目密码本位置：保险柜",
        key="profile.secret",
        source_id="user_profile.md#密码",
        sensitivity="sensitive",
    )
    return store


class TestMemoryStore(unittest.TestCase):
    def test_scope_and_sensitivity_filtering(self):
        store = store_with_facts()
        normal = store.query(scopes=[MemoryScope.USER_FACT])
        values = [r.value for r in normal]
        self.assertIn("家庭人数：3人", values)
        self.assertNotIn("项目密码本位置：保险柜", values)
        sensitive = store.query(
            scopes=[MemoryScope.USER_FACT], include_sensitive=True
        )
        self.assertIn("项目密码本位置：保险柜", [r.value for r in sensitive])

    def test_ttl_expiry(self):
        store = MemoryStore()
        store.add(
            MemoryScope.TASK_STATE,
            "临时状态",
            key="task.tmp",
            ttl_seconds=10,
            now=1000.0,
        )
        # TTL 10s：1005s 未过期，1015s 已过期
        self.assertEqual(len(store.query(now=1005.0)), 1)
        self.assertEqual(len(store.query(now=1015.0)), 0)
        # 无 TTL 记录不受影响
        store.add(MemoryScope.USER_FACT, "长期事实", now=1000.0)
        self.assertEqual(len(store.query(now=99999.0)), 1)

    def test_key_supersedes_old_value(self):
        store = MemoryStore()
        store.add(MemoryScope.USER_FACT, "家庭人数：2人", key="profile.household")
        store.add(MemoryScope.USER_FACT, "家庭人数：4人", key="profile.household")
        records = store.query(scopes=[MemoryScope.USER_FACT])
        values = [r.value for r in records]
        self.assertIn("家庭人数：4人", values)
        self.assertNotIn("家庭人数：2人", values)

    def test_capability_tag_filtering(self):
        store = MemoryStore()
        store.add(
            MemoryScope.USER_FACT, "忌口香菜", capability_tags=("cap.cooking",)
        )
        store.add(
            MemoryScope.USER_FACT, "偏好看盘", capability_tags=("cap.market",)
        )
        cooking = store.query(capability_id="cap.cooking")
        self.assertEqual([r.value for r in cooking], ["忌口香菜"])

    def test_profile_compat_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "user_profile.md")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("- **家庭人数**：3人\n- **饮食忌口**：不吃香菜\n非格式行\n")
            store = MemoryStore()
            count = ProfileCompatProvider(path).load_into(store)
            self.assertEqual(count, 2)
            values = {r.value for r in store.query(scopes=[MemoryScope.USER_FACT])}
            self.assertIn("家庭人数：3人", values)
            self.assertIn("饮食忌口：不吃香菜", values)


class TestContextCompiler(unittest.TestCase):
    def test_relevant_memory_injected_with_reason(self):
        compiler = ContextCompiler(store_with_facts(), token_budget=4000)
        snapshot = compiler.compile(
            CompilationContext(
                user_text="给我推荐今天的菜谱",
                route="route",
                intent="recommend_meal",
                capability_ids=("cap.cooking",),
                memory_scopes=("user_fact", "user_preference"),
            )
        )
        values = [m.value for m in snapshot.memory]
        self.assertIn("家庭人数：3人", values)
        self.assertIn("过敏原：对虾过敏", values)
        self.assertIn("最终报告只能用中文输出", values)
        reasons = {m.record_id: m.reason for m in snapshot.memory}
        self.assertIn("scope=user_fact", reasons["profile.household"])
        self.assertNotIn("项目密码本位置：保险柜", snapshot.system_prompt)

    def test_sensitive_memory_excluded_by_default(self):
        compiler = ContextCompiler(store_with_facts(), token_budget=4000)
        snapshot = compiler.compile(
            CompilationContext(user_text="什么都行", memory_scopes=("user_fact",))
        )
        self.assertNotIn("保险柜", snapshot.system_prompt)

    def test_unrelated_capability_memory_excluded(self):
        store = store_with_facts()
        store.add(
            MemoryScope.TASK_STATE,
            "正在分析茅台K线",
            key="task.market",
            capability_tags=("cap.market",),
        )
        compiler = ContextCompiler(store, token_budget=4000)
        snapshot = compiler.compile(
            CompilationContext(
                user_text="推荐菜谱",
                route="route",
                intent="recommend_meal",
                capability_ids=("cap.cooking",),
                memory_scopes=("task_state",),
            )
        )
        self.assertNotIn("茅台", snapshot.system_prompt)
        # 换到市场能力后出现
        snapshot2 = compiler.compile(
            CompilationContext(
                user_text="看盘",
                route="route",
                intent="quote_lookup",
                capability_ids=("cap.market",),
                memory_scopes=("task_state",),
            )
        )
        self.assertIn("茅台", snapshot2.system_prompt)

    def test_no_tool_route_contains_contract_directive(self):
        compiler = ContextCompiler(store_with_facts(), token_budget=4000)
        snapshot = compiler.compile(
            CompilationContext(user_text="1加1等于几", route="answer_without_tools")
        )
        self.assertIn("route=answer_without_tools", snapshot.system_prompt)
        self.assertIn("未授权任何外部工具", snapshot.system_prompt)

    def test_token_budget_truncates_low_priority_sections(self):
        store = MemoryStore()
        for i in range(50):
            store.add(MemoryScope.USER_FACT, f"事实{i}: " + "长" * 80)
        compiler = ContextCompiler(store, token_budget=300)
        snapshot = compiler.compile(
            CompilationContext(
                user_text="随便聊聊",
                history_digest="历史摘要" * 200,
            )
        )
        self.assertLessEqual(snapshot.estimated_tokens, 300)
        # 规则与契约永不裁剪
        self.assertIn("Invocation contract", snapshot.system_prompt)
        self.assertIn("Rules", snapshot.system_prompt)
        self.assertTrue(snapshot.truncated_sections)

    def test_skill_context_budgeted(self):
        compiler = ContextCompiler(store_with_facts(), token_budget=4000)
        big_skill = "角色说明。" * 800
        snapshot = compiler.compile(
            CompilationContext(
                user_text="用skill回答",
                route="route",
                intent="style",
                skill_context=big_skill,
            )
        )
        self.assertIn("[Selected skill context]", snapshot.system_prompt)
        self.assertIn("[截断]", snapshot.system_prompt)
        self.assertIn("skill_context", snapshot.truncated_sections)

    def test_failure_facts_included(self):
        compiler = ContextCompiler(store_with_facts(), token_budget=4000)
        snapshot = compiler.compile(
            CompilationContext(
                user_text="重试一下",
                route="route",
                failure_facts=("工具X参数schema校验失败一次，本回合禁止重试",),
            )
        )
        self.assertIn("禁止重试", snapshot.system_prompt)

    def test_estimate_tokens_cjk_vs_ascii(self):
        # CJK 每字 1 token
        self.assertEqual(estimate_tokens("测试"), 2)
        # ASCII 4 字符 1 token（不足 4 也按 1 计）
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertGreater(estimate_tokens("中文内容比英文密度高"), 5)


if __name__ == "__main__":
    unittest.main()
