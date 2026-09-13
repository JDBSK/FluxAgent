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


class TestMemoryStorePersistence(unittest.TestCase):
    """重启恢复：持久化往返、TTL 不复活、会话隔离保留、坏文件 fail-open。"""

    def _make_store(self, path):
        store = MemoryStore(persist_path=path)
        store.add(
            MemoryScope.USER_FACT, "家庭人数：3人", key="profile.household"
        )
        store.add(
            MemoryScope.TASK_STATE,
            "当前任务：分析股票行情",
            key="task.s1",
            capability_tags=("cap.market",),
            ttl_seconds=3600,
            session_id="session-a",
        )
        return store

    def test_round_trip_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            store = self._make_store(path)
            self.assertTrue(os.path.exists(path))
            restored = MemoryStore(persist_path=path)
            self.assertEqual(restored.load(path), 2)
            values = {
                r.value for r in restored.query(scopes=[MemoryScope.USER_FACT])
            }
            self.assertEqual(values, {"家庭人数：3人"})
            # 会话级记录按原 session 边界可见
            scoped = restored.query(
                scopes=[MemoryScope.TASK_STATE],
                capability_id="cap.market",
                session_id="session-a",
            )
            self.assertEqual(
                [r.value for r in scoped], ["当前任务：分析股票行情"]
            )

    def test_expired_records_not_revived(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            store = MemoryStore(persist_path=path)
            store.add(
                MemoryScope.TASK_STATE,
                "早已过期",
                key="task.old",
                ttl_seconds=100,
                now=1000.0,
            )
            store.add(MemoryScope.USER_FACT, "长期事实", now=1000.0)
            restored = MemoryStore()
            self.assertEqual(restored.load(path), 1)
            self.assertEqual(
                [r.value for r in restored.query()], ["长期事实"]
            )

    def test_session_isolation_preserved_after_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            self._make_store(path)
            restored = MemoryStore(persist_path=path)
            restored.load(path)
            # 全局记录任何会话可见；会话 A 的 task_state 不泄漏给会话 B，
            # 无会话上下文（session_id=None）时一律排除（fail-closed）
            self.assertEqual(
                [r.value for r in restored.query(session_id="session-b")],
                ["家庭人数：3人"],
            )
            self.assertEqual(
                len(restored.query(scopes=[MemoryScope.TASK_STATE], session_id="session-b")),
                0,
            )
            self.assertEqual(len(restored.query(session_id=None)), 1)

    def test_corrupt_file_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("{not valid json")
            store = MemoryStore(persist_path=path)
            self.assertEqual(store.load(path), 0)
            # 坏文件不阻断使用；下一次变更自动写出好快照
            store.add(MemoryScope.USER_FACT, "恢复后写入")
            again = MemoryStore()
            self.assertEqual(again.load(path), 1)

    def test_format_mismatch_rejected(self):
        import json as _json

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            with open(path, "w", encoding="utf-8") as fh:
                _json.dump({"format": "999", "records": []}, fh)
            self.assertEqual(MemoryStore().load(path), 0)

    def test_auto_save_on_forget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            store = self._make_store(path)
            store.forget("user_fact:profile.household")
            again = MemoryStore()
            again.load(path)
            # 全局画像记录已被删除；会话级 task_state 仍在原会话边界内
            self.assertEqual([r.value for r in again.query()], [])
            self.assertEqual(
                [
                    r.value
                    for r in again.query(
                        capability_id="cap.market", session_id="session-a"
                    )
                ],
                ["当前任务：分析股票行情"],
            )

    def test_counter_high_water_no_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "structured_memory.json")
            store = MemoryStore(persist_path=path)
            first = store.add(MemoryScope.USER_FACT, "无键记录一")
            self.assertEqual(first.record_id, "mem-1")
            restored = MemoryStore(persist_path=path)
            restored.load(path)
            second = restored.add(MemoryScope.USER_FACT, "无键记录二")
            self.assertEqual(second.record_id, "mem-2")
            self.assertEqual(len(restored.query()), 2)

    def test_persist_error_recorded_not_raised(self):
        # 父路径是普通文件时 makedirs 失败（OSError）：落盘失败被吞掉并记录，
        # 不阻断 add 调用方
        with tempfile.TemporaryDirectory() as tmp:
            not_a_dir = os.path.join(tmp, "not_a_dir")
            with open(not_a_dir, "w", encoding="utf-8") as fh:
                fh.write("file, not directory")
            path = os.path.join(not_a_dir, "nested.json")
            store = MemoryStore(persist_path=path)
            record = store.add(MemoryScope.USER_FACT, "写入会失败")
            self.assertIsNotNone(record)
            self.assertIn("Error", store.last_persist_error)


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
        self.assertIn("scope=user_fact", reasons["user_fact:profile.household"])
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
        self.assertIn("未授权任何外部能力工具", snapshot.system_prompt)

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
