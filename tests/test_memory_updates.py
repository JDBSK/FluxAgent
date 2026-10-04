from __future__ import annotations

import os
import tempfile
import unittest

from fluxagent.capability.memory_events import MemoryEventLog, MemoryProjector
from fluxagent.capability.memory_store import MemoryScope, MemoryStore, ProfileCompatProvider, normalize_profile_key
from fluxagent.core.agent import (
    _is_explicitly_one_time_request,
    _is_transient_mood_or_craving,
    _validate_memory_change,
)


class TestMemoryChangeValidation(unittest.TestCase):
    def test_accepts_unseen_domain_key_when_evidence_is_current_user_text(self):
        user = "以后买咖啡豆优先选浅烘焙，记住这个习惯。"
        self.assertEqual(_validate_memory_change({
            "operation": "add", "scope": "user_preference",
            "semantic_key": "shopping.coffee.roast", "value": "优先选择浅烘焙咖啡豆",
            "evidence": "以后买咖啡豆优先选浅烘焙",
        }, user), "")

    def test_rejects_evidence_not_in_latest_user_message(self):
        reason = _validate_memory_change({
            "operation": "update", "scope": "user_preference",
            "semantic_key": "report.language", "value": "中文",
            "evidence": "用户一直使用英文",
        }, "这次请用中文回复。")
        self.assertIn("本轮用户原话", reason)

    def test_rejects_invalid_scope_and_operation(self):
        proposal = {"operation": "update", "scope": "task_state", "semantic_key": "report.language", "value": "中文", "evidence": "这次请用中文"}
        self.assertTrue(_validate_memory_change(proposal, "这次请用中文回复。"))
        proposal.update(scope="user_preference", operation="unknown")
        self.assertTrue(_validate_memory_change(proposal, "这次请用中文回复。"))

    def test_rejects_explicit_one_time_format_request_before_persistence(self):
        user = "这次的答案请用表格，之后不用默认这样。"
        proposal = {
            "operation": "add",
            "scope": "user_preference",
            "semantic_key": "answer_format_preference",
            "value": "默认使用表格",
            "evidence": user,
        }
        self.assertTrue(_is_explicitly_one_time_request(user))
        self.assertIn("只对本次生效", _validate_memory_change(proposal, user))

    def test_allows_explicit_persistent_preference(self):
        user = "这次报告用中文，之后都默认使用中文。"
        proposal = {
            "operation": "add",
            "scope": "user_preference",
            "semantic_key": "report.language",
            "value": "中文",
            "evidence": user,
        }
        self.assertFalse(_is_explicitly_one_time_request(user))
        self.assertEqual(_validate_memory_change(proposal, user), "")

    def test_rejects_short_term_mood_or_craving(self):
        user = "这两天突然很想喝奶茶。"
        proposal = {
            "operation": "add",
            "scope": "user_preference",
            "semantic_key": "preference.drink.milk_tea",
            "value": "用户最近想喝奶茶",
            "evidence": "这两天突然很想喝奶茶",
        }
        self.assertTrue(_is_transient_mood_or_craving(user))
        self.assertIn("短期情绪或冲动", _validate_memory_change(proposal, user))

    def test_persistent_marker_overrides_short_term_wording(self):
        user = "最近开始长期坚持每周游泳两次，请记住。"
        self.assertFalse(_is_transient_mood_or_craving(user))


class TestLegacyProfileMigration(unittest.TestCase):
    def test_migration_preserves_distinct_fields_and_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = os.path.join(tmp, "user_profile.md")
            with open(profile, "w", encoding="utf-8") as handle:
                handle.write(chr(10).join(["- **报告语言**：中文", "- **汇报语言**：英文", ""]))
            store = MemoryStore(os.path.join(tmp, "structured_memory.json"))
            importer = ProfileCompatProvider(profile)
            self.assertEqual(importer.migrate_once(store), 2)
            self.assertEqual(importer.migrate_once(store), 0)
            self.assertEqual(len(store.all_records()), 2)
            self.assertTrue(all(record.scope == MemoryScope.USER_FACT for record in store.all_records()))
            self.assertEqual({r.value for r in store.all_records()}, {"报告语言：中文", "汇报语言：英文"})
            self.assertTrue(os.path.isfile(profile + ".migrated.bak"))

    def test_key_normalization_does_not_merge_synonyms(self):
        self.assertNotEqual(normalize_profile_key("报告语言"), normalize_profile_key("输出语言"))


class TestMemoryChangeProjection(unittest.TestCase):
    def test_add_update_delete_are_durable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "memory_events.sqlite3"))
            store = MemoryStore(os.path.join(tmp, "structured_memory.json"))
            projector = MemoryProjector(log, store)
            common = {"scope": "user_preference", "semantic_key": "shopping.coffee.roast", "evidence": "以后咖啡豆选浅烘焙"}
            first = log.append("memory_change", {**common, "operation": "add", "value": "浅烘焙"}, session_id="session-a")
            self.assertTrue(projector.process_one())
            self.assertFalse(projector.apply(first))
            self.assertEqual(store.all_records()[0].record_id, "user_preference:profile.shopping.coffee.roast")
            log.append("memory_change", {**common, "operation": "update", "value": "中烘焙"}, session_id="session-a")
            self.assertTrue(projector.process_one())
            self.assertEqual(len(store.all_records()), 1)
            self.assertEqual(store.all_records()[0].value, "中烘焙")
            log.append("memory_change", {**common, "operation": "delete", "value": ""}, session_id="session-a")
            self.assertTrue(projector.process_one())
            self.assertEqual(store.all_records(), [])


if __name__ == "__main__":
    unittest.main()
