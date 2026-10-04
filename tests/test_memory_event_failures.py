from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
import unittest
from contextlib import closing
from unittest.mock import patch

from fluxagent.capability.memory_events import MemoryEventLog, MemoryProjector
from fluxagent.capability.memory_store import MemoryScope, MemoryStore


class TestMemoryEventFailures(unittest.TestCase):
    def test_event_and_job_append_rollback_together(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "events.sqlite3")
            log = MemoryEventLog(path)
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute("CREATE TRIGGER reject_job BEFORE INSERT ON memory_jobs BEGIN SELECT RAISE(ABORT, 'injected job failure'); END")
            with self.assertRaises(sqlite3.IntegrityError):
                log.append("memory_change", {"operation": "add"})
            self.assertEqual(log.iter_events(), [])
            self.assertEqual(log.pending_count(), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_snapshot_failure_keeps_event_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "events.sqlite3"), retry_base_seconds=0.01)
            store = MemoryStore(os.path.join(tmp, "structured_memory.json"))
            projector = MemoryProjector(log, store)
            event = log.append("memory_change", {
                "operation": "add", "scope": "user_fact", "semantic_key": "work.role",
                "value": "工程师", "evidence": "我是工程师",
            })
            with patch("fluxagent.capability.memory_store._atomic_write_json", side_effect=OSError("disk busy")):
                self.assertFalse(projector.process_one())
            self.assertFalse(log.is_applied(event.event_id))
            self.assertEqual(log.pending_count(), 1)
            time.sleep(0.02)
            self.assertTrue(projector.process_one())
            self.assertTrue(log.is_applied(event.event_id))
            records = store.query(scopes=[MemoryScope.USER_FACT])
            self.assertEqual([record.value for record in records], ["工程师"])

    def test_event_committed_before_worker_exit_is_reclaimed_after_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.sqlite3")
            log = MemoryEventLog(path, retry_base_seconds=0.01)
            event = log.append(
                "memory_change",
                {
                    "operation": "add", "scope": "user_fact",
                    "semantic_key": "work.role", "value": "工程师",
                    "evidence": "我是工程师",
                },
            )
            claimed = log.claim(lock_seconds=0.01)
            self.assertIsNotNone(claimed)
            del log
            time.sleep(0.02)

            reopened = MemoryEventLog(path, retry_base_seconds=0.01)
            store = MemoryStore()
            projector = MemoryProjector(reopened, store)
            self.assertTrue(projector.process_one())
            self.assertTrue(reopened.is_applied(event.event_id))
            self.assertEqual([record.value for record in store.all_records()], ["工程师"])

    def test_projection_crash_after_store_write_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "events.sqlite3"), retry_base_seconds=0.01)
            store = MemoryStore()
            projector = MemoryProjector(log, store)
            event = log.append("memory_change", {
                "operation": "add", "scope": "user_fact",
                "semantic_key": "work.role", "value": "工程师",
                "evidence": "我是工程师",
            })
            with patch.object(log, "mark_applied", side_effect=RuntimeError("worker crashed")):
                self.assertFalse(projector.process_one())
            self.assertFalse(log.is_applied(event.event_id))
            time.sleep(0.02)
            self.assertTrue(projector.process_one())
            self.assertTrue(log.is_applied(event.event_id))
            records = store.all_records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].value, "工程师")


if __name__ == "__main__":
    unittest.main()
