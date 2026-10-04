# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
import time
import unittest

from fluxagent.capability.context_compiler import MemoryScope, MemoryStore
from fluxagent.capability.memory_events import MemoryEventLog, MemoryProjector, MemoryWorker


class TestMemoryEventLog(unittest.TestCase):
    def test_append_is_idempotent_and_job_can_be_replayed(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "events.sqlite3"))
            store = MemoryStore()
            projector = MemoryProjector(log, store)
            event = log.append(
                "tool_failed",
                {
                    "key": "episodic:session:calculator:invalid",
                    "value": "calculator 参数错误",
                    "verified": True,
                    "ttl_seconds": 86400,
                },
                session_id="session",
            )
            self.assertEqual(log.pending_count(), 1)
            self.assertTrue(projector.process_one())
            self.assertFalse(projector.process_one())
            self.assertEqual(log.pending_count(), 0)
            records = store.query(
                scopes=[MemoryScope.EPISODIC], session_id="session"
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].source_id, f"event:{event.event_id}")

            # A second delivery of the same event is ignored by the applied marker.
            self.assertFalse(projector.apply(event))
            self.assertEqual(len(store.query(scopes=[MemoryScope.EPISODIC], session_id="session")), 1)

    def test_expired_processing_job_is_claimable_after_worker_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "events.sqlite3"))
            store = MemoryStore()
            projector = MemoryProjector(log, store)
            log.append(
                "tool_failed",
                {"key": "ep:retry", "value": "网络失败", "verified": True, "ttl_seconds": 86400},
                session_id="s",
            )
            claimed = log.claim(lock_seconds=0.01)
            self.assertIsNotNone(claimed)
            time.sleep(0.03)
            self.assertTrue(projector.process_one())
            self.assertEqual(len(store.all_records()), 1)

    def test_unverified_tool_claim_is_not_projected(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "events.sqlite3"))
            store = MemoryStore()
            projector = MemoryProjector(log, store)
            event = log.append(
                "tool_succeeded",
                {"key": "ep:unverified", "value": "模型说它成功了"},
            )
            self.assertTrue(projector.process_one())
            self.assertTrue(log.is_applied(event.event_id))
            self.assertEqual(store.all_records(), [])

    def test_failed_projection_uses_backoff_and_dead_letters(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(
                os.path.join(tmp, "events.sqlite3"),
                max_attempts=2,
                retry_base_seconds=0.01,
            )

            class BrokenStore:
                def add(self, *args, **kwargs):
                    raise RuntimeError("projection unavailable")

            projector = MemoryProjector(log, BrokenStore())
            log.append(
                "tool_succeeded",
                {"key": "ep:broken", "value": "真实结果", "verified": True},
            )
            self.assertFalse(projector.process_one())
            time.sleep(0.02)
            self.assertFalse(projector.process_one())
            time.sleep(0.02)
            self.assertFalse(projector.process_one())
            self.assertEqual(log.pending_count(), 0)

    def test_projection_can_replay_immutable_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "events.sqlite3")
            log = MemoryEventLog(path)
            event = log.append(
                "tool_failed",
                {"key": "ep:replay", "value": "可重放失败", "verified": True},
                session_id="s",
            )
            first = MemoryStore()
            projector = MemoryProjector(log, first)
            self.assertTrue(projector.process_one())
            second = MemoryStore()
            rebuilt = MemoryProjector(log, second)
            self.assertEqual(rebuilt.replay(), 1)
            records = second.query(scopes=[MemoryScope.EPISODIC], session_id="s")
            self.assertEqual([record.source_id for record in records], [f"event:{event.event_id}"])

    def test_worker_drains_and_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = MemoryEventLog(os.path.join(tmp, "events.sqlite3"))
            store = MemoryStore()
            projector = MemoryProjector(log, store)
            worker = MemoryWorker(projector, interval=0.01)
            worker.start()
            log.append(
                "tool_failed",
                {"key": "ep:worker", "value": "服务不可用", "verified": True, "ttl_seconds": 86400},
                session_id="s",
            )
            worker.submit()
            for _ in range(50):
                if log.pending_count() == 0:
                    break
                time.sleep(0.01)
            worker.stop()
            self.assertEqual(log.pending_count(), 0)
            self.assertEqual(len(store.all_records()), 1)


if __name__ == "__main__":
    unittest.main()
