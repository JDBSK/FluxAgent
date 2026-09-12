# -*- coding: utf-8 -*-
"""任务存储原子写、心跳故障注入与退出排空回归（移植自 CyberClaw 同名测试）。
"""
import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fluxagent.core.task_storage import write_tasks_atomic
from fluxagent.core.bus import stop_workers
from fluxagent.core.tools import builtins
import fluxagent.core.heartbeat as heartbeat


class AtomicTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "tasks.json"
        self.path.write_text('[{"id":"original"}]', encoding="utf-8")

    def test_round_trip(self):
        write_tasks_atomic(self.path, [{"id": "new", "description": "中文"}])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8"))[0]["description"], "中文")
        # 临时文件不留残余
        self.assertEqual(len(list(self.path.parent.iterdir())), 1)

    def test_replace_failure_keeps_previous_file(self):
        with patch("fluxagent.core.task_storage.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                write_tasks_atomic(self.path, [{"id": "new"}])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), [{"id": "original"}])
        self.assertEqual(len(list(self.path.parent.iterdir())), 1)

    def test_serialization_failure_keeps_previous_file(self):
        with self.assertRaises(TypeError):
            write_tasks_atomic(self.path, [{"bad": object()}])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), [{"id": "original"}])
        self.assertEqual(len(list(self.path.parent.iterdir())), 1)

    def test_invalid_repeat_parameters_do_not_write(self):
        future = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
        with patch.object(builtins, "TASKS_FILE", str(self.path)):
            for repeat, count in (("invalid", None), ("daily", 0), ("daily", -1), (None, 2)):
                result = builtins.schedule_task.invoke({"target_time": future, "description": "test",
                                                        "repeat": repeat, "repeat_count": count})
                self.assertIn("设定失败", result)
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), [{"id": "original"}])

    def test_read_profile_is_registered(self):
        self.assertIn("read_user_profile", [tool.name for tool in builtins.BUILTIN_TOOLS])
        profile = self.path.parent / "profile.md"
        profile.write_text("profile", encoding="utf-8")
        with patch.object(builtins, "PROFILE_PATH", str(profile)):
            self.assertEqual(builtins.read_user_profile.invoke({}), "profile")


class HeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def run_one_cycle(self, queue):
        cycles = 0
        async def sleep(_):
            nonlocal cycles
            cycles += 1
            if cycles > 1:
                raise asyncio.CancelledError
        with patch.object(heartbeat.asyncio, "sleep", side_effect=sleep):
            with self.assertRaises(asyncio.CancelledError):
                await heartbeat.pacemaker_loop(queue, check_interval=0)

    async def test_failed_write_does_not_enqueue(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tasks.json"
            record = {"id": "due", "target_time": "2000-01-01 00:00:00", "description": "test"}
            path.write_text(json.dumps([record]), encoding="utf-8")
            queue = asyncio.Queue()
            with patch.object(heartbeat, "TASKS_FILE", str(path)), \
                 patch.object(heartbeat, "write_tasks_atomic", side_effect=OSError("disk error")):
                await self.run_one_cycle(queue)
            self.assertTrue(queue.empty())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), [record])

    async def test_invalid_record_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tasks.json"
            bad = {"id": "bad", "target_time": "invalid", "description": "bad"}
            path.write_text(json.dumps([bad, {"id": "due", "target_time": "2000-01-01 00:00:00", "description": "due"}]),
                            encoding="utf-8")
            queue = asyncio.Queue()
            with patch.object(heartbeat, "TASKS_FILE", str(path)):
                await self.run_one_cycle(queue)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), [bad])
            self.assertEqual(queue.qsize(), 1)

    async def test_producer_is_stopped_before_exit_sentinel(self):
        queue, received = asyncio.Queue(), []
        await queue.put("user task")
        async def producer():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await queue.put("last heartbeat")
                raise
        async def consumer():
            while True:
                item = await queue.get()
                try:
                    if item == "/exit":
                        return
                    received.append(item)
                finally:
                    queue.task_done()
        heartbeat_worker = asyncio.create_task(producer())
        worker = asyncio.create_task(consumer())
        await asyncio.sleep(0)
        await asyncio.wait_for(stop_workers(worker, heartbeat_worker, queue), timeout=2)
        self.assertEqual(received, ["user task", "last heartbeat"])
        self.assertTrue(queue.empty())
        self.assertTrue(worker.done())


if __name__ == "__main__":
    unittest.main()
