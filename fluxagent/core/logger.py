# -*- coding: utf-8 -*-
"""审计日志：有界异步 JSONL 写入器，幂等关闭。

前台只做一次 in-memory put（O(1) 不阻塞）；守护线程从队列取着写 JSONL。
队列有界（默认 4096），打满时显式丢弃并计数（dropped_events），不无限膨胀；
写盘失败计入 write_errors。shutdown 幂等，关闭后 log_event 返回 False。

审计事件覆盖 agent 全生命周期，含图节点转移、子任务父子关联字段。
"""

import atexit
import json
import os
import queue
import sys
import threading
from datetime import datetime, timezone


class JSONLEventLogger:
    """Bounded asynchronous JSONL writer for per-session audit logs."""

    def __init__(self, log_dir: str = "logs", max_queue_size: int = 4096):
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_queue: "queue.Queue" = queue.Queue(maxsize=max_queue_size)
        self._state_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._closed = False
        self.dropped_events = 0
        self.write_errors = 0
        self.worker_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.worker_thread.start()
        atexit.register(self.shutdown)

    def _write_loop(self):
        """Drain queued events to per-session JSONL files until the sentinel."""
        while True:
            item = self.log_queue.get()
            try:
                if item is None:
                    return
                thread_id = str(item.get("thread_id", "system"))
                safe_id = "".join(c for c in thread_id if c.isalnum() or c in "-_") or "default"
                file_path = os.path.join(self.log_dir, f"{safe_id}.jsonl")
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            except Exception as e:
                # stderr：stdout 在 MCP stdio 模式下是协议通道
                self.write_errors += 1
                print(f"[Logger Error] 异步写日志失败: {e}", file=sys.stderr)
            finally:
                self.log_queue.task_done()

    def log_event(self, thread_id: str, event: str, **kwargs) -> bool:
        """Queue one structured audit event; False means dropped or closed.

        队列打满时宁可丢弃并计数，也不阻塞被审计的执行路径。
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log_item = {
            "ts": now_utc,
            "thread_id": thread_id,
            "event": event,
            **kwargs,
        }
        with self._state_lock:
            if self._closed:
                return False
            try:
                self.log_queue.put_nowait(log_item)
                return True
            except queue.Full:
                self.dropped_events += 1
                return False

    def shutdown(self):
        """Flush and stop the worker exactly once; safe to call repeatedly."""
        with self._shutdown_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True
            # 队列可能已满：先挤掉一个待写事件也要保证哨兵能入队，
            # 否则关闭路径会永久阻塞。
            try:
                self.log_queue.put_nowait(None)
            except queue.Full:
                try:
                    self.log_queue.get_nowait()
                    self.log_queue.task_done()
                except (queue.Empty, ValueError):
                    pass
                self.dropped_events += 1
                self.log_queue.put_nowait(None)
            self.log_queue.join()
            self.worker_thread.join()


# 进程级单例：各模块 `from .logger import audit_logger` 共享同一写入器
audit_logger = JSONLEventLogger()
