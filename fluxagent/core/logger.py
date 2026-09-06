# -*- coding: utf-8 -*-
"""审计日志：单例 JSONLEventLogger，内存队列 + 后台守护线程异步写盘。

前台只做一次 in-memory put（O(1) 不阻塞），守护线程从队列取着写 JSONL 文件。
atexit 注册 shutdown 保证退出时剩日志刷盘。审计事件覆盖 agent 全生命周期，包含图节点转移。
"""

import os
import json
import sys
import threading
import queue
import atexit
from datetime import datetime, timezone

# 内存队列 + 守护线程
class JSONLEventLogger:
    """Process-wide asynchronous writer for per-session JSONL audit logs."""

    # 单例模式
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: str = "logs"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_logger(log_dir)
            return cls._instance
        
    def _init_logger(self, log_dir: str):
        """Initialize the queue worker once for the process-wide singleton."""
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # 无界内存队列，用于缓冲日志事件
        self.log_queue = queue.Queue()
        self._shutdown_lock = threading.Lock()
        self._closed = False

        self.worker_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.worker_thread.start()

        # 确保程序被关闭时，队列里的剩下日志能写完
        atexit.register(self.shutdown)

    # 后台线程的死循环：一直盯着队列，有日志就写，没日志就阻塞休眠
    def _write_loop(self):
        """Drain queued events to per-session JSONL files."""
        while True:
            log_item = self.log_queue.get()

            if log_item is None:
                self.log_queue.task_done()
                break

            try:
                thread_id = log_item.get("thread_id", "system")
                safe_id = "".join(c for c in thread_id if c.isalnum() or c in "-_") or "default"
                file_path = os.path.join(self.log_dir, f"{safe_id}.jsonl")

                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
            except Exception as e:
                # stderr：stdout 在 MCP stdio 模式下是协议通道
                print(f"[Logger Error] 异步写日志失败: {e}", file=sys.stderr)
            finally:
                self.log_queue.task_done()

    # 前台调用的埋点方法
    def log_event(self, thread_id: str, event: str, **kwargs):
        """Queue one structured audit event unless shutdown has started."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        log_item = {
            "ts": now_utc,
            "thread_id": thread_id,
            "event": event,
            **kwargs
        }

        with self._shutdown_lock:
            if self._closed:
                return
            self.log_queue.put(log_item)

    def shutdown(self):
        """Flush and stop the worker exactly once."""
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True
            self.log_queue.put(None)
        self.log_queue.join()
        self.worker_thread.join(timeout=2)

audit_logger = JSONLEventLogger()
