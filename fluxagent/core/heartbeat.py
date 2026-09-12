# -*- coding: utf-8 -*-
"""心跳起搏器：后台协程每 10s 扫任务文件，到点就把任务塞进消息总线。

支持循环任务（hourly/daily/weekly/monthly）自动续期，用 threading.Lock
防多协程并发写坏任务文件。任务存 JSON 而非数据库，因为量小、读写简单、好调试。

可靠性契约（与 tests/test_task_storage_and_shutdown.py 对齐）：
  - 任务写回用原子替换；写回失败时本次触发的任务不投递（宁可迟到不丢失）；
  - 单条记录非法（如时间格式坏）时保留原记录并记审计，不影响同批其他任务；
  - 心跳写日志失败不进入队列。
"""
import os
import json
import asyncio
import calendar
from datetime import datetime, timedelta

from .config import TASKS_FILE
from .logger import audit_logger
from .task_storage import write_tasks_atomic
from .tools.builtins import tasks_lock

_TASK_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _advance_repeat(target_dt: datetime, repeat_freq: str) -> datetime | None:
    """计算循环任务的下次触发时间；未知频率返回 None。"""
    if repeat_freq == "hourly":
        return target_dt + timedelta(hours=1)
    if repeat_freq == "daily":
        return target_dt + timedelta(days=1)
    if repeat_freq == "weekly":
        return target_dt + timedelta(days=7)
    if repeat_freq == "monthly":
        month = target_dt.month + 1
        year = target_dt.year
        if month > 12:
            month = 1
            year += 1
        last_day = calendar.monthrange(year, month)[1]
        day = min(target_dt.day, last_day)
        return target_dt.replace(year=year, month=month, day=day)
    return None


async def pacemaker_loop(task_queue: asyncio.Queue, check_interval: int = 10):
    """后台协程每 check_interval 秒检查一次任务文件，到点投递任务消息。"""
    while True:
        await asyncio.sleep(check_interval)

        if not os.path.exists(TASKS_FILE):
            continue

        now = datetime.now()
        pending_tasks = []   # 还未到触发时间的任务 + 续期后的循环任务
        triggered_tasks = []  # 已到触发时间的任务

        with tasks_lock:
            try:
                with open(TASKS_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    tasks = json.loads(content)
            except Exception:
                # 文件读取/解析失败：不动文件，等下一轮（可能正在被原子替换）
                continue

            if not tasks:
                continue

            for t in tasks:
                try:
                    target_dt = datetime.strptime(t["target_time"], _TASK_TIME_FORMAT)
                    if now >= target_dt:
                        triggered_tasks.append(t)

                        repeat_freq = t.get("repeat")
                        if repeat_freq:
                            repeat_count = t.get("repeat_count")
                            if repeat_count is not None:
                                if repeat_count <= 1:
                                    continue  # 次数耗尽：触发但不续期
                                t["repeat_count"] = repeat_count - 1
                            next_dt = _advance_repeat(target_dt, repeat_freq)
                            if next_dt is None:
                                continue  # 未知频率：触发本次，不续期
                            t["target_time"] = next_dt.strftime(_TASK_TIME_FORMAT)
                            pending_tasks.append(t)
                    else:
                        pending_tasks.append(t)  # 未到时间，原样保留
                except Exception as exc:
                    # 坏记录保留并可诊断，不静默吞掉，也不影响同批正常任务
                    pending_tasks.append(t)
                    audit_logger.log_event(
                        thread_id="system",
                        event="system_action",
                        action="heartbeat_invalid_task",
                        task_id=str(t.get("id", "unknown")) if isinstance(t, dict) else "unknown",
                        error=str(exc)[:200],
                    )

            # 先原子写回，写失败则本次不投递：避免任务丢失
            if triggered_tasks:
                try:
                    write_tasks_atomic(TASKS_FILE, pending_tasks)
                except Exception as exc:
                    audit_logger.log_event(
                        thread_id="system",
                        event="system_action",
                        action="heartbeat_write_failed",
                        error=str(exc)[:200],
                    )
                    continue

        for t in triggered_tasks:
            system_msg = (
                f"【系统内部心跳触发】\n"
                f"你设定的定时任务已到期，请立即主动提醒用户或执行动作。\n"
                f"任务内容：{t['description']}"
            )
            await task_queue.put(system_msg)
