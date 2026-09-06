# -*- coding: utf-8 -*-
"""心跳起搏器：后台协程每 10s 扫任务文件，到点就把任务塞进消息总线。

支持循环任务（hourly/daily/weekly/monthly）自动续期，用 threading.Lock
防多协程并发写坏任务文件。任务存 JSON 而非数据库，因为量小、读写简单、好调试。
"""
import os
import json
import asyncio
import calendar
from datetime import datetime, timedelta
from .config import TASKS_FILE
from .tools.builtins import tasks_lock

async def pacemaker_loop(task_queue: asyncio.Queue, check_interval: int = 10):
    """详细逻辑：后台协程每 check_interval 秒检查一次任务文件，到点就把任务塞进消息总线。"""    
    while True:
        # 每隔 check_interval 秒检查一次任务文件
        await asyncio.sleep(check_interval)
        
        if not os.path.exists(TASKS_FILE):
            continue
            
        now = datetime.now()
        pending_tasks = []  # 存放还未到触发时间的任务
        triggered_tasks = []  # 存放已到触发时间的任务

        # 线程锁，防止多线程/多协程同时读写任务文件导致的竞争条件和数据损坏
        with tasks_lock:
            # 读取任务文件内容
            try:
                with open(TASKS_FILE, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    tasks = json.loads(content)
            except Exception:
                continue
                
            if not tasks:
                continue

            # 遍历任务列表，检查是否有任务到期
            for t in tasks:
                try:
                    #这个target_dt 是任务的目标触发时间
                    target_dt = datetime.strptime(t["target_time"], "%Y-%m-%d %H:%M:%S")
                    if now >= target_dt:

                        triggered_tasks.append(t) #记录为“需要触发”

                        #如果是循环任务就把次数减1，次数耗尽就不再触发
                        repeat_freq = t.get("repeat")
                        if repeat_freq:
                            repeat_count = t.get("repeat_count")
                            
                            if repeat_count is not None:
                                if repeat_count <= 1:
                                    continue
                                else:
                                    t["repeat_count"] = repeat_count - 1

                            #计算下次触发时间
                            if repeat_freq == "hourly":
                                next_dt = target_dt + timedelta(hours=1)
                            elif repeat_freq == "daily":
                                next_dt = target_dt + timedelta(days=1)
                            elif repeat_freq == "weekly":
                                next_dt = target_dt + timedelta(days=7)
                            elif repeat_freq == "monthly":
                                month = target_dt.month + 1
                                year = target_dt.year
                                if month > 12:
                                    month = 1
                                    year += 1
                                last_day = calendar.monthrange(year, month)[1]
                                day = min(target_dt.day, last_day)
                                next_dt = target_dt.replace(year=year, month=month, day=day)
                            else:
                                continue
                                
                            t["target_time"] = next_dt.strftime("%Y-%m-%d %H:%M:%S")
                            pending_tasks.append(t)
                    else:
                        pending_tasks.append(t) #还未到触发时间的任务继续保留在待办列表里
                except Exception:
                    pass

            # 将还没到触发时间的任务和续期后的循环任务写回文件，覆盖原有内容
            if triggered_tasks:
                try:
                    with open(TASKS_FILE, "w", encoding="utf-8") as f:
                        json.dump(pending_tasks, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

        for t in triggered_tasks:
            system_msg = (
                f"【系统内部心跳触发】\n"
                f"你设定的定时任务已到期，请立即主动提醒用户或执行动作。\n"
                f"任务内容：{t['description']}"
            )
            await task_queue.put(system_msg)