# -*- coding: utf-8 -*-
"""消息总线：用一个 asyncio.Queue 统一收用户输入和心跳触发任务。

设计要点：用户打字和定时任务都往 task_queue 塞，agent_worker 统一消费，
agent 逻辑不用区分消息来源——心跳只是个生产者。

退出协议：先停所有生产者（心跳取消后可能还有最后一个任务入队），再向
消费者发送 "/exit" 哨兵并 join 队列，保证已入队的工作全部被处理完后
消费者才退出（exit-time drain）。
"""

import asyncio
from contextlib import suppress

# 消息总线，用于统一收发用户输入和心跳触发的任务
task_queue = asyncio.Queue()  # 异步队列


async def emit_task(content: str):
    """向消息总线 task_queue 投递一条消息（用户输入或心跳触发任务）。"""
    await task_queue.put(content)  # 向消息总线投递任务 await是异步的，确保不会阻塞其他协程


async def stop_workers(worker: asyncio.Task, heartbeat_worker: asyncio.Task,
                       queue: asyncio.Queue) -> None:
    """按序关停：先取消心跳生产者，等它完全退出后再发送退出哨兵并排空队列。"""
    heartbeat_worker.cancel()
    with suppress(asyncio.CancelledError):
        await heartbeat_worker
    await queue.put("/exit")
    await queue.join()
    await worker
