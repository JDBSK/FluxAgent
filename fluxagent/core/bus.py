# -*- coding: utf-8 -*-
"""消息总线：用一个 asyncio.Queue 统一收用户输入和心跳触发任务。

设计要点：用户打字和定时任务都往 task_queue 塞，agent_worker 统一消费，
agent 逻辑不用区分消息来源——心跳只是个生产者。
"""

import asyncio

# 消息总线，用于统一收发用户输入和心跳触发的任务
task_queue = asyncio.Queue()  # 异步队列


async def emit_task(content: str):
    """向消息总线 task_queue 投递一条消息（用户输入或心跳触发任务）。"""
    await task_queue.put(content)  # 向消息总线投递任务 await是异步的，确保不会阻塞其他协程
