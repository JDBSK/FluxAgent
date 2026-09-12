# -*- coding: utf-8 -*-
"""任务 JSON 的原子写入。

调用方负责用 tasks_lock 串行化访问；本模块保证"要么完整替换、要么保留
旧文件"——临时文件 + fsync + os.replace 三步，任何一步失败都不会留下
半截 JSON 或空文件，调用方据此决定是否继续投递任务。
"""
import json
import os
import tempfile
from pathlib import Path


def write_tasks_atomic(path, tasks) -> None:
    """把任务列表原子写入 path；失败时抛异常且旧文件保持不变。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=target.name + ".", suffix=".tmp", delete=False,
        ) as stream:
            temporary = stream.name
            # 先序列化再落盘：序列化失败时连临时文件都不产生
            payload = json.dumps(tasks, ensure_ascii=False, indent=2)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def read_tasks(path) -> list:
    """读取任务列表；文件不存在或为空返回 []，JSON 损坏抛 ValueError。"""
    target = Path(path)
    if not target.exists():
        return []
    content = target.read_text(encoding="utf-8").strip()
    if not content:
        return []
    tasks = json.loads(content)
    if not isinstance(tasks, list):
        raise ValueError("任务文件顶层必须是列表")
    return tasks
