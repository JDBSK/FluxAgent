# -*- coding: utf-8 -*-
"""多账号 key 池与轮询调用器。

每个 key 来自不同账号、各自独立限流：acquire 轮询健康 key，请求失败
（429/5xx/超时）自动冷却并换下一个。RotatingChat 直接用 OpenAI 兼容
客户端，在调用层完成换 key 重试——对所有调用方（judge/AB/校准）透明。
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

from openai import OpenAI


class KeyPool:
    """线程安全的多账号 key 池：轮询 + 失败冷却。"""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as fh:
            self._keys = [line.strip() for line in fh
                          if line.strip() and not line.strip().startswith("#")]
        if not self._keys:
            raise ValueError(f"key 文件为空：{path}")
        self._lock = threading.Lock()
        self._idx = 0
        self._cooldown: dict[str, float] = {}

    def __len__(self):
        return len(self._keys)

    def acquire(self) -> str:
        """取下一个健康 key；全部冷却时取最早解冻的（宁可等待不空转）。"""
        now = time.monotonic()
        with self._lock:
            ordered = self._keys[self._idx:] + self._keys[:self._idx]
            self._idx = (self._idx + 1) % len(self._keys)
        healthy = [k for k in ordered if self._cooldown.get(k, 0) <= now]
        if healthy:
            return healthy[0]
        # 全部冷却：返回最早解冻的 key，调用方按其剩余冷却时间退避
        soonest = min(self._keys, key=lambda k: self._cooldown.get(k, 0))
        wait = self._cooldown.get(soonest, 0) - now
        if wait > 0:
            time.sleep(min(wait, 30))
        return soonest

    def report_fail(self, key: str, seconds: float = 45) -> None:
        with self._lock:
            self._cooldown[key] = time.monotonic() + seconds

    def report_ok(self, key: str) -> None:
        with self._lock:
            self._cooldown.pop(key, None)


class RotatingChat:
    """多 key 轮询的 OpenAI 兼容调用器，接口兼容 langchain（.invoke → .content）。"""

    def __init__(self, pool: KeyPool, base_url: str, model: str,
                 timeout: float = 300, max_retries: int = 0, temperature: float = 0.0):
        self.pool = pool
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    def _messages(self, messages: list) -> list[dict]:
        out = []
        for m in messages:
            if isinstance(m, dict):
                out.append(m)
            else:  # langchain 消息类
                role = "assistant" if getattr(m, "type", "") == "ai" else \
                       "user" if getattr(m, "type", "") == "human" else "system"
                out.append({"role": role, "content": str(getattr(m, "content", m))})
        return out

    def invoke(self, messages: list, max_attempts: int = 8) -> Any:
        """带换 key 重试的调用；全 key 失败抛最后一层异常。

        尝试次数有上限（默认 8）：换 key 只解决限流/单账号故障，
        解决不了超时——超时场景轮完全部 key 只会成倍拖慢。
        """
        payload = self._messages(messages)
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            key = self.pool.acquire()
            client = OpenAI(api_key=key, base_url=self.base_url,
                            timeout=self.timeout, max_retries=self.max_retries)
            try:
                resp = client.chat.completions.create(
                    model=self.model, messages=payload,
                    temperature=self.temperature)
                self.pool.report_ok(key)
                content = resp.choices[0].message.content or ""
                return SimpleNamespace(content=content)
            except Exception as exc:
                last_exc = exc
                self.pool.report_fail(key, seconds=45 * (1 + attempt // len(self.pool)))
                # 限流/网关故障换 key 立即重试；解析类故障同样换 key 无妨
                time.sleep(min(1.5 * attempt, 10))
        raise RuntimeError(f"all {len(self.pool)} keys exhausted: {last_exc}")
