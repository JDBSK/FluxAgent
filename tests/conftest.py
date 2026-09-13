# -*- coding: utf-8 -*-
"""测试会话级环境隔离。

FLUXAGENT_ROUTER_MODEL 会让 create_agent_app 装配 LLM 重排器；测试里
get_provider 通常被替换为 stub，重排器的额外 invoke 会打乱 stub 的响应
序列。除非某个测试显式设置该变量，一律清空，保证装配走确定性启发式重排。
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_router_llm_env(monkeypatch):
    monkeypatch.delenv("FLUXAGENT_ROUTER_MODEL", raising=False)
    yield
