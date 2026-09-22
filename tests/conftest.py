# -*- coding: utf-8 -*-
"""测试会话级环境隔离。

FLUXAGENT_ROUTER_MODEL 会让 create_agent_app 装配 LLM 重排器；测试里
get_provider 通常被替换为 stub，重排器的额外 invoke 会打乱 stub 的响应
序列。除非某个测试显式设置该变量，一律清空，保证装配走确定性启发式重排。

同理，规划入口分类器默认取 .env 的 DEFAULT_MODEL，一旦本机 .env 配了模型，
装配期就会多建一个 provider；stub 化的 get_provider 会把同一个 stub 同时
交给分类器和主 LLM，分类器于是吃掉脚本里的第一条回复，测试结果取决于本机
.env。默认关闭，只有专门测混合路由的用例显式打开。
"""
import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_router_llm_env(monkeypatch):
    monkeypatch.delenv("FLUXAGENT_ROUTER_MODEL", raising=False)
    monkeypatch.setenv("FLUXAGENT_PLAN_CLASSIFIER", "0")
    yield
