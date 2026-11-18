# -*- coding: utf-8 -*-
"""多模型适配工厂。

通过 provider_name 统一入口创建不同厂商的 ChatModel 实例：
  openai/aliyun/dashscope/z.ai/tencent/other -> ChatOpenAI（OpenAI 兼容接口）
  anthropic -> ChatAnthropic
  ollama -> ChatOllama

未配置 base_url 时用 COMPATIBLE_BASE_URLS 字典兜底国产模型兼容地址，
工厂 + 配置驱动，而非散落的 if-else。
"""

import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv
'''
多模型适配(Factory)
'''
load_dotenv()

# 单次 LLM 请求的硬超时与重试上限。不设置时 openai SDK 默认可等 600s：
# 一次挂起的调用会拖着整条串行流水线一起等（预算熔断只在节点边界检查，
# 拦不住在途请求），必须在这里给出上限。可用环境变量覆盖。
LLM_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("FLUXAGENT_LLM_TIMEOUT_SECONDS", "120")
)
LLM_MAX_RETRIES = int(os.getenv("FLUXAGENT_LLM_MAX_RETRIES", "2"))

# 各大厂商官方的 OpenAI 兼容接口地址 (当用户未配置 BASE_URL 时作为兜底)
COMPATIBLE_BASE_URLS = {
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "z.ai": "https://open.bigmodel.cn/api/paas/v4",
    "tencent": "https://api.hunyuan.cloud.tencent.com/v1"
}

def get_provider(
    provider_name: str = "openai", 
    model_name: str = "gpt-4o-mini", 
    temperature: float = 0.0,
    base_url: str | None = None,  # 允许外部传入
    api_key: str | None = None,   # 允许外部传入
    **kwargs: Any
) -> BaseChatModel:
    """
    模型适配器工厂
    """
    provider_name = provider_name.lower()
    
    if provider_name in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
        from langchain_openai import ChatOpenAI
        
        current_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not current_api_key:
            raise ValueError(f"未找到 API Key！请确保 .env 中配置了 OPENAI_API_KEY")
            

        final_base_url = base_url or os.environ.get("OPENAI_API_BASE")
        if not final_base_url:
            final_base_url = COMPATIBLE_BASE_URLS.get(provider_name) 

        if "timeout" not in kwargs:
            kwargs.setdefault("request_timeout", LLM_REQUEST_TIMEOUT_SECONDS)
        kwargs.setdefault("max_retries", LLM_MAX_RETRIES)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        
        current_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not current_api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量！")
            
        final_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        if "default_request_timeout" not in kwargs:
            kwargs.setdefault("timeout", LLM_REQUEST_TIMEOUT_SECONDS)
        kwargs.setdefault("max_retries", LLM_MAX_RETRIES)
        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )
        
    elif provider_name == "ollama":
        from langchain_ollama import ChatOllama
        
        final_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name, 
            temperature=temperature, 
            base_url=final_base_url,
            **kwargs
        )
        
    else:
        raise ValueError(f"不支持的模型提供商: {provider_name}")

# 测试模型调用    
# LLM = get_provider(provider_name='aliyun', model_name='glm-5')
# res = LLM.invoke('你是谁')
# print(type(res))
# print(res)


