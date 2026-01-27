# LLM Adapters Module
"""
通用 LLM 适配器框架

支持多种 LLM 后端的统一接口，包括：
- DeepSeek (API/本地部署)
- Ollama (本地开源模型)
- vLLM (高性能本地推理)
- OpenAI 兼容接口

使用方式：
    from llm.adapters import LLMAdapterFactory, BaseLLMAdapter
    
    # 创建适配器
    adapter = LLMAdapterFactory.create("deepseek", config={...})
    
    # 使用适配器
    response = adapter.chat([{"role": "user", "content": "Hello"}])
"""

from .base import BaseLLMAdapter, LLMResponse, LLMConfig
from .deepseek import DeepSeekAdapter
from .ollama import OllamaAdapter
from .vllm import VLLMAdapter
from .openai_compat import OpenAICompatAdapter
from .factory import LLMAdapterFactory, LLMAdapterRegistry

__all__ = [
    # 基础类
    "BaseLLMAdapter",
    "LLMResponse",
    "LLMConfig",
    # 具体适配器
    "DeepSeekAdapter",
    "OllamaAdapter",
    "VLLMAdapter",
    "OpenAICompatAdapter",
    # 工厂和注册
    "LLMAdapterFactory",
    "LLMAdapterRegistry",
]

