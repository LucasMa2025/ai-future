# LLM Integration Module
"""
LLM 集成模块

提供统一的 LLM 访问接口，支持多种后端：
- DeepSeek (API/本地部署)
- Ollama (本地开源模型)
- vLLM (高性能推理)
- OpenAI 兼容接口

使用示例：
    from llm import LLMAdapterFactory, BaseLLMAdapter
    
    # 创建适配器
    adapter = LLMAdapterFactory.create("deepseek", base_url="...")
    
    # 使用适配器
    response = adapter.chat([{"role": "user", "content": "Hello"}])
"""

# 保持向后兼容
from .client import DeepSeekClient, MockDeepSeekClient
from .prompts import PromptTemplates
from .risk_evaluator import LLMRiskEvaluator

# 新的适配器框架
from .adapters import (
    # 基础类
    BaseLLMAdapter,
    LLMResponse,
    LLMConfig,
    # 具体适配器
    DeepSeekAdapter,
    OllamaAdapter,
    VLLMAdapter,
    OpenAICompatAdapter,
    # 工厂和注册
    LLMAdapterFactory,
    LLMAdapterRegistry,
)

__all__ = [
    # 向后兼容
    'DeepSeekClient',
    'MockDeepSeekClient',
    'PromptTemplates',
    'LLMRiskEvaluator',
    # 新的适配器框架
    'BaseLLMAdapter',
    'LLMResponse',
    'LLMConfig',
    'DeepSeekAdapter',
    'OllamaAdapter',
    'VLLMAdapter',
    'OpenAICompatAdapter',
    'LLMAdapterFactory',
    'LLMAdapterRegistry',
]

