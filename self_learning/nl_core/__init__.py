"""
Nested Learning 核心模块

基于 Google Research 论文 "Nested Learning: The Illusion of Deep Learning Architectures"
(Behrouz et al., arXiv:2512.24695, 2025)

本模块实现 NL 框架与 NLGSM 治理系统的对接适配层。

LLM 支持：
- 支持多种 LLM 后端（通过适配器注入）
- DeepSeek, Ollama, vLLM, OpenAI 兼容接口
- 学习的起点是 LLM 的现有知识库

使用示例：
    from self_learning.nl_core import KernelFactory
    
    # 使用 Ollama 创建内核
    kernel = KernelFactory.create_with_adapter(
        "ollama",
        model="llama3.2",
        config={"memory_config": {...}}
    )
"""
from .types import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    MemoryLevel,
    ContinuumMemoryState,
    LevelDelta,
)
from .kernel import NestedLearningKernel, LLMBasedNLKernel, KernelFactory
from .memory import ContinuumMemorySystem
from .optimizer import ExpressiveOptimizer

__all__ = [
    # 类型
    "NLLevel",
    "LearningScope",
    "ContextFlowSegment",
    "MemoryLevel",
    "ContinuumMemoryState",
    "LevelDelta",
    # 内核
    "NestedLearningKernel",
    "LLMBasedNLKernel",
    "KernelFactory",
    # 记忆系统
    "ContinuumMemorySystem",
    # 优化器
    "ExpressiveOptimizer",
]

