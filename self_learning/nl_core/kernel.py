"""
Nested Learning 内核

提供 NL 框架与 NLGSM 治理系统的标准接口。
支持多种内核实现（LLM 驱动、真实 NL 等）。

LLM 支持：
- 支持多种 LLM 后端（通过适配器注入）
- DeepSeek, Ollama, vLLM, OpenAI 兼容接口
- 学习的起点是 LLM 的现有知识库
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import json

from .types import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    ContinuumMemoryState,
    LevelDelta,
)
from .memory import ContinuumMemorySystem
from .optimizer import ExpressiveOptimizer, DeepMomentumOptimizer

# LLM 适配器框架
from llm.adapters import BaseLLMAdapter, LLMAdapterFactory
from llm.adapters.base import MockLLMAdapter
from llm.prompts import PromptTemplates


class NestedLearningKernel(ABC):
    """
    Nested Learning 内核抽象接口
    
    任何实现 NL 范式的具体系统都应实现此接口，
    以便与 NLGSM 治理框架对接。
    
    核心治理约束：
    1. freeze/unfreeze 必须与 NLGSM 状态机同步
    2. 所有学习操作必须通过 Scope 控制
    3. 所有变更必须产生可审计的 ContextFlowSegment
    """
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        初始化 NL 内核
        
        Args:
            config: 配置参数，包括：
                - model_config: 基础模型配置
                - memory_config: 记忆系统配置
                - optimizer_config: 优化器配置
        """
        pass
    
    @abstractmethod
    def get_current_state(self) -> Dict[NLLevel, Dict[str, Any]]:
        """
        获取当前各层级状态快照
        
        Returns:
            各层级的状态字典
        """
        pass
    
    @abstractmethod
    def execute_learning_step(
        self,
        context: Dict[str, Any],
        scope: LearningScope
    ) -> ContextFlowSegment:
        """
        执行一步学习
        
        这是 NL 内核的核心方法。在 NLGSM 的 Learning State 中被调用。
        
        Args:
            context: 学习上下文（数据、目标、约束等）
            scope: 学习范围控制
            
        Returns:
            本次学习产生的 ContextFlowSegment
        """
        pass
    
    @abstractmethod
    def apply_state_delta(
        self,
        delta: Dict[NLLevel, Dict[str, Any]],
        scope: LearningScope
    ) -> bool:
        """
        应用状态变更
        
        在 NLGSM 的 Release State 中，将经批准的变更应用到内核。
        
        Args:
            delta: 要应用的状态变更
            scope: 允许的变更范围
            
        Returns:
            是否成功应用
        """
        pass
    
    @abstractmethod
    def rollback_to_state(self, state_snapshot: ContinuumMemoryState) -> bool:
        """
        回滚到指定状态
        
        在 NLGSM 的 Rollback State 中被调用。
        
        Args:
            state_snapshot: 目标状态快照
            
        Returns:
            是否成功回滚
        """
        pass
    
    @abstractmethod
    def freeze(self) -> None:
        """
        冻结内核，禁止任何学习操作
        
        在进入 NLGSM 的 Frozen State 时调用。
        """
        pass
    
    @abstractmethod
    def unfreeze(self) -> None:
        """
        解冻内核，允许学习操作
        
        在进入 NLGSM 的 Learning State 时调用。
        """
        pass
    
    @abstractmethod
    def get_memory_state(self) -> ContinuumMemoryState:
        """
        获取连续记忆系统状态
        
        Returns:
            CMS 状态对象
        """
        pass
    
    @abstractmethod
    def create_snapshot(self) -> ContinuumMemoryState:
        """
        创建当前状态快照
        
        用于 NLGSM 的 Frozen 状态存档
        """
        pass


class LLMBasedNLKernel(NestedLearningKernel):
    """
    基于 LLM 的 NL 内核实现
    
    这是一个原型级实现，将 LLM 探索适配为 NL 框架。
    适用于 LLM 驱动的知识发现和学习。
    
    与真实 NL 内核的区别：
    - 无真实梯度计算
    - 通过 LLM 推理模拟学习过程
    - 记忆系统存储知识片段而非参数
    
    LLM 注入方式：
    1. 直接传入适配器实例 (llm_adapter)
    2. 通过工厂创建 (使用 with_adapter 类方法)
    3. 使用旧的 llm_client（向后兼容）
    
    使用示例：
        # 方式1：使用适配器
        from llm.adapters import OllamaAdapter
        adapter = OllamaAdapter(config)
        kernel = LLMBasedNLKernel(llm_adapter=adapter)
        
        # 方式2：使用工厂
        kernel = LLMBasedNLKernel.with_adapter("deepseek", model="deepseek-chat")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化 LLM-NL 内核
        
        Args:
            llm_client: 旧的 LLM 客户端（向后兼容）
            llm_adapter: 新的 LLM 适配器（推荐）
            config: 内核配置
        """
        self.config = config or {}
        
        # 优先使用新的适配器
        if llm_adapter is not None:
            self._adapter = llm_adapter
            self._use_adapter = True
            self.llm = None
        elif llm_client is not None:
            self.llm = llm_client
            self._adapter = None
            self._use_adapter = False
        else:
            # 默认使用模拟适配器
            self._adapter = MockLLMAdapter()
            self._use_adapter = True
            self.llm = None
        
        # 内核状态
        self._frozen = True
        self._initialized = False
        
        # 连续记忆系统
        self.cms: Optional[ContinuumMemorySystem] = None
        
        # 表达性优化器（模拟）
        self.optimizer: Optional[ExpressiveOptimizer] = None
        
        # 学习历史
        self.learning_history: List[ContextFlowSegment] = []
        
        # 当前学习会话
        self.current_session_id: Optional[str] = None
    
    @classmethod
    def with_adapter(
        cls,
        adapter_type: str,
        config: Optional[Dict[str, Any]] = None,
        **adapter_kwargs
    ) -> "LLMBasedNLKernel":
        """
        使用指定类型的适配器创建 NL 内核
        
        Args:
            adapter_type: 适配器类型 ("deepseek", "ollama", "vllm", etc.)
            config: 内核配置
            **adapter_kwargs: 传递给适配器的参数
            
        Returns:
            LLMBasedNLKernel 实例
        """
        adapter = LLMAdapterFactory.create_and_initialize(
            adapter_type, **adapter_kwargs
        )
        return cls(llm_adapter=adapter, config=config)
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """初始化 LLM-NL 内核"""
        self.config.update(config)
        
        # 初始化 CMS
        cms_config = config.get("memory_config", {})
        self.cms = ContinuumMemorySystem(config=cms_config)
        
        # 初始化优化器
        opt_config = config.get("optimizer_config", {})
        self.optimizer = DeepMomentumOptimizer(
            learning_rate=opt_config.get("learning_rate", 0.001),
            momentum_depth=opt_config.get("momentum_depth", 3),
        )
        
        self._initialized = True
        self._frozen = True  # 初始化后默认冻结
        
        print(f"[LLM-NL Kernel] Initialized with config: {list(config.keys())}")
    
    def get_current_state(self) -> Dict[NLLevel, Dict[str, Any]]:
        """获取当前各层级状态"""
        if not self.cms:
            return {}
        
        return {
            level: self.cms.get_level_state(level)
            for level in NLLevel
        }
    
    def execute_learning_step(
        self,
        context: Dict[str, Any],
        scope: LearningScope
    ) -> ContextFlowSegment:
        """
        执行一步 LLM 驱动的学习
        
        将 LLM 探索结果映射到 NL 层级结构
        """
        if self._frozen:
            raise RuntimeError("Kernel is frozen, cannot execute learning step")
        
        if not self._initialized:
            raise RuntimeError("Kernel not initialized")
        
        # 生成 segment ID
        segment_id = f"seg_{uuid.uuid4().hex[:12]}"
        
        # 执行 CMS 步骤
        deltas = self.cms.step(context, scope)
        
        # 构建 level_deltas 字典
        level_deltas = {delta.level: delta for delta in deltas}
        
        # 如果有 LLM，执行 LLM 推理
        llm_signals = []
        if self.llm:
            llm_result = self._execute_llm_reasoning(context)
            llm_signals = llm_result.get("signals", [])
            
            # 将 LLM 结果存入记忆层
            if scope.is_level_allowed(NLLevel.MEMORY):
                memory_delta = self.cms.levels[NLLevel.MEMORY].update(
                    {
                        "llm_findings": llm_result.get("findings", []),
                        "reasoning_chain": llm_result.get("reasoning", ""),
                    },
                    self.cms.global_step
                )
                level_deltas[NLLevel.MEMORY] = memory_delta
        
        # 更新优化器策略
        if scope.is_level_allowed(NLLevel.OPTIMIZER) and self.optimizer:
            meta_context = {
                "loss_history": context.get("loss_history", []),
                "gradient_stats": context.get("gradient_stats", {}),
                "stability_score": context.get("stability_score", 0.5),
            }
            self.optimizer.update_strategy(meta_context)
        
        # 构建 ContextFlowSegment
        segment = ContextFlowSegment(
            segment_id=segment_id,
            scope_id=scope.scope_id,
            input_context=context,
            level_deltas=level_deltas,
            signals=llm_signals,
        )
        
        self.learning_history.append(segment)
        
        return segment
    
    def _call_llm_json(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> Dict[str, Any]:
        """
        调用 LLM 并返回 JSON
        
        统一处理新旧两种 LLM 接口
        """
        if self._use_adapter and self._adapter:
            return self._adapter.chat_json(messages, temperature=temperature)
        elif self.llm:
            return self.llm.chat_json(messages, temperature=temperature)
        else:
            # 返回模拟结果
            return {"findings": [], "reasoning": "", "signals": []}
    
    def _execute_llm_reasoning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行 LLM 推理
        
        使用 LLM 从现有知识库进行推理，发现新知识。
        这是自学习的核心：学习的起点不是 0，而是 LLM 的现有知识。
        """
        # 检查是否有 LLM 可用
        has_llm = (self._use_adapter and self._adapter) or self.llm
        if not has_llm:
            return {"findings": [], "reasoning": "", "signals": []}
        
        # 构建 prompt
        goal = context.get("goal", "")
        current_findings = context.get("findings", [])
        step_info = {
            "step": context.get("step", 0),
            "action": context.get("action", "reasoning"),
            "query": context.get("query", ""),
        }
        
        # 构建系统提示词
        system_prompt = """你是一个知识发现引擎，负责从已有知识中推理出新的见解。

你的任务是：
1. 分析给定的学习目标和当前发现
2. 基于你的知识库进行推理
3. 发现新的知识点、模式或关联
4. 评估发现的置信度

请以 JSON 格式返回结果，包含以下字段：
- findings: 发现列表，每个发现包含 type, content, confidence
- reasoning: 推理链路描述
- next_steps: 建议的下一步探索方向
- signals: 学习信号列表"""
        
        # 构建用户提示词
        user_prompt = f"""学习目标: {goal}

当前步骤信息:
{json.dumps(step_info, ensure_ascii=False, indent=2)}

已有发现 ({len(current_findings)} 个):
{json.dumps(current_findings[-5:] if current_findings else [], ensure_ascii=False, indent=2)}

请基于你的知识库，分析以上信息并发现新知识。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self._call_llm_json(messages, temperature=0.7)
            
            # 确保响应格式正确
            findings = response.get("findings", [])
            reasoning = response.get("reasoning", "")
            signals = response.get("signals", [])
            
            # 添加 LLM 信息到信号
            llm_info = self.get_llm_info()
            signals.append({
                "signal_type": "llm_inference",
                "content": {
                    "goal": goal,
                    "adapter": llm_info.get("adapter_name", "unknown"),
                    "model": llm_info.get("model", "unknown"),
                }
            })
            
            return {
                "findings": findings,
                "reasoning": reasoning,
                "signals": signals,
                "next_steps": response.get("next_steps", []),
            }
        
        except Exception as e:
            print(f"  [NL Kernel] LLM 推理失败: {e}")
            return {
                "findings": [],
                "reasoning": f"LLM 调用失败: {e}",
                "signals": [{"signal_type": "llm_error", "content": {"error": str(e)}}],
            }
    
    def get_llm_info(self) -> Dict[str, Any]:
        """获取当前 LLM 信息"""
        if self._use_adapter and self._adapter:
            return self._adapter.get_statistics()
        elif self.llm and hasattr(self.llm, 'get_statistics'):
            return self.llm.get_statistics()
        return {"status": "no_llm"}
    
    def set_adapter(self, adapter: BaseLLMAdapter) -> None:
        """
        设置新的 LLM 适配器
        
        允许在运行时切换 LLM 后端
        
        Args:
            adapter: LLM 适配器实例
        """
        self._adapter = adapter
        self._use_adapter = True
        self.llm = None
        print(f"[NL Kernel] LLM 适配器已切换为: {adapter.adapter_name}")
    
    def apply_state_delta(
        self,
        delta: Dict[NLLevel, Dict[str, Any]],
        scope: LearningScope
    ) -> bool:
        """应用状态变更"""
        if not self.cms:
            return False
        
        for level, level_delta in delta.items():
            if not scope.is_level_allowed(level):
                continue
            
            if level in self.cms.levels:
                self.cms.levels[level].update(level_delta, self.cms.global_step)
        
        return True
    
    def rollback_to_state(self, state_snapshot: ContinuumMemoryState) -> bool:
        """回滚到指定状态"""
        if not self.cms:
            return False
        
        return self.cms.restore_snapshot(state_snapshot.state_id)
    
    def freeze(self) -> None:
        """冻结内核"""
        self._frozen = True
        if self.cms:
            self.cms.freeze()
        print("[LLM-NL Kernel] Frozen")
    
    def unfreeze(self) -> None:
        """解冻内核"""
        self._frozen = False
        if self.cms:
            self.cms.unfreeze()
        print("[LLM-NL Kernel] Unfrozen")
    
    def get_memory_state(self) -> ContinuumMemoryState:
        """获取记忆系统状态"""
        if not self.cms:
            return ContinuumMemoryState(state_id="empty")
        return self.cms.get_state()
    
    def create_snapshot(self) -> ContinuumMemoryState:
        """创建快照"""
        if not self.cms:
            return ContinuumMemoryState(state_id="empty")
        return self.cms.create_snapshot()
    
    def is_frozen(self) -> bool:
        """检查是否冻结"""
        return self._frozen
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "initialized": self._initialized,
            "frozen": self._frozen,
            "learning_history_count": len(self.learning_history),
            "current_session": self.current_session_id,
        }
        
        if self.cms:
            stats["cms"] = self.cms.get_statistics()
        
        if self.optimizer and hasattr(self.optimizer, "get_statistics"):
            stats["optimizer"] = self.optimizer.get_statistics()
        
        # 添加 LLM 信息
        stats["llm"] = self.get_llm_info()
        
        return stats


@dataclass
class KernelFactory:
    """
    NL 内核工厂
    
    用于创建不同类型的 NL 内核实例。
    支持多种 LLM 后端。
    """
    
    @staticmethod
    def create(
        kernel_type: str,
        llm_client: Optional[Any] = None,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        adapter_type: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        **adapter_kwargs
    ) -> NestedLearningKernel:
        """
        创建 NL 内核
        
        Args:
            kernel_type: 内核类型 ("llm", "pytorch", "custom")
            llm_client: LLM 客户端（向后兼容）
            llm_adapter: LLM 适配器实例
            adapter_type: LLM 适配器类型（自动创建适配器）
            config: 内核配置
            **adapter_kwargs: 传递给适配器的参数
            
        Returns:
            NL 内核实例
            
        使用示例：
            # 使用适配器类型
            kernel = KernelFactory.create(
                "llm",
                adapter_type="ollama",
                model="llama3.2",
                config={"memory_config": {...}}
            )
            
            # 使用适配器实例
            adapter = OllamaAdapter(config)
            kernel = KernelFactory.create("llm", llm_adapter=adapter)
        """
        # 如果指定了 adapter_type，自动创建适配器
        if adapter_type and not llm_adapter:
            llm_adapter = LLMAdapterFactory.create_and_initialize(
                adapter_type, **adapter_kwargs
            )
        
        if kernel_type == "llm":
            kernel = LLMBasedNLKernel(
                llm_client=llm_client,
                llm_adapter=llm_adapter,
                config=config
            )
        else:
            # 默认使用 LLM 内核
            kernel = LLMBasedNLKernel(
                llm_client=llm_client,
                llm_adapter=llm_adapter,
                config=config
            )
        
        # 初始化
        kernel.initialize(config or {})
        
        return kernel
    
    @staticmethod
    def create_with_adapter(
        adapter_type: str,
        config: Optional[Dict[str, Any]] = None,
        **adapter_kwargs
    ) -> NestedLearningKernel:
        """
        便捷方法：使用指定 LLM 适配器创建内核
        
        Args:
            adapter_type: LLM 适配器类型
            config: 内核配置
            **adapter_kwargs: 适配器参数
            
        Returns:
            已初始化的 NL 内核
        """
        return KernelFactory.create(
            kernel_type="llm",
            adapter_type=adapter_type,
            config=config,
            **adapter_kwargs
        )

