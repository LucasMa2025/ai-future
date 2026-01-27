"""
知识生成器

将探索发现转换为结构化知识。

支持多种 LLM 后端：
- DeepSeek (API/本地部署)
- Ollama (本地开源模型)
- vLLM (高性能推理)
- 任何 OpenAI 兼容接口

学习的起点是 LLM 的现有知识库，通过分析探索发现生成新知识。
"""
from typing import Dict, List, Any, Optional
import json

from core.types import (
    KnowledgeContent, ProposedConstraint, LearningSignal,
    ExplorationStep, generate_id
)
from core.enums import LearningUnitType, DecisionType

# 支持旧的客户端（向后兼容）
from llm.client import DeepSeekClient, MockDeepSeekClient
from llm.prompts import PromptTemplates

# 新的适配器框架
from llm.adapters import BaseLLMAdapter, LLMAdapterFactory
from llm.adapters.base import MockLLMAdapter


class KnowledgeGenerator:
    """
    知识生成器
    
    特性：
    - 将探索发现转换为结构化知识
    - 生成提议的约束（不定义风险等级）
    - 生成学习信号
    - 支持多种 LLM 后端（通过适配器注入）
    
    LLM 注入方式：
    1. 直接传入适配器实例
    2. 通过工厂创建
    3. 使用旧的 DeepSeekClient（向后兼容）
    
    使用示例：
        # 方式1：使用适配器
        from llm.adapters import OllamaAdapter, LLMConfig
        adapter = OllamaAdapter(LLMConfig(model="llama3.2"))
        generator = KnowledgeGenerator(llm_adapter=adapter)
        
        # 方式2：使用工厂
        generator = KnowledgeGenerator.with_adapter("vllm", model="deepseek-coder")
        
        # 方式3：向后兼容
        from llm.client import DeepSeekClient
        client = DeepSeekClient(base_url="...")
        generator = KnowledgeGenerator(llm_client=client)
    """
    
    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        llm_adapter: Optional[BaseLLMAdapter] = None,
    ):
        """
        初始化知识生成器
        
        Args:
            llm_client: 旧的 LLM 客户端（向后兼容）
            llm_adapter: 新的 LLM 适配器（推荐）
        """
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
    
    @classmethod
    def with_adapter(
        cls,
        adapter_type: str,
        **adapter_kwargs
    ) -> "KnowledgeGenerator":
        """
        使用指定类型的适配器创建知识生成器
        
        Args:
            adapter_type: 适配器类型 ("deepseek", "ollama", "vllm", etc.)
            **adapter_kwargs: 传递给适配器的参数
            
        Returns:
            KnowledgeGenerator 实例
            
        示例：
            generator = KnowledgeGenerator.with_adapter(
                "ollama",
                model="llama3.2"
            )
        """
        adapter = LLMAdapterFactory.create_and_initialize(
            adapter_type, **adapter_kwargs
        )
        return cls(llm_adapter=adapter)
    
    def _call_llm_json(self, messages: List[Dict[str, str]], temperature: float = 0.5) -> Dict[str, Any]:
        """
        调用 LLM 并返回 JSON
        
        统一处理新旧两种 LLM 接口
        """
        if self._use_adapter and self._adapter:
            return self._adapter.chat_json(messages, temperature=temperature)
        elif self.llm:
            return self.llm.chat_json(messages, temperature=temperature)
        else:
            raise RuntimeError("No LLM configured")
    
    def generate(
        self,
        goal: str,
        exploration_path: List[Dict[str, Any]],
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        生成知识
        
        Args:
            goal: 学习目标
            exploration_path: 探索路径
            findings: 发现列表
            
        Returns:
            生成结果，包含 knowledge, proposed_constraints, signals
        """
        print(f"\n[知识生成] 基于 {len(findings)} 个发现生成知识")
        
        # 构建 prompt
        messages = [
            {"role": "system", "content": PromptTemplates.KNOWLEDGE_GENERATION_SYSTEM},
            {"role": "user", "content": PromptTemplates.format_knowledge_generation(
                goal=goal,
                exploration_path=exploration_path[-10:],  # 最近 10 步
                findings=findings
            )}
        ]
        
        try:
            response = self._call_llm_json(messages, temperature=0.5)
        except Exception as e:
            print(f"  [知识生成] LLM 调用失败: {e}")
            return self._generate_fallback(goal, findings)
        
        # 解析知识内容
        knowledge = self._parse_knowledge(response)
        
        # 解析提议的约束
        proposed_constraints = self._parse_constraints(response)
        
        # 生成学习信号
        signals = self._generate_signals(goal, exploration_path, findings, response)
        
        print(f"  [知识生成] 生成知识: {knowledge.domain}/{knowledge.type.value}")
        print(f"  [知识生成] 提议约束: {len(proposed_constraints)} 个")
        
        return {
            'knowledge': knowledge,
            'proposed_constraints': proposed_constraints,
            'signals': signals,
        }
    
    def _parse_knowledge(self, response: Dict[str, Any]) -> KnowledgeContent:
        """解析知识内容"""
        type_str = response.get('type', 'knowledge')
        try:
            knowledge_type = LearningUnitType(type_str)
        except ValueError:
            knowledge_type = LearningUnitType.KNOWLEDGE
        
        return KnowledgeContent(
            domain=response.get('domain', 'general'),
            type=knowledge_type,
            content=response.get('content', {}),
            confidence=response.get('confidence', 0.5),
            rationale=response.get('rationale', '')
        )
    
    def _parse_constraints(
        self,
        response: Dict[str, Any]
    ) -> List[ProposedConstraint]:
        """解析提议的约束"""
        constraints = []
        
        for c_data in response.get('proposed_constraints', []):
            decision_str = c_data.get('proposed_decision', 'REVIEW')
            try:
                decision = DecisionType[decision_str]
            except KeyError:
                decision = DecisionType.REVIEW
            
            constraint = ProposedConstraint(
                constraint_id=c_data.get('constraint_id', generate_id("constraint")),
                condition=c_data.get('condition', 'True'),
                proposed_decision=decision,
                rationale=c_data.get('rationale', ''),
                confidence=c_data.get('confidence', 0.8)
            )
            constraints.append(constraint)
        
        return constraints
    
    def _generate_signals(
        self,
        goal: str,
        exploration_path: List[Dict[str, Any]],
        findings: List[Dict[str, Any]],
        response: Dict[str, Any]
    ) -> List[LearningSignal]:
        """生成学习信号"""
        signals = []
        
        # 知识来源信号
        signals.append(LearningSignal(
            signal_type="knowledge_source",
            content={
                'source': 'llm_exploration',
                'goal': goal,
                'path_length': len(exploration_path),
            }
        ))
        
        # 推理链路信号
        signals.append(LearningSignal(
            signal_type="reasoning_chain",
            content={
                'steps': len(exploration_path),
                'findings': len(findings),
            }
        ))
        
        # 置信度信号
        signals.append(LearningSignal(
            signal_type="confidence_score",
            content={
                'confidence': response.get('confidence', 0.5),
                'rationale': response.get('rationale', ''),
            }
        ))
        
        # LLM 信息信号
        llm_info = self.get_llm_info()
        signals.append(LearningSignal(
            signal_type="llm_info",
            content={
                'adapter': llm_info.get('adapter_name', 'unknown'),
                'model': llm_info.get('model', 'unknown'),
            }
        ))
        
        return signals
    
    def _generate_fallback(
        self,
        goal: str,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """生成回退结果（当 LLM 调用失败时）"""
        # 从发现中提取知识
        content = {}
        if findings:
            content = {
                'summary': f"基于 {len(findings)} 个发现",
                'findings': findings[:5],
            }
        
        knowledge = KnowledgeContent(
            domain='general',
            type=LearningUnitType.KNOWLEDGE,
            content=content,
            confidence=0.3,
            rationale=f"自动生成（LLM 调用失败）: {goal}"
        )
        
        signals = [
            LearningSignal(
                signal_type="fallback_generation",
                content={'reason': 'LLM call failed'}
            )
        ]
        
        return {
            'knowledge': knowledge,
            'proposed_constraints': [],
            'signals': signals,
        }
    
    def get_llm_info(self) -> Dict[str, Any]:
        """获取当前 LLM 信息"""
        if self._use_adapter and self._adapter:
            return self._adapter.get_statistics()
        elif self.llm:
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
        print(f"[KnowledgeGenerator] LLM 适配器已切换为: {adapter.adapter_name}")
