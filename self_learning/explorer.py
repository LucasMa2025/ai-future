"""
自主探索引擎

基于 LLM 的无用户参与的自主学习。

支持多种 LLM 后端：
- DeepSeek (API/本地部署)
- Ollama (本地开源模型)
- vLLM (高性能推理)
- 任何 OpenAI 兼容接口

学习的起点是 LLM 的现有知识库，通过探索发现新知识。
"""
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
import json

from core.types import ExplorationStep, generate_id
from core.enums import ExplorationAction
from core.exceptions import ExplorationFailed

# 支持旧的客户端（向后兼容）
from llm.client import DeepSeekClient, MockDeepSeekClient
from llm.prompts import PromptTemplates

# 新的适配器框架
from llm.adapters import BaseLLMAdapter, LLMAdapterFactory
from llm.adapters.base import MockLLMAdapter


class AutonomousExplorer:
    """
    自主探索引擎
    
    特性：
    - 基于 LLM 的自主探索
    - 无用户参与
    - 自动触发 Checkpoint
    - 探索路径记录
    - 支持多种 LLM 后端（通过适配器注入）
    
    LLM 注入方式：
    1. 直接传入适配器实例
    2. 通过工厂创建
    3. 使用旧的 DeepSeekClient（向后兼容）
    
    使用示例：
        # 方式1：使用适配器
        from llm.adapters import DeepSeekAdapter, LLMConfig
        adapter = DeepSeekAdapter(LLMConfig(base_url="..."))
        explorer = AutonomousExplorer(llm_adapter=adapter)
        
        # 方式2：使用工厂
        explorer = AutonomousExplorer.with_adapter("ollama", model="llama3.2")
        
        # 方式3：向后兼容
        from llm.client import DeepSeekClient
        client = DeepSeekClient(base_url="...")
        explorer = AutonomousExplorer(llm_client=client)
    """
    
    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        max_depth: int = 5,
        max_iterations: int = 100,
        checkpoint_callback: Optional[Callable] = None
    ):
        """
        初始化自主探索引擎
        
        Args:
            llm_client: 旧的 LLM 客户端（向后兼容）
            llm_adapter: 新的 LLM 适配器（推荐）
            max_depth: 最大探索深度
            max_iterations: 最大迭代次数
            checkpoint_callback: Checkpoint 回调
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
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.checkpoint_callback = checkpoint_callback
        
        # 探索状态
        self.current_goal: str = ""
        self.exploration_path: List[ExplorationStep] = []
        self.findings: List[Dict[str, Any]] = []
        self.current_depth: int = 0
        self.iteration_count: int = 0
        
        # 是否正在探索
        self.is_exploring: bool = False
    
    def explore(
        self,
        goal: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        开始自主探索
        
        Args:
            goal: 学习目标
            initial_context: 初始上下文
            
        Returns:
            探索结果
        """
        print(f"\n[自主探索] 开始探索: {goal}")
        
        # 初始化状态
        self.current_goal = goal
        self.exploration_path = []
        self.findings = []
        self.current_depth = 0
        self.iteration_count = 0
        self.is_exploring = True
        
        checkpoints_triggered = []
        
        try:
            while self.is_exploring and self.iteration_count < self.max_iterations:
                self.iteration_count += 1
                
                # 执行一步探索
                step_result = self._explore_step()
                
                if step_result is None:
                    break
                
                # 检查是否需要 Checkpoint
                if step_result.get('should_checkpoint', False):
                    checkpoint_result = self._trigger_checkpoint(
                        step_result.get('checkpoint_reason', 'Auto checkpoint')
                    )
                    checkpoints_triggered.append(checkpoint_result)
                
                # 检查是否完成
                if self._should_stop(step_result):
                    break
            
            return {
                'status': 'completed',
                'goal': goal,
                'exploration_path': [s.to_dict() for s in self.exploration_path],
                'findings': self.findings,
                'iterations': self.iteration_count,
                'checkpoints': checkpoints_triggered,
            }
        
        except Exception as e:
            return {
                'status': 'failed',
                'goal': goal,
                'error': str(e),
                'exploration_path': [s.to_dict() for s in self.exploration_path],
                'findings': self.findings,
            }
        
        finally:
            self.is_exploring = False
    
    @classmethod
    def with_adapter(
        cls,
        adapter_type: str,
        max_depth: int = 5,
        max_iterations: int = 100,
        checkpoint_callback: Optional[Callable] = None,
        **adapter_kwargs
    ) -> "AutonomousExplorer":
        """
        使用指定类型的适配器创建探索器
        
        Args:
            adapter_type: 适配器类型 ("deepseek", "ollama", "vllm", etc.)
            max_depth: 最大探索深度
            max_iterations: 最大迭代次数
            checkpoint_callback: Checkpoint 回调
            **adapter_kwargs: 传递给适配器的参数
            
        Returns:
            AutonomousExplorer 实例
            
        示例：
            explorer = AutonomousExplorer.with_adapter(
                "ollama",
                model="llama3.2",
                max_depth=10
            )
        """
        adapter = LLMAdapterFactory.create_and_initialize(
            adapter_type, **adapter_kwargs
        )
        return cls(
            llm_adapter=adapter,
            max_depth=max_depth,
            max_iterations=max_iterations,
            checkpoint_callback=checkpoint_callback,
        )
    
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
            raise RuntimeError("No LLM configured")
    
    def _explore_step(self) -> Optional[Dict[str, Any]]:
        """执行一步探索"""
        # 构建 prompt
        messages = [
            {"role": "system", "content": PromptTemplates.EXPLORATION_SYSTEM},
            {"role": "user", "content": PromptTemplates.format_exploration(
                goal=self.current_goal,
                depth=self.current_depth,
                findings=self.findings[-5:] if self.findings else []  # 最近 5 个发现
            )}
        ]
        
        try:
            response = self._call_llm_json(messages, temperature=0.7)
        except Exception as e:
            print(f"  [探索] LLM 调用失败: {e}")
            return None
        
        # 解析响应
        action_str = response.get('action', 'reasoning')
        try:
            action = ExplorationAction(action_str)
        except ValueError:
            action = ExplorationAction.REASONING
        
        # 记录探索步骤
        step = ExplorationStep(
            step_id=generate_id("step"),
            action=action,
            query=response.get('query', ''),
            result=json.dumps(response.get('findings', []), ensure_ascii=False)
        )
        self.exploration_path.append(step)
        
        # 记录发现
        for finding in response.get('findings', []):
            self.findings.append({
                'step_id': step.step_id,
                'depth': self.current_depth,
                **finding
            })
        
        # 更新深度
        if action in [ExplorationAction.REASONING, ExplorationAction.HYPOTHESIS]:
            self.current_depth += 1
        
        print(f"  [探索] Step {self.iteration_count}: {action.value} - 发现 {len(response.get('findings', []))} 项")
        
        return response
    
    def _should_stop(self, step_result: Dict[str, Any]) -> bool:
        """判断是否应该停止探索"""
        # 达到最大深度
        if self.current_depth >= self.max_depth:
            print(f"  [探索] 达到最大深度 {self.max_depth}")
            return True
        
        # 没有更多发现
        if not step_result.get('findings') and not step_result.get('next_steps'):
            print(f"  [探索] 没有更多发现")
            return True
        
        # 发现足够多
        if len(self.findings) >= 10:
            print(f"  [探索] 发现足够多 ({len(self.findings)})")
            return True
        
        return False
    
    def _trigger_checkpoint(self, reason: str) -> Dict[str, Any]:
        """触发 Checkpoint"""
        print(f"  [Checkpoint] 触发: {reason}")
        
        checkpoint_data = {
            'checkpoint_id': generate_id("ckpt"),
            'goal': self.current_goal,
            'reason': reason,
            'exploration_path': [s.to_dict() for s in self.exploration_path],
            'findings': self.findings.copy(),
            'depth': self.current_depth,
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat(),
        }
        
        # 调用回调
        if self.checkpoint_callback:
            self.checkpoint_callback(checkpoint_data)
        
        return checkpoint_data
    
    def stop(self):
        """停止探索"""
        self.is_exploring = False
    
    def get_exploration_summary(self) -> Dict[str, Any]:
        """获取探索摘要"""
        return {
            'goal': self.current_goal,
            'is_exploring': self.is_exploring,
            'current_depth': self.current_depth,
            'iteration_count': self.iteration_count,
            'findings_count': len(self.findings),
            'path_length': len(self.exploration_path),
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
        print(f"[Explorer] LLM 适配器已切换为: {adapter.adapter_name}")

