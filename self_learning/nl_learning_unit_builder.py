"""
增强版 Learning Unit 构建器

集成 Nested Learning 框架，支持：
1. LearningScope 控制
2. ContextFlowSegment 映射
3. 多层级状态追踪
4. 与 NLGSM 治理系统的完整对接
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import uuid

from core.types import (
    LearningUnit, Provenance, ExplorationStep,
    KnowledgeContent, ProposedConstraint, LearningSignal,
    generate_id
)
from core.enums import ExplorationAction, AuditStatus

from .explorer import AutonomousExplorer
from .knowledge_generator import KnowledgeGenerator
from .checkpoint import CheckpointManager
from .nl_core import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    NestedLearningKernel,
    LLMBasedNLKernel,
    ContinuumMemorySystem,
)


class NLLearningUnitBuilder:
    """
    Nested Learning 增强版 Learning Unit 构建器
    
    核心治理原则：
    1. 所有学习必须在 Scope 控制下进行
    2. 自学习系统无权定义风险等级（由 Scope 决定最高级别）
    3. 自学习系统无权直接写入生产桥接器
    4. 所有变更产生 ContextFlowSegment 用于审计
    
    与原版 LearningUnitBuilder 的区别：
    - 集成 NL 内核
    - 支持多层级状态追踪
    - 产生 ContextFlowSegment 而非简单的 ExplorationStep
    """
    
    def __init__(
        self,
        explorer: AutonomousExplorer,
        knowledge_generator: KnowledgeGenerator,
        checkpoint_manager: CheckpointManager,
        nl_kernel: Optional[NestedLearningKernel] = None,
        submit_callback: Optional[Callable[[LearningUnit], None]] = None,
        scope_provider: Optional[Callable[[], LearningScope]] = None,
    ):
        self.explorer = explorer
        self.knowledge_generator = knowledge_generator
        self.checkpoint_manager = checkpoint_manager
        self.submit_callback = submit_callback
        
        # NL 内核
        self.nl_kernel = nl_kernel or LLMBasedNLKernel()
        if not self.nl_kernel._initialized:
            self.nl_kernel.initialize({})
        
        # Scope 提供者（由治理系统提供）
        self.scope_provider = scope_provider or self._default_scope_provider
        
        # 当前学习会话
        self.current_scope: Optional[LearningScope] = None
        self.current_session_id: Optional[str] = None
        
        # 构建历史
        self.built_units: List[LearningUnit] = []
        self.context_flow_segments: List[ContextFlowSegment] = []
    
    def _default_scope_provider(self) -> LearningScope:
        """默认 Scope 提供者（仅用于测试）"""
        return LearningScope(
            scope_id=f"scope_{uuid.uuid4().hex[:8]}",
            max_level=NLLevel.MEMORY,
            allowed_levels=[NLLevel.PARAMETER, NLLevel.MEMORY],
            created_by="default_provider",
        )
    
    def start_learning_session(
        self,
        scope: Optional[LearningScope] = None
    ) -> str:
        """
        开始学习会话
        
        必须由治理系统调用，提供 Scope
        """
        self.current_scope = scope or self.scope_provider()
        self.current_session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # 解冻 NL 内核
        self.nl_kernel.unfreeze()
        
        print(f"\n{'='*60}")
        print(f"[NL Learning Session] Started: {self.current_session_id}")
        print(f"  Scope: {self.current_scope.scope_id}")
        print(f"  Max Level: {self.current_scope.max_level.name}")
        print(f"  Risk Level: {self.current_scope.get_risk_level()}")
        print(f"{'='*60}")
        
        return self.current_session_id
    
    def end_learning_session(self) -> Dict[str, Any]:
        """
        结束学习会话
        
        冻结 NL 内核，创建快照
        """
        if not self.current_session_id:
            return {"error": "No active session"}
        
        # 冻结 NL 内核
        self.nl_kernel.freeze()
        
        # 创建快照
        snapshot = self.nl_kernel.create_snapshot()
        
        summary = {
            "session_id": self.current_session_id,
            "scope_id": self.current_scope.scope_id if self.current_scope else None,
            "snapshot_id": snapshot.state_id,
            "segments_count": len(self.context_flow_segments),
            "units_built": len(self.built_units),
            "budget_used": self.current_scope.budget_used if self.current_scope else {},
        }
        
        print(f"\n{'='*60}")
        print(f"[NL Learning Session] Ended: {self.current_session_id}")
        print(f"  Segments: {summary['segments_count']}")
        print(f"  Units Built: {summary['units_built']}")
        print(f"{'='*60}")
        
        self.current_session_id = None
        self.current_scope = None
        
        return summary
    
    def build_from_exploration(
        self,
        goal: str,
        initial_context: Optional[Dict[str, Any]] = None,
        scope: Optional[LearningScope] = None
    ) -> Optional[LearningUnit]:
        """
        从探索构建 Learning Unit（NL 增强版）
        
        Args:
            goal: 学习目标
            initial_context: 初始上下文
            scope: 学习范围（如果没有活跃会话则必须提供）
            
        Returns:
            构建的 Learning Unit
        """
        # 确保有 Scope
        if scope:
            self.current_scope = scope
        elif not self.current_scope:
            self.current_scope = self.scope_provider()
        
        # 确保有活跃会话
        if not self.current_session_id:
            self.start_learning_session(self.current_scope)
        
        print(f"\n{'='*60}")
        print(f"[NL Learning Unit Builder] 开始构建")
        print(f"  目标: {goal}")
        print(f"  Scope: {self.current_scope.scope_id}")
        print(f"  Max Level: {self.current_scope.max_level.name}")
        print(f"{'='*60}")
        
        # 1. 执行探索
        exploration_result = self.explorer.explore(goal, initial_context)
        
        if exploration_result['status'] != 'completed':
            print(f"[Builder] 探索失败: {exploration_result.get('error', 'Unknown')}")
            return None
        
        # 2. 执行 NL 学习步骤
        context_flow_segments = self._execute_nl_learning(
            goal=goal,
            exploration_result=exploration_result,
        )
        
        # 3. 生成知识
        generation_result = self.knowledge_generator.generate(
            goal=goal,
            exploration_path=exploration_result['exploration_path'],
            findings=exploration_result['findings']
        )
        
        # 4. 构建 Learning Unit
        unit = self._build_unit(
            goal=goal,
            exploration_result=exploration_result,
            generation_result=generation_result,
            context_flow_segments=context_flow_segments,
        )
        
        # 5. 创建检查点
        self.checkpoint_manager.create_checkpoint(
            exploration_data={
                'goal': goal,
                'unit_id': unit.id,
                'session_id': self.current_session_id,
                'scope': self.current_scope.to_dict(),
                'segments_count': len(context_flow_segments),
            },
            reason=f"NL Learning Unit 构建完成: {unit.id}"
        )
        
        # 6. 记录
        self.built_units.append(unit)
        
        # 7. 提交给审计系统
        if self.submit_callback:
            print(f"\n[Builder] 提交 Learning Unit 到审计系统: {unit.id}")
            self.submit_callback(unit)
        
        self._print_build_summary(unit, context_flow_segments)
        
        return unit
    
    def _execute_nl_learning(
        self,
        goal: str,
        exploration_result: Dict[str, Any],
    ) -> List[ContextFlowSegment]:
        """
        执行 NL 学习步骤
        
        将探索结果转换为 NL ContextFlowSegment
        """
        segments = []
        
        findings = exploration_result.get('findings', [])
        exploration_path = exploration_result.get('exploration_path', [])
        
        # 为每个探索步骤执行 NL 学习
        for i, step in enumerate(exploration_path):
            # 构建 NL 上下文
            context = {
                "goal": goal,
                "step": i,
                "action": step.get('action', 'reasoning'),
                "query": step.get('query', ''),
                "result": step.get('result', ''),
                "findings": findings[:i+1],  # 到当前步骤的发现
                # 模拟梯度信息
                "gradient_norm": 0.1 * (i + 1),
                "loss_history": [1.0 / (j + 1) for j in range(i + 1)],
            }
            
            # 执行 NL 学习步骤
            try:
                segment = self.nl_kernel.execute_learning_step(
                    context=context,
                    scope=self.current_scope,
                )
                segments.append(segment)
                self.context_flow_segments.append(segment)
            except Exception as e:
                print(f"  [NL] Step {i} failed: {e}")
        
        return segments
    
    def _build_unit(
        self,
        goal: str,
        exploration_result: Dict[str, Any],
        generation_result: Dict[str, Any],
        context_flow_segments: List[ContextFlowSegment],
    ) -> LearningUnit:
        """构建 Learning Unit"""
        unit_id = generate_id("lu")
        
        # 构建探索路径
        exploration_steps = []
        for step_data in exploration_result.get('exploration_path', []):
            try:
                action = ExplorationAction(step_data.get('action', 'reasoning'))
            except ValueError:
                action = ExplorationAction.REASONING
            
            step = ExplorationStep(
                step_id=step_data.get('step_id', generate_id("step")),
                action=action,
                query=step_data.get('query', ''),
                result=step_data.get('result', '')
            )
            exploration_steps.append(step)
        
        # 构建来源追溯（增强版）
        provenance = Provenance(
            learning_goal=goal,
            exploration_path=exploration_steps,
            base_knowledge_refs=[]
        )
        
        # 构建学习信号（增加 NL 信号）
        signals = generation_result.get('signals', [])
        
        # 添加 NL 特定信号
        signals.extend([
            LearningSignal(
                signal_type="nl_session",
                content={
                    "session_id": self.current_session_id,
                    "scope_id": self.current_scope.scope_id if self.current_scope else None,
                    "max_level": self.current_scope.max_level.name if self.current_scope else None,
                }
            ),
            LearningSignal(
                signal_type="nl_segments",
                content={
                    "count": len(context_flow_segments),
                    "affected_levels": list(set(
                        level.name
                        for seg in context_flow_segments
                        for level in seg.get_affected_levels()
                    )),
                }
            ),
            LearningSignal(
                signal_type="nl_risk_assessment",
                content={
                    "scope_risk_level": self.current_scope.get_risk_level() if self.current_scope else "low",
                    "max_segment_risk": max(
                        (seg.get_max_risk_level() for seg in context_flow_segments),
                        default="low"
                    ),
                }
            ),
        ])
        
        # 构建 Learning Unit
        unit = LearningUnit(
            id=unit_id,
            version="2.0.0",  # NL 增强版本
            source="nl_self_learning_system",
            provenance=provenance,
            knowledge=generation_result.get('knowledge'),
            proposed_constraints=generation_result.get('proposed_constraints', []),
            signals=signals,
            audit_status=AuditStatus.PENDING,
            # 注意：risk_level 仍为 None，由审计系统根据 Scope 和 Segment 确定
        )
        
        return unit
    
    def _print_build_summary(
        self,
        unit: LearningUnit,
        segments: List[ContextFlowSegment]
    ):
        """打印构建摘要"""
        print(f"\n{'='*60}")
        print(f"[NL Builder] Learning Unit 构建完成: {unit.id}")
        print(f"  版本: {unit.version}")
        print(f"  领域: {unit.knowledge.domain if unit.knowledge else 'N/A'}")
        print(f"  类型: {unit.knowledge.type.value if unit.knowledge else 'N/A'}")
        print(f"  约束: {len(unit.proposed_constraints)} 个")
        print(f"  状态: {unit.audit_status.value}")
        print(f"  NL Segments: {len(segments)} 个")
        
        # 层级影响统计
        level_counts = {}
        for seg in segments:
            for level in seg.get_affected_levels():
                level_counts[level.name] = level_counts.get(level.name, 0) + 1
        
        print(f"  层级影响: {level_counts}")
        
        if self.current_scope:
            print(f"  Scope 风险等级: {self.current_scope.get_risk_level()}")
            print(f"  预算使用: {dict(self.current_scope.budget_used)}")
        
        print(f"{'='*60}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_built": len(self.built_units),
            "total_segments": len(self.context_flow_segments),
            "pending_audit": len([
                u for u in self.built_units 
                if u.audit_status == AuditStatus.PENDING
            ]),
            "approved": len([
                u for u in self.built_units 
                if u.audit_status == AuditStatus.APPROVED
            ]),
            "current_session": self.current_session_id,
            "kernel_stats": self.nl_kernel.get_statistics() if hasattr(self.nl_kernel, 'get_statistics') else {},
        }
    
    def get_nl_state(self) -> Dict[str, Any]:
        """获取 NL 内核状态"""
        return {
            "is_frozen": self.nl_kernel.is_frozen() if hasattr(self.nl_kernel, 'is_frozen') else True,
            "memory_state": self.nl_kernel.get_memory_state().to_dict(),
            "current_scope": self.current_scope.to_dict() if self.current_scope else None,
        }

