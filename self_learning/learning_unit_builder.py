"""
Learning Unit 构建器

将探索结果打包为 Learning Unit，提交给审计系统
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from core.types import (
    LearningUnit, Provenance, ExplorationStep,
    KnowledgeContent, ProposedConstraint, LearningSignal,
    generate_id
)
from core.enums import ExplorationAction, AuditStatus

from .explorer import AutonomousExplorer
from .knowledge_generator import KnowledgeGenerator
from .checkpoint import CheckpointManager


class LearningUnitBuilder:
    """
    Learning Unit 构建器
    
    特性：
    - 整合探索、知识生成、检查点
    - 构建完整的 Learning Unit
    - 提交给审计系统（不直接写入生产）
    
    注意：
    - 自学习系统无权定义风险等级
    - 自学习系统无权直接写入生产桥接器
    """
    
    def __init__(
        self,
        explorer: AutonomousExplorer,
        knowledge_generator: KnowledgeGenerator,
        checkpoint_manager: CheckpointManager,
        submit_callback: Optional[Callable[[LearningUnit], None]] = None
    ):
        self.explorer = explorer
        self.knowledge_generator = knowledge_generator
        self.checkpoint_manager = checkpoint_manager
        self.submit_callback = submit_callback
        
        # 已构建的 Learning Units
        self.built_units: List[LearningUnit] = []
    
    def build_from_exploration(
        self,
        goal: str,
        initial_context: Optional[Dict[str, Any]] = None
    ) -> Optional[LearningUnit]:
        """
        从探索构建 Learning Unit
        
        Args:
            goal: 学习目标
            initial_context: 初始上下文
            
        Returns:
            构建的 Learning Unit，如果失败则返回 None
        """
        print(f"\n{'='*60}")
        print(f"[Learning Unit Builder] 开始构建")
        print(f"  目标: {goal}")
        print(f"{'='*60}")
        
        # 1. 执行探索
        exploration_result = self.explorer.explore(goal, initial_context)
        
        if exploration_result['status'] != 'completed':
            print(f"[Builder] 探索失败: {exploration_result.get('error', 'Unknown')}")
            return None
        
        # 2. 生成知识
        generation_result = self.knowledge_generator.generate(
            goal=goal,
            exploration_path=exploration_result['exploration_path'],
            findings=exploration_result['findings']
        )
        
        # 3. 构建 Learning Unit
        unit = self._build_unit(
            goal=goal,
            exploration_result=exploration_result,
            generation_result=generation_result
        )
        
        # 4. 创建检查点
        self.checkpoint_manager.create_checkpoint(
            exploration_data={
                'goal': goal,
                'unit_id': unit.id,
                'exploration_path': exploration_result['exploration_path'],
                'findings': exploration_result['findings'],
            },
            reason=f"Learning Unit 构建完成: {unit.id}"
        )
        
        # 5. 记录
        self.built_units.append(unit)
        
        # 6. 提交给审计系统
        if self.submit_callback:
            print(f"\n[Builder] 提交 Learning Unit 到审计系统: {unit.id}")
            self.submit_callback(unit)
        
        print(f"\n{'='*60}")
        print(f"[Builder] Learning Unit 构建完成: {unit.id}")
        print(f"  领域: {unit.knowledge.domain if unit.knowledge else 'N/A'}")
        print(f"  类型: {unit.knowledge.type.value if unit.knowledge else 'N/A'}")
        print(f"  约束: {len(unit.proposed_constraints)} 个")
        print(f"  状态: {unit.audit_status.value}")
        print(f"{'='*60}")
        
        return unit
    
    def _build_unit(
        self,
        goal: str,
        exploration_result: Dict[str, Any],
        generation_result: Dict[str, Any]
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
        
        # 构建来源追溯
        provenance = Provenance(
            learning_goal=goal,
            exploration_path=exploration_steps,
            base_knowledge_refs=[]  # 可以添加 LLM 知识引用
        )
        
        # 构建 Learning Unit
        unit = LearningUnit(
            id=unit_id,
            version="1.0.0",
            source="self_learning_system",
            provenance=provenance,
            knowledge=generation_result.get('knowledge'),
            proposed_constraints=generation_result.get('proposed_constraints', []),
            signals=generation_result.get('signals', []),
            audit_status=AuditStatus.PENDING,
            # 注意：risk_level 为 None，由审计系统定义
        )
        
        return unit
    
    def build_from_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[LearningUnit]:
        """
        从检查点恢复并构建 Learning Unit
        
        Args:
            checkpoint_id: 检查点 ID
            
        Returns:
            构建的 Learning Unit
        """
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        
        if checkpoint is None:
            print(f"[Builder] 检查点未找到: {checkpoint_id}")
            return None
        
        exploration_data = checkpoint.get('exploration_data', {})
        
        # 生成知识
        generation_result = self.knowledge_generator.generate(
            goal=exploration_data.get('goal', ''),
            exploration_path=exploration_data.get('exploration_path', []),
            findings=exploration_data.get('findings', [])
        )
        
        # 构建 Learning Unit
        unit = self._build_unit(
            goal=exploration_data.get('goal', ''),
            exploration_result=exploration_data,
            generation_result=generation_result
        )
        
        self.built_units.append(unit)
        
        if self.submit_callback:
            self.submit_callback(unit)
        
        return unit
    
    def get_built_units(self) -> List[Dict[str, Any]]:
        """获取已构建的 Learning Units"""
        return [unit.to_dict() for unit in self.built_units]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_built': len(self.built_units),
            'pending_audit': len([u for u in self.built_units if u.audit_status == AuditStatus.PENDING]),
            'approved': len([u for u in self.built_units if u.audit_status == AuditStatus.APPROVED]),
        }

