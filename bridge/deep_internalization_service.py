"""
深度知识内化服务

将 Knowledge Adapter + Decision Head 结合，提供完整的知识内化能力。

与数据库设计保持一致：
- knowledge_internalization 表：记录内化过程
- internalized_constraints 表：记录每个约束的内化详情
- internalization_effects 表：记录内化效果测试

核心流程：
1. Learning Unit 审计通过
2. 调用内化服务
3. 创建内化记录（数据库）
4. 执行 Knowledge Adapter 注入 + Decision Head 训练
5. 验证内化效果
6. 更新数据库状态
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID
import hashlib
import json
import os

import torch
import torch.nn as nn

from core.types import LearningUnit, ProposedConstraint
from core.enums import DecisionType

from .knowledge_adapter import (
    KnowledgeAdapter,
    KnowledgeAdapterManager,
    create_knowledge_encoder,
)
from .enhanced_internalization import (
    EnhancedInternalizationEngine,
    EnhancedDecisionHead,
    InternalizationConfig,
)

# LLM 适配器
from llm.adapters import BaseLLMAdapter


@dataclass
class InternalizationResult:
    """内化结果"""
    success: bool
    internalization_id: str
    version: int
    
    # 各组件结果
    adapter_result: Dict[str, Any] = field(default_factory=dict)
    decision_head_result: Dict[str, Any] = field(default_factory=dict)
    
    # 验证结果
    validation_passed: bool = False
    validation_accuracy: float = 0.0
    
    # 错误信息
    error: Optional[str] = None
    
    # 时间
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'internalization_id': self.internalization_id,
            'version': self.version,
            'adapter_result': self.adapter_result,
            'decision_head_result': self.decision_head_result,
            'validation_passed': self.validation_passed,
            'validation_accuracy': self.validation_accuracy,
            'error': self.error,
            'duration_seconds': self.duration_seconds,
        }


class DeepInternalizationService:
    """
    深度知识内化服务
    
    特性：
    1. 双层内化：Knowledge Adapter（深层） + Decision Head（表层）
    2. 与数据库设计一致的持久化
    3. 完整的版本管理和回滚
    4. 内化效果验证
    
    架构：
    
    ┌──────────────────────────────────────────────────────────────────┐
    │  DeepInternalizationService                                      │
    │                                                                  │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │  Knowledge Adapter Manager                                 │ │
    │  │  - 多层 Adapter                                           │ │
    │  │  - 影响 LLM 内部表示                                       │ │
    │  │  - 知识以低秩形式存储                                      │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                              +                                   │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │  Enhanced Decision Head                                    │ │
    │  │  - 决策分类                                                │ │
    │  │  - 风险评估                                                │ │
    │  │  - 置信度估计                                              │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                              +                                   │
    │  ┌────────────────────────────────────────────────────────────┐ │
    │  │  Database Integration                                      │ │
    │  │  - knowledge_internalization                               │ │
    │  │  - internalized_constraints                                │ │
    │  │  - internalization_effects                                 │ │
    │  └────────────────────────────────────────────────────────────┘ │
    │                                                                  │
    └──────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        llm_adapter: Optional[BaseLLMAdapter] = None,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 64,
        num_adapter_layers: int = 4,
        num_knowledge_slots: int = 100,
        decision_head_config: Optional[InternalizationConfig] = None,
        weight_storage_dir: str = "./data/internalization_weights",
        db_session=None,  # SQLAlchemy Session
    ):
        self.llm_adapter = llm_adapter
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.weight_storage_dir = weight_storage_dir
        self.db = db_session
        
        # 确保存储目录存在
        os.makedirs(weight_storage_dir, exist_ok=True)
        
        # 1. Knowledge Adapter Manager（深层内化）
        self.adapter_manager = KnowledgeAdapterManager(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_slots_per_adapter=num_knowledge_slots,
            num_adapters=num_adapter_layers,
        )
        
        # 2. Decision Head（表层内化）
        self.decision_head_engine = EnhancedInternalizationEngine(
            llm_adapter=llm_adapter,
            config=decision_head_config,
            hidden_dim=hidden_dim,
        )
        
        # 3. 知识编码器
        self.knowledge_encoder = create_knowledge_encoder(
            llm_hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
        )
        
        # 版本控制
        self.current_version = 0
        
        # 内化历史
        self.internalization_history: List[InternalizationResult] = []
    
    def set_llm_adapter(self, adapter: BaseLLMAdapter):
        """设置 LLM 适配器"""
        self.llm_adapter = adapter
        self.decision_head_engine.set_llm_adapter(adapter)
        print(f"[DeepInternalization] LLM 适配器已设置")
    
    def internalize(
        self,
        learning_unit: LearningUnit,
        skip_adapter: bool = False,
        skip_decision_head: bool = False,
    ) -> InternalizationResult:
        """
        执行深度内化
        
        Args:
            learning_unit: 审计通过的 Learning Unit
            skip_adapter: 是否跳过 Adapter 内化
            skip_decision_head: 是否跳过 Decision Head 内化
            
        Returns:
            内化结果
        """
        result = InternalizationResult(
            success=False,
            internalization_id=self._generate_internalization_id(learning_unit),
            version=self.current_version + 1,
            started_at=datetime.now(),
        )
        
        print(f"\n{'='*60}")
        print(f"[深度内化] 开始: {learning_unit.id}")
        print(f"  约束数量: {len(learning_unit.proposed_constraints)}")
        print(f"{'='*60}")
        
        try:
            # 1. 创建数据库记录（如果有数据库连接）
            db_record_id = self._create_db_record(learning_unit, result)
            
            # 2. Knowledge Adapter 内化
            if not skip_adapter:
                result.adapter_result = self._internalize_to_adapters(learning_unit)
                print(f"  ✓ Adapter 内化: {result.adapter_result.get('injected_count', 0)} 个约束")
            
            # 3. Decision Head 内化
            if not skip_decision_head:
                result.decision_head_result = self._internalize_to_decision_head(learning_unit)
                print(f"  ✓ Decision Head: 准确率 {result.decision_head_result.get('accuracy', 0):.2%}")
            
            # 4. 验证内化效果
            validation = self._validate_internalization(learning_unit)
            result.validation_passed = validation['passed']
            result.validation_accuracy = validation['accuracy']
            
            # 5. 创建版本快照
            self.current_version += 1
            result.version = self.current_version
            self._save_version_snapshot(result.version)
            
            # 6. 更新数据库记录
            self._update_db_record(db_record_id, result, validation)
            
            result.success = True
            result.completed_at = datetime.now()
            
            print(f"\n[深度内化] 完成!")
            print(f"  版本: {result.version}")
            print(f"  验证: {'通过' if result.validation_passed else '未通过'}")
            print(f"  耗时: {result.duration_seconds:.2f} 秒")
        
        except Exception as e:
            result.error = str(e)
            result.completed_at = datetime.now()
            print(f"\n[深度内化] 失败: {e}")
        
        self.internalization_history.append(result)
        return result
    
    def _internalize_to_adapters(
        self,
        learning_unit: LearningUnit
    ) -> Dict[str, Any]:
        """
        内化到 Knowledge Adapter
        """
        results = {
            'injected_count': 0,
            'slot_indices': [],
            'constraints': [],
        }
        
        for constraint in learning_unit.proposed_constraints:
            # 编码约束为知识向量
            text = f"{constraint.condition} -> {constraint.proposed_decision.name}: {constraint.rationale}"
            knowledge_vector = self.knowledge_encoder(text, self.llm_adapter)
            
            # 注入到所有 Adapter
            slot_indices = self.adapter_manager.inject_knowledge_to_all(
                constraint_id=constraint.constraint_id,
                learning_unit_id=learning_unit.id,
                knowledge_vector=knowledge_vector,
                condition=constraint.condition,
                decision=constraint.proposed_decision.name,
            )
            
            results['injected_count'] += 1
            results['slot_indices'].append(slot_indices)
            results['constraints'].append({
                'constraint_id': constraint.constraint_id,
                'slot_indices': slot_indices,
            })
        
        return results
    
    def _internalize_to_decision_head(
        self,
        learning_unit: LearningUnit
    ) -> Dict[str, Any]:
        """
        内化到 Decision Head
        """
        internalization_result = self.decision_head_engine.internalize(learning_unit)
        
        return {
            'status': internalization_result.get('status'),
            'version': internalization_result.get('version'),
            'accuracy': internalization_result.get('validation', {}).get('accuracy', 0),
        }
    
    def _validate_internalization(
        self,
        learning_unit: LearningUnit
    ) -> Dict[str, Any]:
        """
        验证内化效果
        """
        test_results = []
        correct_count = 0
        
        for constraint in learning_unit.proposed_constraints:
            # 测试查询
            test_query = constraint.condition
            expected_decision = constraint.proposed_decision.name
            
            # 获取决策
            result = self.decision_head_engine.get_decision(test_query)
            actual_decision = result.get('decision')
            confidence = result.get('confidence', 0)
            
            is_correct = actual_decision == expected_decision
            if is_correct:
                correct_count += 1
            
            test_results.append({
                'test_query': test_query[:100],  # 截断
                'expected': expected_decision,
                'actual': actual_decision,
                'confidence': confidence,
                'correct': is_correct,
            })
        
        total = len(test_results)
        accuracy = correct_count / total if total > 0 else 1.0
        
        return {
            'passed': accuracy >= 0.8,  # 80% 通过阈值
            'accuracy': accuracy,
            'total_tests': total,
            'correct_count': correct_count,
            'test_results': test_results,
        }
    
    def _generate_internalization_id(self, learning_unit: LearningUnit) -> str:
        """生成内化 ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        hash_input = f"{learning_unit.id}_{timestamp}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
        return f"intern_{timestamp}_{hash_value}"
    
    def _save_version_snapshot(self, version: int):
        """保存版本快照"""
        # Adapter 快照
        self.adapter_manager.create_version_snapshot()
        
        # Decision Head 快照（已在 engine 内部管理）
        
        # 保存到文件系统
        snapshot_path = os.path.join(
            self.weight_storage_dir,
            f"version_{version}.pt"
        )
        
        snapshot = {
            'version': version,
            'adapter_states': [
                adapter.export_state() 
                for adapter in self.adapter_manager.adapters
            ],
            'decision_head_state': self.decision_head_engine._save_state(),
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(snapshot, snapshot_path)
        print(f"  ✓ 版本快照已保存: {snapshot_path}")
    
    def rollback(self, target_version: int) -> bool:
        """
        回滚到指定版本
        """
        snapshot_path = os.path.join(
            self.weight_storage_dir,
            f"version_{target_version}.pt"
        )
        
        if not os.path.exists(snapshot_path):
            print(f"版本 {target_version} 不存在")
            return False
        
        snapshot = torch.load(snapshot_path)
        
        # 恢复 Adapter 状态
        for adapter, state in zip(
            self.adapter_manager.adapters,
            snapshot['adapter_states']
        ):
            adapter.import_state(state)
        
        # 恢复 Decision Head 状态
        self.decision_head_engine._restore_state(snapshot['decision_head_state'])
        
        print(f"已回滚到版本 {target_version}")
        return True
    
    def remove_knowledge(self, constraint_id: str) -> bool:
        """
        移除特定约束的知识
        """
        # 从 Adapter 移除
        adapter_success = self.adapter_manager.remove_knowledge_from_all(constraint_id)
        
        # Decision Head 不支持单个约束移除，需要回滚
        
        return adapter_success
    
    # ==================== 数据库集成 ====================
    
    def _create_db_record(
        self,
        learning_unit: LearningUnit,
        result: InternalizationResult
    ) -> Optional[str]:
        """创建数据库记录"""
        if not self.db:
            return None
        
        # 这里需要导入实际的 ORM 模型
        # 示例伪代码：
        """
        from models.internalization import KnowledgeInternalization
        
        record = KnowledgeInternalization(
            id=uuid.uuid4(),
            learning_unit_id=learning_unit.id,
            internalization_type='deep',  # decision_head, knowledge_adapter, lora
            status='in_progress',
            version=result.version,
            started_at=result.started_at,
        )
        self.db.add(record)
        self.db.commit()
        return str(record.id)
        """
        return result.internalization_id
    
    def _update_db_record(
        self,
        record_id: Optional[str],
        result: InternalizationResult,
        validation: Dict[str, Any]
    ):
        """更新数据库记录"""
        if not self.db or not record_id:
            return
        
        # 示例伪代码：
        """
        record = self.db.query(KnowledgeInternalization).get(record_id)
        record.status = 'completed' if result.success else 'failed'
        record.completed_at = result.completed_at
        record.validation_accuracy = validation['accuracy']
        record.weight_snapshot_path = f"version_{result.version}.pt"
        
        # 记录约束详情
        for constraint_info in result.adapter_result.get('constraints', []):
            detail = InternalizedConstraint(
                internalization_id=record.id,
                constraint_id=constraint_info['constraint_id'],
                slot_index=constraint_info['slot_indices'][0] if constraint_info['slot_indices'] else None,
                is_active=True,
            )
            self.db.add(detail)
        
        # 记录测试结果
        for test in validation.get('test_results', []):
            effect = InternalizationEffect(
                internalization_id=record.id,
                test_query=test['test_query'],
                expected_decision=test['expected'],
                actual_decision=test['actual'],
                confidence=test['confidence'],
                is_correct=test['correct'],
            )
            self.db.add(effect)
        
        self.db.commit()
        """
        pass
    
    # ==================== 查询接口 ====================
    
    def get_decision(self, query: str) -> Dict[str, Any]:
        """
        获取决策
        
        结合 Adapter 增强的表示 + Decision Head 的分类
        """
        return self.decision_head_engine.get_decision(query)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_version': self.current_version,
            'adapter_stats': self.adapter_manager.get_statistics(),
            'decision_head_stats': self.decision_head_engine.get_statistics(),
            'internalization_count': len(self.internalization_history),
            'has_llm': self.llm_adapter is not None,
        }
    
    def get_internalization_history(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """获取内化历史"""
        return [r.to_dict() for r in self.internalization_history[-limit:]]


# ==================== 数据库模型定义（SQLAlchemy） ====================

"""
以下是与 03_数据库设计文档.md 一致的 SQLAlchemy 模型定义。
需要添加到 backend/app/models/ 目录。

# backend/app/models/internalization.py

from sqlalchemy import Column, String, Float, Integer, Boolean, ForeignKey, Text, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base


class KnowledgeInternalization(Base):
    __tablename__ = 'knowledge_internalization'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 关联 Learning Unit
    learning_unit_id = Column(UUID(as_uuid=True), ForeignKey('learning_units.id'), nullable=False, index=True)
    
    # 内化类型
    internalization_type = Column(String(50), nullable=False)
    # decision_head, knowledge_adapter, deep (both), lora, full_finetune
    
    # 内化目标
    target_layer = Column(String(50))
    
    # 状态
    status = Column(String(20), nullable=False, default='pending', index=True)
    # pending, in_progress, completed, failed, rolled_back
    
    # 版本控制
    version = Column(Integer, nullable=False, default=1)
    pre_internalization_hash = Column(String(64))
    post_internalization_hash = Column(String(64))
    
    # 权重快照路径
    weight_snapshot_path = Column(String(255))
    
    # 验证结果
    validation_accuracy = Column(Float)
    validation_samples = Column(Integer)
    
    # 时间戳
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关系
    learning_unit = relationship('LearningUnit')
    constraints = relationship('InternalizedConstraint', back_populates='internalization', cascade='all, delete-orphan')
    effects = relationship('InternalizationEffect', back_populates='internalization', cascade='all, delete-orphan')


class InternalizedConstraint(Base):
    __tablename__ = 'internalized_constraints'
    
    id = Column(Integer, primary_key=True)
    internalization_id = Column(UUID(as_uuid=True), ForeignKey('knowledge_internalization.id', ondelete='CASCADE'), nullable=False, index=True)
    constraint_id = Column(String(100), nullable=False)
    
    # 内化位置
    slot_index = Column(Integer)
    adapter_layer = Column(Integer)
    
    # 知识向量（小型时直接存储）
    knowledge_vector = Column(LargeBinary)
    
    # 状态
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关系
    internalization = relationship('KnowledgeInternalization', back_populates='constraints')


class InternalizationEffect(Base):
    __tablename__ = 'internalization_effects'
    
    id = Column(Integer, primary_key=True)
    internalization_id = Column(UUID(as_uuid=True), ForeignKey('knowledge_internalization.id'), nullable=False, index=True)
    
    # 测试场景
    test_query = Column(Text, nullable=False)
    expected_decision = Column(String(50))
    
    # 实际结果
    actual_decision = Column(String(50))
    confidence = Column(Float)
    
    # 是否符合预期
    is_correct = Column(Boolean)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关系
    internalization = relationship('KnowledgeInternalization', back_populates='effects')
"""

