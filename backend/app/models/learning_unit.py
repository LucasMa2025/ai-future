"""
Learning Unit 模型

Learning Unit 是 NLGSM 治理系统的核心数据结构，
封装了自学习系统产生的知识、约束和审计状态。

v2.0 新增：
- 链式学习支持（parent_lu_id, chain_depth, chain_root_id）
- 知识内容字段（knowledge, provenance）
- 继续学习上下文（continue_from_context）
"""
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, DateTime, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
import uuid
from datetime import datetime

from .base import Base, TimestampMixin


class LearningUnit(Base, TimestampMixin):
    """
    学习单元表
    
    存储自学习系统产生的知识单元，包括：
    - 学习来源和上下文
    - 提议的约束
    - 审计状态和历史
    - AGA 内化状态
    - 链式学习信息（v2.0）
    """
    __tablename__ = "learning_units"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 基本信息
    title = Column(String(255), nullable=False, comment="标题")
    description = Column(Text, comment="描述")
    
    # 学习来源
    learning_session_id = Column(String(100), index=True, comment="学习会话ID")
    checkpoint_id = Column(String(100), comment="检查点ID")
    scope_id = Column(String(100), comment="学习范围ID")
    
    # NL 层级信息
    nl_level = Column(String(50), comment="NL层级: parameter/memory/optimizer/policy")
    learning_goal = Column(Text, comment="学习目标")
    
    # 知识内容（v2.0）
    knowledge = Column(JSONB, comment="知识内容 JSON")
    provenance = Column(JSONB, comment="来源追溯 JSON")
    
    # 链式学习信息（v2.0）
    parent_lu_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("learning_units.id"),
        index=True,
        comment="父 Learning Unit ID（链式学习）"
    )
    chain_depth = Column(Integer, default=0, comment="链式学习深度（0 表示根节点）")
    chain_root_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id"),
        index=True,
        comment="链式学习根节点 ID"
    )
    continue_from_context = Column(JSONB, comment="继续学习的上下文信息")
    
    # 状态
    status = Column(
        String(50), 
        default="pending",
        index=True,
        comment="状态: pending/auto_classified/human_review/approved/corrected/rejected/terminated"
    )
    
    # 风险评估（由审计系统定义）
    risk_level = Column(String(20), comment="风险等级: low/medium/high/critical")
    risk_score = Column(Float, comment="风险评分 0-100")
    risk_factors = Column(JSONB, default=[], comment="风险因素列表")
    
    # 审计信息
    auto_classification = Column(JSONB, comment="自动分类结果")
    requires_human_review = Column(Boolean, default=True, comment="是否需要人工审核")
    review_priority = Column(Integer, default=5, comment="审核优先级 1-10")
    
    # 审批信息
    approval_id = Column(String(100), comment="关联的审批ID")
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), comment="审批人")
    approved_at = Column(DateTime(timezone=True), comment="审批时间")
    approval_comments = Column(Text, comment="审批意见")
    
    # 修正信息（如果被修正）
    is_corrected = Column(Boolean, default=False, comment="是否被修正")
    correction_summary = Column(Text, comment="修正摘要")
    original_constraints = Column(JSONB, comment="原始约束（修正前）")
    
    # AGA 内化状态
    is_internalized = Column(Boolean, default=False, comment="是否已内化到AGA")
    internalized_at = Column(DateTime(timezone=True), comment="内化时间")
    aga_slot_mapping = Column(JSONB, comment="AGA槽位映射 {layer_idx: slot_idx}")
    lifecycle_state = Column(
        String(50), 
        default="quarantined",
        comment="生命周期状态: quarantined/probationary/confirmed/deprecated"
    )
    
    # 元数据
    metadata = Column(JSONB, default={}, comment="附加元数据")
    tags = Column(ARRAY(String), default=[], comment="标签")
    
    # 关系
    constraints = relationship("LUConstraint", back_populates="learning_unit", cascade="all, delete-orphan")
    audit_history = relationship("LUAuditHistory", back_populates="learning_unit", cascade="all, delete-orphan")
    approver = relationship("User", foreign_keys=[approved_by])
    
    # 链式学习关系（v2.0）
    parent_lu = relationship(
        "LearningUnit",
        remote_side=[id],
        foreign_keys=[parent_lu_id],
        backref="children_lus"
    )
    chain_root = relationship(
        "LearningUnit",
        remote_side=[id],
        foreign_keys=[chain_root_id],
    )
    
    def __repr__(self):
        return f"<LearningUnit {self.id} status={self.status}>"
    
    def get_chain_path(self) -> list:
        """获取从根到当前节点的链路径"""
        path = [self]
        current = self
        while current.parent_lu:
            path.insert(0, current.parent_lu)
            current = current.parent_lu
        return path
    
    def get_chain_depth_actual(self) -> int:
        """计算实际链深度"""
        depth = 0
        current = self
        while current.parent_lu:
            depth += 1
            current = current.parent_lu
        return depth


class LUConstraint(Base, TimestampMixin):
    """
    学习单元约束表
    
    存储 Learning Unit 中的具体约束条目
    """
    __tablename__ = "lu_constraints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    learning_unit_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("learning_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 约束内容
    condition = Column(Text, nullable=False, comment="条件（何时激活）")
    decision = Column(Text, nullable=False, comment="决策（修正信号）")
    confidence = Column(Float, default=0.5, comment="置信度 0-1")
    
    # 来源
    source_type = Column(String(50), comment="来源类型: llm/rule/human")
    source_evidence = Column(JSONB, comment="来源证据")
    
    # 审计状态
    is_approved = Column(Boolean, default=False, comment="是否审批通过")
    is_modified = Column(Boolean, default=False, comment="是否被修改")
    original_condition = Column(Text, comment="原始条件（修改前）")
    original_decision = Column(Text, comment="原始决策（修改前）")
    
    # AGA 内化
    aga_key_vector = Column(JSONB, comment="AGA Key向量（序列化）")
    aga_value_vector = Column(JSONB, comment="AGA Value向量（序列化）")
    
    # 关系
    learning_unit = relationship("LearningUnit", back_populates="constraints")
    
    def __repr__(self):
        return f"<LUConstraint {self.id}>"


class LUAuditHistory(Base):
    """
    学习单元审计历史表
    
    记录 Learning Unit 的所有审计操作
    """
    __tablename__ = "lu_audit_history"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    learning_unit_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 操作信息
    action = Column(String(50), nullable=False, comment="操作: create/review/approve/reject/correct/internalize/quarantine")
    actor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), comment="操作人")
    actor_name = Column(String(100), comment="操作人名称")
    
    # 状态变更
    from_status = Column(String(50), comment="变更前状态")
    to_status = Column(String(50), comment="变更后状态")
    
    # 详情
    comments = Column(Text, comment="备注")
    details = Column(JSONB, comment="详细信息")
    
    # 时间
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    
    # 关系
    learning_unit = relationship("LearningUnit", back_populates="audit_history")
    actor = relationship("User")
    
    def __repr__(self):
        return f"<LUAuditHistory {self.id} action={self.action}>"


class LearningSession(Base, TimestampMixin):
    """
    学习会话表
    
    记录自学习系统的学习会话
    """
    __tablename__ = "learning_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String(100), unique=True, nullable=False, index=True, comment="会话标识")
    
    # 会话配置
    scope_id = Column(String(100), comment="学习范围ID")
    goal = Column(Text, comment="学习目标")
    max_nl_level = Column(String(50), comment="最大NL层级")
    allowed_levels = Column(ARRAY(String), comment="允许的NL层级")
    
    # 状态
    status = Column(
        String(50),
        default="active",
        comment="状态: active/paused/completed/terminated"
    )
    
    # 统计
    total_checkpoints = Column(Integer, default=0, comment="检查点总数")
    total_learning_units = Column(Integer, default=0, comment="产生的LU总数")
    approved_units = Column(Integer, default=0, comment="审批通过的LU数")
    rejected_units = Column(Integer, default=0, comment="拒绝的LU数")
    
    # 干预记录
    interventions = Column(JSONB, default=[], comment="干预记录")
    
    # 时间
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    paused_at = Column(DateTime(timezone=True), comment="暂停时间")
    completed_at = Column(DateTime(timezone=True), comment="完成时间")
    
    # 元数据
    metadata = Column(JSONB, default={})
    
    def __repr__(self):
        return f"<LearningSession {self.session_id} status={self.status}>"


class Checkpoint(Base, TimestampMixin):
    """
    检查点表
    
    记录学习过程中的检查点
    """
    __tablename__ = "checkpoints"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    checkpoint_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # 关联
    session_id = Column(String(100), ForeignKey("learning_sessions.session_id"), index=True)
    
    # 状态快照
    nl_level = Column(String(50), comment="当前NL层级")
    knowledge_count = Column(Integer, default=0, comment="知识数量")
    state_snapshot = Column(JSONB, comment="状态快照")
    
    # 审查状态
    review_status = Column(
        String(50),
        default="pending",
        comment="审查状态: pending/reviewed/approved/rejected"
    )
    reviewed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    reviewed_at = Column(DateTime(timezone=True))
    review_decision = Column(String(50), comment="审查决策: continue/modify/pause/terminate")
    review_comments = Column(Text)
    
    # 修改的范围（如果有）
    modified_scope = Column(JSONB, comment="修改后的学习范围")
    modified_goal = Column(Text, comment="修改后的学习目标")
    
    def __repr__(self):
        return f"<Checkpoint {self.checkpoint_id}>"


# ==================== v2.0 新增模型 ====================


class LURelation(Base, TimestampMixin):
    """
    Learning Unit 关系表
    
    记录 LU 之间的关系，用于知识图谱追踪
    """
    __tablename__ = "lu_relations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 关系双方
    source_lu_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="源 LU ID"
    )
    target_lu_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="目标 LU ID"
    )
    
    # 关系类型
    relation_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="关系类型: continues/refines/contradicts/supports/depends_on/supersedes"
    )
    
    # 关系强度
    strength = Column(Float, default=1.0, comment="关系强度 0-1")
    
    # 关系描述
    description = Column(Text, comment="关系描述")
    
    # 元数据
    metadata = Column(JSONB, default={}, comment="附加元数据")
    
    # 创建信息
    created_by = Column(String(100), comment="创建者")
    
    # 关系
    source_lu = relationship(
        "LearningUnit",
        foreign_keys=[source_lu_id],
        backref="outgoing_relations"
    )
    target_lu = relationship(
        "LearningUnit",
        foreign_keys=[target_lu_id],
        backref="incoming_relations"
    )
    
    def __repr__(self):
        return f"<LURelation {self.source_lu_id} -{self.relation_type}-> {self.target_lu_id}>"


class LearningChain(Base, TimestampMixin):
    """
    学习链表
    
    追踪完整的学习链路
    """
    __tablename__ = "learning_chains"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 链标识
    chain_id = Column(String(100), unique=True, nullable=False, index=True, comment="链标识")
    
    # 根节点
    root_lu_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id"),
        nullable=False,
        index=True,
        comment="根 LU ID"
    )
    
    # 当前头节点
    head_lu_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id"),
        nullable=False,
        index=True,
        comment="头 LU ID"
    )
    
    # 链信息
    total_depth = Column(Integer, default=1, comment="总深度")
    total_units = Column(Integer, default=1, comment="总单元数")
    
    # 学习目标
    initial_goal = Column(Text, comment="初始学习目标")
    current_goal = Column(Text, comment="当前学习目标")
    
    # 状态
    status = Column(
        String(50),
        default="active",
        comment="状态: active/completed/abandoned"
    )
    
    # 统计
    approved_count = Column(Integer, default=0, comment="审批通过数")
    rejected_count = Column(Integer, default=0, comment="拒绝数")
    
    # 完成时间
    completed_at = Column(DateTime(timezone=True), comment="完成时间")
    
    # 关系
    root_lu = relationship("LearningUnit", foreign_keys=[root_lu_id])
    head_lu = relationship("LearningUnit", foreign_keys=[head_lu_id])
    
    def __repr__(self):
        return f"<LearningChain {self.chain_id} depth={self.total_depth}>"


class ProductionKnowledge(Base, TimestampMixin):
    """
    生产知识表
    
    存储已内化到生产系统的知识，供自学习系统查询
    """
    __tablename__ = "production_knowledge"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 来源 Learning Unit
    source_lu_id = Column(
        UUID(as_uuid=True),
        ForeignKey("learning_units.id"),
        nullable=False,
        index=True,
        comment="来源 LU ID"
    )
    
    # 知识内容
    domain = Column(String(100), nullable=False, index=True, comment="领域")
    knowledge_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="知识类型: decision_rule/knowledge/pattern/constraint"
    )
    
    # 条件和决策
    condition = Column(Text, nullable=False, comment="条件")
    decision = Column(Text, nullable=False, comment="决策")
    
    # 向量表示（用于相似度搜索）
    embedding_vector = Column(JSONB, comment="嵌入向量")
    embedding_model = Column(String(100), comment="嵌入模型")
    
    # 置信度和风险
    confidence = Column(Float, default=0.5, comment="置信度")
    risk_level = Column(String(20), comment="风险等级")
    
    # 状态
    status = Column(
        String(50),
        default="active",
        index=True,
        comment="状态: active/deprecated/superseded"
    )
    
    # 使用统计
    hit_count = Column(Integer, default=0, comment="命中次数")
    last_hit_at = Column(DateTime(timezone=True), comment="最后命中时间")
    
    # AGA 映射
    aga_slot_idx = Column(Integer, comment="AGA 槽位索引")
    aga_layer_idx = Column(Integer, comment="AGA 层索引")
    
    # 废弃时间
    deprecated_at = Column(DateTime(timezone=True), comment="废弃时间")
    
    # 关系
    source_lu = relationship("LearningUnit")
    
    def __repr__(self):
        return f"<ProductionKnowledge {self.id} domain={self.domain}>"

