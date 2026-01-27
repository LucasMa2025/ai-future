"""
事务相关数据库模型
"""
from sqlalchemy import Column, String, Boolean, Integer, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base, TimestampMixin


class Transaction(Base, TimestampMixin):
    """
    事务记录表
    
    记录所有事务操作，支持回滚和审计
    """
    __tablename__ = 'transactions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 事务类型
    tx_type = Column(String(50), nullable=False, index=True)  # rollback, release, update
    
    # 目标工件
    target_artifact_id = Column(UUID(as_uuid=True), ForeignKey('artifacts.id'))
    
    # 事务状态
    status = Column(String(20), nullable=False, default='pending', index=True)
    # pending, in_progress, committed, aborted
    
    # 描述
    description = Column(Text)
    
    # 事务前状态
    pre_state = Column(JSONB)
    
    # 事务后状态
    post_state = Column(JSONB)
    
    # 执行信息
    started_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    completed_at = Column(String)
    executed_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    
    # 结果
    success = Column(Boolean)
    error_message = Column(Text)
    rollback_reason = Column(Text)
    
    # 关系
    target_artifact = relationship('Artifact')
    executor = relationship('User')

