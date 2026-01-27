"""
异常检测相关数据库模型
"""
from sqlalchemy import Column, String, Float, Integer, Boolean, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base, TimestampMixin


class AnomalyEvent(Base, TimestampMixin):
    """
    异常事件表
    
    存储检测到的异常事件及其处理状态
    """
    __tablename__ = 'anomaly_events'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # 严重性
    severity = Column(String(20), nullable=False, index=True)  # low, medium, high, critical
    composite_score = Column(Float)
    
    # 检测结果
    detected_by = Column(JSONB, default=[])  # 触发的检测器列表
    recommendation = Column(Text)
    
    # 响应决策
    response_decision = Column(String(20))  # log, alert, diagnose, rollback, halt
    
    # 处理状态
    status = Column(String(20), default='open', index=True)  # open, investigating, resolved, ignored
    resolved_by = Column(UUID(as_uuid=True), ForeignKey('users.id'))
    resolved_at = Column(String)
    resolution_notes = Column(Text)
    
    # 关联
    learning_unit_id = Column(UUID(as_uuid=True), ForeignKey('learning_units.id'))
    artifact_id = Column(UUID(as_uuid=True), ForeignKey('artifacts.id'))
    
    # 时间戳
    detected_at = Column(String, default=lambda: datetime.utcnow().isoformat())
    
    # 关系
    signals = relationship('AnomalySignalRecord', back_populates='anomaly_event', cascade='all, delete-orphan')


class AnomalySignalRecord(Base, TimestampMixin):
    """
    异常信号详情表
    
    存储各检测器产生的具体信号
    """
    __tablename__ = 'anomaly_signals'
    
    id = Column(Integer, primary_key=True)
    anomaly_event_id = Column(UUID(as_uuid=True), ForeignKey('anomaly_events.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # 检测器信息
    detector_type = Column(String(50), nullable=False, index=True)  # metric, behavior, drift, external
    detected = Column(Boolean, default=True)
    severity = Column(String(20))
    
    # 详细信号数据
    signal_data = Column(JSONB, nullable=False)
    
    # 关系
    anomaly_event = relationship('AnomalyEvent', back_populates='signals')

