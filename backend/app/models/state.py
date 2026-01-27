"""
NLGSM 状态机模型
"""
from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID, JSONB

from .base import Base, TimestampMixin


class SystemState(Base, TimestampMixin):
    """
    系统状态表
    
    记录 NLGSM 状态机的状态历史
    """
    __tablename__ = "system_states"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 状态信息
    state = Column(String(50), nullable=False, index=True, comment="状态值")
    entered_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        comment="进入时间"
    )
    
    # 触发信息
    trigger_event = Column(String(50), comment="触发事件")
    trigger_source = Column(String(100), comment="触发来源")
    
    # 元数据
    metadata = Column(JSONB, default={}, comment="元数据")
    
    # 状态标记
    is_current = Column(Boolean, default=False, index=True, comment="是否当前状态")
    
    # 统计
    iteration_count = Column(Integer, default=0, comment="迭代计数")
    duration_seconds = Column(Integer, comment="持续时间(秒)")
    
    __table_args__ = (
        {"comment": "系统状态表"},
    )
    
    def __repr__(self):
        return f"<SystemState {self.state} current={self.is_current}>"


class StateTransition(Base):
    """
    状态转换记录表
    
    记录每次状态转换的详细信息
    """
    __tablename__ = "state_transitions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 转换信息
    from_state = Column(String(50), nullable=False, index=True, comment="源状态")
    to_state = Column(String(50), nullable=False, index=True, comment="目标状态")
    trigger_event = Column(String(50), nullable=False, index=True, comment="触发事件")
    trigger_source = Column(String(100), comment="触发来源")
    
    # 决策信息
    decision = Column(String(50), nullable=False, comment="决策结果")
    decision_reason = Column(Text, comment="决策原因")
    decision_evidence = Column(JSONB, comment="决策证据")
    triggered_rules = Column(JSONB, default=[], comment="触发的规则")
    
    # 执行信息
    actions_executed = Column(JSONB, default=[], comment="执行的动作")
    
    # 结果
    success = Column(Boolean, default=True, comment="是否成功")
    error_message = Column(Text, comment="错误信息")
    
    # 性能
    duration_ms = Column(Integer, comment="耗时(毫秒)")
    
    # 时间戳
    created_at = Column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False,
        index=True,
        comment="创建时间"
    )
    
    __table_args__ = (
        {"comment": "状态转换记录表"},
    )
    
    def __repr__(self):
        return f"<StateTransition {self.from_state}->{self.to_state}>"

