"""
审计与日志模型
"""
from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from .base import Base


class BusinessAuditLog(Base):
    """
    业务审计日志表
    
    记录重要业务操作的审计信息（非学习单元审计）
    使用链式哈希保证不可篡改性
    """
    __tablename__ = "business_audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entry_id = Column(String(100), unique=True, nullable=False, index=True, comment="条目ID")
    
    # 事件信息
    event_type = Column(String(100), nullable=False, index=True, comment="事件类型")
    event_category = Column(String(50), index=True, comment="事件分类: state/approval/artifact/user/system")
    
    # 操作者
    actor_id = Column(UUID(as_uuid=True), index=True, comment="操作者ID")
    actor_name = Column(String(100), comment="操作者名称")
    actor_ip = Column(String(50), comment="操作者IP")
    
    # 操作内容
    action = Column(String(100), nullable=False, index=True, comment="操作动作")
    
    # 目标
    target_type = Column(String(50), index=True, comment="目标类型")
    target_id = Column(String(100), index=True, comment="目标ID")
    target_name = Column(String(200), comment="目标名称")
    
    # 详情
    details = Column(JSONB, default={}, comment="详细信息")
    request_data = Column(JSONB, comment="请求数据")
    response_data = Column(JSONB, comment="响应数据")
    
    # 结果
    result = Column(String(20), default="success", comment="结果: success/failure")
    error_message = Column(Text, comment="错误信息")
    
    # 审计级别
    audit_level = Column(String(20), default="normal", index=True, comment="审计级别")
    
    # 链式哈希
    previous_hash = Column(String(64), nullable=False, comment="上一条哈希")
    entry_hash = Column(String(64), nullable=False, comment="本条哈希")
    
    # 时间戳
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="创建时间"
    )
    
    __table_args__ = (
        {"comment": "业务审计日志表"},
    )
    
    def __repr__(self):
        return f"<BusinessAuditLog {self.event_type}:{self.action}>"


class OperationLog(Base):
    """
    用户操作日志表
    
    记录用户的所有操作行为，用于行为分析和问题追溯
    """
    __tablename__ = "operation_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 用户信息
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), index=True, comment="用户ID")
    username = Column(String(50), comment="用户名")
    
    # 请求信息
    request_id = Column(String(100), index=True, comment="请求ID")
    method = Column(String(10), comment="HTTP方法")
    path = Column(String(500), comment="请求路径")
    query_params = Column(JSONB, comment="查询参数")
    request_body = Column(JSONB, comment="请求体(脱敏)")
    
    # 客户端信息
    ip_address = Column(String(50), comment="IP地址")
    user_agent = Column(String(500), comment="用户代理")
    
    # 响应信息
    status_code = Column(Integer, comment="响应状态码")
    response_time_ms = Column(Integer, comment="响应时间(毫秒)")
    
    # 关联功能
    function_code = Column(String(100), index=True, comment="功能代码")
    
    # 结果
    is_success = Column(Integer, default=1, comment="是否成功: 1=成功, 0=失败")
    error_message = Column(Text, comment="错误信息")
    
    # 时间戳
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        comment="创建时间"
    )
    
    # 关系
    user = relationship("User", back_populates="operation_logs")
    
    __table_args__ = (
        {"comment": "用户操作日志表"},
    )
    
    def __repr__(self):
        return f"<OperationLog {self.method} {self.path}>"

