"""
通知模型
"""
from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB

from .base import Base, TimestampMixin


class Notification(Base, TimestampMixin):
    """
    通知表
    
    存储发送给用户的通知
    """
    __tablename__ = "notifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    
    # 接收者
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True,
        comment="接收用户ID"
    )
    
    # 通知类型
    type = Column(String(50), nullable=False, index=True, comment="通知类型")
    category = Column(String(50), default="system", comment="通知分类: system/approval/anomaly/state")
    
    # 通知渠道
    channel = Column(String(20), default="websocket", comment="发送渠道: websocket/email/both")
    
    # 内容
    title = Column(String(200), nullable=False, comment="标题")
    message = Column(Text, comment="消息内容")
    metadata = Column(JSONB, default={}, comment="元数据")
    
    # 关联
    related_type = Column(String(50), comment="关联对象类型")
    related_id = Column(String(100), comment="关联对象ID")
    
    # 状态
    is_read = Column(Boolean, default=False, index=True, comment="是否已读")
    read_at = Column(DateTime(timezone=True), comment="已读时间")
    
    # 发送状态
    ws_sent = Column(Boolean, default=False, comment="WebSocket是否已发送")
    ws_sent_at = Column(DateTime(timezone=True), comment="WebSocket发送时间")
    email_sent = Column(Boolean, default=False, comment="邮件是否已发送")
    email_sent_at = Column(DateTime(timezone=True), comment="邮件发送时间")
    email_error = Column(Text, comment="邮件发送错误")
    
    # 优先级
    priority = Column(String(20), default="normal", comment="优先级: low/normal/high/urgent")
    
    # 过期
    expires_at = Column(DateTime(timezone=True), comment="过期时间")
    
    # 关系
    user = relationship("User", back_populates="notifications")
    
    __table_args__ = (
        {"comment": "通知表"},
    )
    
    def __repr__(self):
        return f"<Notification {self.type} to {self.user_id}>"


class UserNotificationSetting(Base, TimestampMixin):
    """
    用户通知设置表
    
    用户的通知偏好配置
    """
    __tablename__ = "user_notification_settings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="用户ID"
    )
    
    # 通知类型
    notification_type = Column(String(50), nullable=False, comment="通知类型")
    
    # 渠道开关
    websocket_enabled = Column(Boolean, default=True, comment="WebSocket通知开关")
    email_enabled = Column(Boolean, default=False, comment="邮件通知开关")
    
    # 免打扰
    muted = Column(Boolean, default=False, comment="是否静音")
    muted_until = Column(DateTime(timezone=True), comment="静音到期时间")
    
    __table_args__ = (
        {"comment": "用户通知设置表"},
    )

