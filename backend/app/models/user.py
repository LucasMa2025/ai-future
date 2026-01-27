"""
用户与角色模型
"""
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid

from .base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True, comment="用户名")
    email = Column(String(255), unique=True, nullable=False, index=True, comment="邮箱")
    hashed_password = Column(String(255), nullable=False, comment="密码哈希")
    full_name = Column(String(100), comment="全名")
    phone = Column(String(20), comment="电话")
    avatar = Column(String(500), comment="头像URL")
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_superuser = Column(Boolean, default=False, comment="是否超级管理员")
    last_login = Column(DateTime(timezone=True), comment="最后登录时间")
    login_count = Column(Integer, default=0, comment="登录次数")
    
    # 通知偏好
    notification_preferences = Column(JSONB, default={}, comment="通知偏好设置")
    
    # 关系
    roles = relationship("Role", secondary="user_roles", back_populates="users")
    notifications = relationship("Notification", back_populates="user")
    operation_logs = relationship("OperationLog", back_populates="user")
    
    def __repr__(self):
        return f"<User {self.username}>"


class Role(Base, TimestampMixin):
    """角色表"""
    __tablename__ = "roles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False, index=True, comment="角色标识")
    display_name = Column(String(100), nullable=False, comment="显示名称")
    description = Column(Text, comment="描述")
    risk_level_limit = Column(String(20), comment="可审批的最高风险等级")
    is_system = Column(Boolean, default=False, comment="是否系统内置角色")
    sort_order = Column(Integer, default=0, comment="排序")
    
    # 关系
    users = relationship("User", secondary="user_roles", back_populates="roles")
    permissions = relationship("Permission", secondary="role_permissions", back_populates="roles")
    
    def __repr__(self):
        return f"<Role {self.name}>"


class UserRole(Base):
    """用户角色关联表"""
    __tablename__ = "user_roles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    role_id = Column(
        Integer, 
        ForeignKey("roles.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    assigned_at = Column(DateTime(timezone=True), server_default="now()", comment="分配时间")
    assigned_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), comment="分配人")
    
    # 唯一约束
    __table_args__ = (
        {"comment": "用户角色关联表"},
    )

