"""
Token 管理模型
"""
from sqlalchemy import Column, String, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


class TokenBlacklist(Base):
    """
    Token 黑名单表
    
    用于存储已注销的 Token，配合 Redis 实现双重验证
    PostgreSQL 作为持久化存储，Redis 作为快速缓存
    """
    __tablename__ = "token_blacklist"
    
    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    
    # Token 信息
    jti = Column(String(100), unique=True, nullable=False, index=True, comment="JWT ID")
    token_type = Column(String(20), default="access", comment="Token类型: access/refresh")
    
    # 用户信息
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True, comment="用户ID")
    
    # 原因
    reason = Column(String(100), comment="加入黑名单原因: logout/revoked/password_changed")
    
    # 时间
    expires_at = Column(DateTime(timezone=True), nullable=False, comment="Token原过期时间")
    blacklisted_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="加入黑名单时间"
    )
    
    __table_args__ = (
        {"comment": "Token黑名单表"},
    )
    
    def __repr__(self):
        return f"<TokenBlacklist {self.jti}>"

