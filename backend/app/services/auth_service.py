"""
认证服务

实现:
1. 用户登录/登出
2. Token 管理
3. 密码验证
"""
from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from redis.asyncio import Redis

from ..config import settings
from ..models.user import User
from ..models.token import TokenBlacklist
from ..core.security import verify_password
from ..core.exceptions import (
    InvalidCredentialsError,
    UserInactiveError,
    TokenExpiredError,
    TokenInvalidError,
)
from ..middleware.auth import JWTAuthMiddleware


class AuthService:
    """
    认证服务
    
    处理用户认证相关的业务逻辑
    """
    
    def __init__(self, db: Session, redis: Optional[Redis] = None):
        self.db = db
        self.redis = redis
    
    async def login(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> dict:
        """
        用户登录
        
        Args:
            username: 用户名或邮箱
            password: 密码
            ip_address: 客户端IP
            
        Returns:
            包含 token 信息的字典
        """
        # 查找用户
        user = self.db.query(User).filter(
            (User.username == username) | (User.email == username)
        ).first()
        
        if not user:
            raise InvalidCredentialsError()
        
        # 验证密码
        if not verify_password(password, user.hashed_password):
            raise InvalidCredentialsError()
        
        # 检查激活状态
        if not user.is_active:
            raise UserInactiveError()
        
        # 生成 Token
        roles = [role.name for role in user.roles]
        permissions = []
        for role in user.roles:
            for perm in role.permissions:
                if perm.code not in permissions:
                    permissions.append(perm.code)
        
        access_token, access_jti = JWTAuthMiddleware.create_access_token(
            user_id=str(user.id),
            roles=roles,
            permissions=permissions,
        )
        
        refresh_token, refresh_jti = JWTAuthMiddleware.create_refresh_token(
            user_id=str(user.id),
        )
        
        # 缓存 Token
        if self.redis:
            await JWTAuthMiddleware.cache_token(
                redis=self.redis,
                jti=access_jti,
                user_id=str(user.id),
                token_type="access",
                expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
            
            await JWTAuthMiddleware.cache_token(
                redis=self.redis,
                jti=refresh_jti,
                user_id=str(user.id),
                token_type="refresh",
                expires_in=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
            )
        
        # 更新登录信息
        user.last_login = datetime.utcnow()
        user.login_count = (user.login_count or 0) + 1
        self.db.commit()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "is_superuser": user.is_superuser,
                "roles": roles,
            }
        }
    
    async def logout(
        self,
        jti: str,
        user_id: UUID,
        expires_at: datetime
    ) -> bool:
        """
        用户登出
        
        将当前 Token 加入黑名单
        """
        if self.redis:
            await JWTAuthMiddleware.blacklist_token(
                redis=self.redis,
                db=self.db,
                jti=jti,
                user_id=user_id,
                expires_at=expires_at,
                reason="logout"
            )
        else:
            # 仅写入数据库
            blacklist_entry = TokenBlacklist(
                jti=jti,
                user_id=user_id,
                token_type="access",
                reason="logout",
                expires_at=expires_at,
            )
            self.db.add(blacklist_entry)
            self.db.commit()
        
        return True
    
    async def refresh_token(self, refresh_token: str) -> dict:
        """
        刷新访问 Token
        
        Args:
            refresh_token: 刷新令牌
            
        Returns:
            新的 token 信息
        """
        try:
            # 解码刷新令牌
            payload = JWTAuthMiddleware.decode_token(refresh_token)
            
            if payload.type != "refresh":
                raise TokenInvalidError("Not a refresh token")
            
            # 检查黑名单
            if self.redis:
                is_blacklisted = await JWTAuthMiddleware.is_token_blacklisted(
                    redis=self.redis,
                    db=self.db,
                    jti=payload.jti
                )
                if is_blacklisted:
                    raise TokenInvalidError("Token has been revoked")
            
            # 获取用户
            user = self.db.query(User).filter(User.id == payload.sub).first()
            if not user:
                raise TokenInvalidError("User not found")
            
            if not user.is_active:
                raise UserInactiveError()
            
            # 将旧的刷新令牌加入黑名单
            if self.redis:
                await JWTAuthMiddleware.blacklist_token(
                    redis=self.redis,
                    db=self.db,
                    jti=payload.jti,
                    user_id=user.id,
                    expires_at=payload.exp,
                    reason="refresh"
                )
            
            # 生成新的 Token
            roles = [role.name for role in user.roles]
            permissions = []
            for role in user.roles:
                for perm in role.permissions:
                    if perm.code not in permissions:
                        permissions.append(perm.code)
            
            new_access_token, access_jti = JWTAuthMiddleware.create_access_token(
                user_id=str(user.id),
                roles=roles,
                permissions=permissions,
            )
            
            new_refresh_token, refresh_jti = JWTAuthMiddleware.create_refresh_token(
                user_id=str(user.id),
            )
            
            # 缓存新 Token
            if self.redis:
                await JWTAuthMiddleware.cache_token(
                    redis=self.redis,
                    jti=access_jti,
                    user_id=str(user.id),
                    token_type="access",
                    expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
                )
                
                await JWTAuthMiddleware.cache_token(
                    redis=self.redis,
                    jti=refresh_jti,
                    user_id=str(user.id),
                    token_type="refresh",
                    expires_in=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
                )
            
            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "bearer",
                "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            }
            
        except (TokenExpiredError, TokenInvalidError):
            raise
        except Exception as e:
            raise TokenInvalidError(str(e))
    
    async def revoke_all_tokens(self, user_id: UUID) -> int:
        """
        撤销用户所有 Token
        
        用于密码变更或安全锁定
        """
        if self.redis:
            # 设置用户级别的撤销标记
            revoke_key = f"revoke_all:{user_id}"
            await self.redis.setex(
                revoke_key,
                settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
                datetime.utcnow().isoformat()
            )
        
        # 清理数据库中未过期的 Token 黑名单记录
        count = self.db.query(TokenBlacklist).filter(
            TokenBlacklist.user_id == user_id,
            TokenBlacklist.expires_at > datetime.utcnow()
        ).count()
        
        return count

