"""
JWT 认证中间件

实现:
1. JWT Token 验证
2. Redis 缓存验证 (快速路径)
3. PostgreSQL 黑名单验证 (持久化)
4. 用户信息注入
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import uuid
import json

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from redis.asyncio import Redis

from ..config import settings
from ..db.session import get_db
from ..db.redis import get_token_redis
from ..models.user import User
from ..models.token import TokenBlacklist
from ..core.exceptions import (
    TokenExpiredError, 
    TokenInvalidError, 
    TokenBlacklistedError,
    UserInactiveError
)

# HTTP Bearer 认证方案
security = HTTPBearer(auto_error=False)


class TokenPayload:
    """Token 载荷"""
    
    def __init__(
        self,
        sub: str,  # 用户ID
        jti: str,  # JWT ID
        exp: datetime,
        iat: datetime,
        type: str = "access",  # access or refresh
        roles: list = None,
        permissions: list = None,
    ):
        self.sub = sub
        self.jti = jti
        self.exp = exp
        self.iat = iat
        self.type = type
        self.roles = roles or []
        self.permissions = permissions or []


class JWTAuthMiddleware:
    """JWT 认证处理器"""
    
    @staticmethod
    def create_access_token(
        user_id: str,
        roles: list = None,
        permissions: list = None,
        expires_delta: Optional[timedelta] = None
    ) -> tuple[str, str]:
        """
        创建访问 Token
        
        Returns:
            (token, jti)
        """
        jti = str(uuid.uuid4())
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        payload = {
            "sub": str(user_id),
            "jti": jti,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access",
            "roles": roles or [],
            "permissions": permissions or [],
        }
        
        token = jwt.encode(
            payload, 
            settings.SECRET_KEY, 
            algorithm=settings.ALGORITHM
        )
        
        return token, jti
    
    @staticmethod
    def create_refresh_token(
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> tuple[str, str]:
        """
        创建刷新 Token
        
        Returns:
            (token, jti)
        """
        jti = str(uuid.uuid4())
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=settings.REFRESH_TOKEN_EXPIRE_DAYS
            )
        
        payload = {
            "sub": str(user_id),
            "jti": jti,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
        }
        
        token = jwt.encode(
            payload, 
            settings.SECRET_KEY, 
            algorithm=settings.ALGORITHM
        )
        
        return token, jti
    
    @staticmethod
    def decode_token(token: str) -> TokenPayload:
        """解码并验证 Token"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            
            # 检查过期
            exp = datetime.fromtimestamp(payload.get("exp", 0))
            if exp < datetime.utcnow():
                raise TokenExpiredError()
            
            return TokenPayload(
                sub=payload.get("sub"),
                jti=payload.get("jti"),
                exp=exp,
                iat=datetime.fromtimestamp(payload.get("iat", 0)),
                type=payload.get("type", "access"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
            )
            
        except JWTError as e:
            raise TokenInvalidError(f"Invalid token: {str(e)}")
    
    @staticmethod
    async def cache_token(
        redis: Redis,
        jti: str,
        user_id: str,
        token_type: str,
        expires_in: int
    ) -> None:
        """缓存 Token 到 Redis"""
        key = f"token:{jti}"
        data = {
            "user_id": user_id,
            "type": token_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        await redis.setex(key, expires_in, json.dumps(data))
    
    @staticmethod
    async def is_token_blacklisted(
        redis: Redis,
        db: Session,
        jti: str
    ) -> bool:
        """
        检查 Token 是否在黑名单中
        
        先查 Redis 缓存，再查数据库
        """
        # 1. 检查 Redis 缓存
        blacklist_key = f"blacklist:{jti}"
        if await redis.exists(blacklist_key):
            return True
        
        # 2. 检查数据库
        blacklisted = db.query(TokenBlacklist).filter(
            TokenBlacklist.jti == jti
        ).first()
        
        if blacklisted:
            # 写入 Redis 缓存
            ttl = int((blacklisted.expires_at - datetime.utcnow()).total_seconds())
            if ttl > 0:
                await redis.setex(blacklist_key, ttl, "1")
            return True
        
        return False
    
    @staticmethod
    async def blacklist_token(
        redis: Redis,
        db: Session,
        jti: str,
        user_id: str,
        expires_at: datetime,
        reason: str = "logout"
    ) -> None:
        """
        将 Token 加入黑名单
        
        同时写入 Redis 和数据库
        """
        # 1. 写入数据库
        blacklist_entry = TokenBlacklist(
            jti=jti,
            user_id=user_id,
            token_type="access",
            reason=reason,
            expires_at=expires_at,
        )
        db.add(blacklist_entry)
        db.commit()
        
        # 2. 写入 Redis
        ttl = int((expires_at - datetime.utcnow()).total_seconds())
        if ttl > 0:
            blacklist_key = f"blacklist:{jti}"
            await redis.setex(blacklist_key, ttl, "1")
        
        # 3. 删除 Token 缓存
        token_key = f"token:{jti}"
        await redis.delete(token_key)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_token_redis),
) -> User:
    """
    获取当前用户
    
    FastAPI 依赖注入函数
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        # 1. 解码 Token
        payload = JWTAuthMiddleware.decode_token(token)
        
        # 2. 检查黑名单
        if await JWTAuthMiddleware.is_token_blacklisted(redis, db, payload.jti):
            raise TokenBlacklistedError()
        
        # 3. 获取用户
        user = db.query(User).filter(User.id == payload.sub).first()
        if not user:
            raise TokenInvalidError("User not found")
        
        # 4. 将 Token 信息存入请求
        request.state.token_payload = payload
        request.state.user = user
        
        return user
        
    except (TokenExpiredError, TokenInvalidError, TokenBlacklistedError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """获取当前激活用户"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return current_user


def require_permissions(*permissions: str):
    """
    权限检查装饰器
    
    用法:
        @router.get("/", dependencies=[Depends(require_permissions("users.view"))])
    """
    async def permission_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.is_superuser:
            return current_user
        
        # 收集用户所有权限
        user_permissions = set()
        for role in current_user.roles:
            for perm in role.permissions:
                user_permissions.add(perm.code)
        
        # 检查所需权限
        for required in permissions:
            if required not in user_permissions:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {required}"
                )
        
        return current_user
    
    return permission_checker


def require_roles(*roles: str):
    """
    角色检查装饰器
    
    用法:
        @router.get("/", dependencies=[Depends(require_roles("admin"))])
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.is_superuser:
            return current_user
        
        user_roles = {role.name for role in current_user.roles}
        
        if not any(role in user_roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(roles)}"
            )
        
        return current_user
    
    return role_checker

