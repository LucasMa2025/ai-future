"""
认证相关 API 端点

所有业务逻辑通过服务层实现
"""
from fastapi import APIRouter, Depends, Request, HTTPException, status
from sqlalchemy.orm import Session
from redis.asyncio import Redis

from ....db.session import get_db
from ....db.redis import get_token_redis
from ....services.auth_service import AuthService
from ....middleware.auth import get_current_active_user
from ....models.user import User
from ....schemas.auth import (
    LoginRequest,
    LoginResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
)
from ....schemas.common import MessageResponse

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_token_redis),
):
    """
    用户登录
    
    返回访问令牌和刷新令牌
    """
    # 获取客户端 IP
    ip_address = request.client.host if request.client else None
    
    # 调用服务层
    auth_service = AuthService(db, redis)
    result = await auth_service.login(
        username=login_data.username,
        password=login_data.password,
        ip_address=ip_address
    )
    
    return result


@router.post("/refresh", response_model=TokenRefreshResponse)
async def refresh_token(
    refresh_data: TokenRefreshRequest,
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_token_redis),
):
    """
    刷新访问令牌
    """
    auth_service = AuthService(db, redis)
    result = await auth_service.refresh_token(refresh_data.refresh_token)
    
    return result


@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    db: Session = Depends(get_db),
    redis: Redis = Depends(get_token_redis),
    current_user: User = Depends(get_current_active_user),
):
    """
    用户登出
    
    将当前令牌加入黑名单
    """
    auth_service = AuthService(db, redis)
    
    # 从请求状态获取 Token 信息
    token_payload = getattr(request.state, "token_payload", None)
    if token_payload:
        await auth_service.logout(
            jti=token_payload.jti,
            user_id=current_user.id,
            expires_at=token_payload.exp
        )
    
    return MessageResponse(message="Logged out successfully")


@router.get("/me", response_model=dict)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user),
):
    """
    获取当前用户信息
    """
    return {
        "id": str(current_user.id),
        "username": current_user.username,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "is_superuser": current_user.is_superuser,
        "is_active": current_user.is_active,
        "roles": [
            {"id": role.id, "name": role.name, "display_name": role.display_name}
            for role in current_user.roles
        ],
        "last_login": current_user.last_login.isoformat() if current_user.last_login else None,
    }

