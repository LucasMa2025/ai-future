"""
认证相关 Schema
"""
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=1, max_length=100, description="用户名或邮箱")
    password: str = Field(..., min_length=1, description="密码")


class UserInfo(BaseModel):
    """用户信息"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    is_superuser: bool = False
    roles: List[str] = []


class LoginResponse(BaseModel):
    """登录响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserInfo


class TokenRefreshRequest(BaseModel):
    """Token 刷新请求"""
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    """Token 刷新响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class LogoutRequest(BaseModel):
    """登出请求"""
    all_devices: bool = Field(default=False, description="是否注销所有设备")

