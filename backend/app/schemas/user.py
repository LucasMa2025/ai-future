"""
用户相关 Schema
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID


class RoleBasic(BaseModel):
    """角色基本信息"""
    id: int
    name: str
    display_name: str
    
    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """创建用户请求"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    is_active: bool = True
    role_ids: Optional[List[int]] = None


class UserUpdate(BaseModel):
    """更新用户请求"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    is_active: Optional[bool] = None
    avatar: Optional[str] = None
    role_ids: Optional[List[int]] = None
    notification_preferences: Optional[Dict[str, Any]] = None


class PasswordChange(BaseModel):
    """修改密码请求"""
    old_password: str
    new_password: str = Field(..., min_length=6, max_length=100)


class PasswordReset(BaseModel):
    """重置密码请求"""
    new_password: str = Field(..., min_length=6, max_length=100)


class UserResponse(BaseModel):
    """用户响应"""
    id: UUID
    username: str
    email: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    avatar: Optional[str] = None
    is_active: bool
    is_superuser: bool
    last_login: Optional[datetime] = None
    login_count: int = 0
    roles: List[RoleBasic] = []
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """用户列表响应"""
    items: List[UserResponse]
    total: int
    skip: int
    limit: int


class AssignRolesRequest(BaseModel):
    """分配角色请求"""
    role_ids: List[int]

