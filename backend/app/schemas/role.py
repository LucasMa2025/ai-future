"""
角色相关 Schema
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field


class PermissionBasic(BaseModel):
    """权限基本信息"""
    id: int
    code: str
    name: str
    action: str
    
    class Config:
        from_attributes = True


class RoleCreate(BaseModel):
    """创建角色请求"""
    name: str = Field(..., min_length=2, max_length=50, pattern=r'^[a-z][a-z0-9_]*$')
    display_name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = None
    risk_level_limit: Optional[str] = Field(None, pattern=r'^(low|medium|high|critical)$')
    permission_ids: Optional[List[int]] = None
    sort_order: int = 0


class RoleUpdate(BaseModel):
    """更新角色请求"""
    name: Optional[str] = Field(None, min_length=2, max_length=50, pattern=r'^[a-z][a-z0-9_]*$')
    display_name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = None
    risk_level_limit: Optional[str] = Field(None, pattern=r'^(low|medium|high|critical)$')
    permission_ids: Optional[List[int]] = None
    sort_order: Optional[int] = None


class RoleResponse(BaseModel):
    """角色响应"""
    id: int
    name: str
    display_name: str
    description: Optional[str] = None
    risk_level_limit: Optional[str] = None
    is_system: bool
    sort_order: int
    permissions: List[PermissionBasic] = []
    user_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class RoleListResponse(BaseModel):
    """角色列表响应"""
    items: List[RoleResponse]
    total: int
    skip: int
    limit: int


class AssignPermissionsRequest(BaseModel):
    """分配权限请求"""
    permission_ids: List[int]

