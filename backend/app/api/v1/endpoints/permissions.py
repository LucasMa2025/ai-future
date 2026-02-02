"""
权限管理 API 端点
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....db.session import get_db
from ....services.system_permission_service import SystemPermissionService
from ....core.security import get_current_user
from ....models.user import User

router = APIRouter()


# ==================== Schema 定义 ====================

class PermissionCreate(BaseModel):
    code: str = Field(..., description="权限代码")
    name: str = Field(..., description="权限名称")
    function_id: Optional[int] = None
    action: str = "access"
    description: Optional[str] = None
    resource_type: Optional[str] = None
    resource_scope: str = "all"


class RolePermissionAssign(BaseModel):
    permission_ids: List[int] = Field(..., description="权限ID列表")


class UserPermissionAssign(BaseModel):
    permissions: List[dict] = Field(..., description="权限列表，格式: [{permission_id: 1, grant_type: 'allow'}]")


class PermissionCheckRequest(BaseModel):
    permission_code: str = Field(..., description="权限代码")


# ==================== API 端点 ====================

@router.get("/", summary="获取所有权限")
async def list_permissions(
    include_disabled: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取所有权限"""
    service = SystemPermissionService(db)
    permissions = service.get_all_permissions(include_disabled)
    
    return {
        "success": True,
        "data": [
            {
                "id": p.id,
                "code": p.code,
                "name": p.name,
                "description": p.description,
                "function_id": p.function_id,
                "action": p.action,
                "resource_type": p.resource_type,
                "resource_scope": p.resource_scope,
                "is_enabled": p.is_enabled,
            }
            for p in permissions
        ]
    }


@router.get("/tree", summary="获取权限树")
async def get_permission_tree(
    role_id: Optional[int] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    获取权限树，带有授权状态
    
    用于权限分配页面展示
    """
    service = SystemPermissionService(db)
    tree = service.get_permission_tree_with_status(user_id, role_id)
    
    return {
        "success": True,
        "data": tree
    }


@router.post("/", summary="创建权限")
async def create_permission(
    data: PermissionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建权限"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以创建权限"
        )
    
    service = SystemPermissionService(db)
    
    try:
        permission = service.create_permission(
            code=data.code,
            name=data.name,
            function_id=data.function_id,
            action=data.action,
            description=data.description,
            resource_type=data.resource_type,
            resource_scope=data.resource_scope,
        )
        
        return {
            "success": True,
            "message": "权限创建成功",
            "data": {"id": permission.id}
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


# ==================== 角色权限管理 ====================

@router.get("/role/{role_id}", summary="获取角色权限")
async def get_role_permissions(
    role_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取角色的所有权限"""
    service = SystemPermissionService(db)
    permissions = service.get_role_permissions(role_id)
    
    return {
        "success": True,
        "data": [
            {
                "id": p.id,
                "code": p.code,
                "name": p.name,
                "action": p.action,
            }
            for p in permissions
        ]
    }


@router.put("/role/{role_id}", summary="设置角色权限")
async def set_role_permissions(
    role_id: int,
    data: RolePermissionAssign,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """设置角色的权限（替换现有权限）"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以设置角色权限"
        )
    
    service = SystemPermissionService(db)
    service.set_role_permissions(role_id, data.permission_ids)
    
    return {
        "success": True,
        "message": "角色权限设置成功"
    }


# ==================== 用户权限管理 ====================

@router.get("/user/{user_id}", summary="获取用户权限")
async def get_user_permissions(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取用户的直接权限"""
    service = SystemPermissionService(db)
    permissions = service.get_user_direct_permissions(user_id)
    
    return {
        "success": True,
        "data": permissions
    }


@router.get("/user/{user_id}/effective", summary="获取用户有效权限")
async def get_user_effective_permissions(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取用户的有效权限（合并角色和直接权限）"""
    service = SystemPermissionService(db)
    effective = service.get_user_effective_permissions(user_id)
    
    return {
        "success": True,
        "data": effective
    }


@router.put("/user/{user_id}", summary="设置用户权限")
async def set_user_permissions(
    user_id: str,
    data: UserPermissionAssign,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """设置用户的直接权限"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以设置用户权限"
        )
    
    service = SystemPermissionService(db)
    service.set_user_permissions(
        user_id, 
        data.permissions, 
        granted_by=str(current_user.id)
    )
    
    return {
        "success": True,
        "message": "用户权限设置成功"
    }


# ==================== 权限检查 ====================

@router.post("/check", summary="检查权限")
async def check_permission(
    data: PermissionCheckRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """检查当前用户是否拥有指定权限"""
    service = SystemPermissionService(db)
    has_permission = service.check_permission(str(current_user.id), data.permission_code)
    
    return {
        "success": True,
        "data": {
            "permission_code": data.permission_code,
            "allowed": has_permission
        }
    }


@router.get("/my", summary="获取我的权限")
async def get_my_permissions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取当前用户的所有有效权限"""
    service = SystemPermissionService(db)
    effective = service.get_user_effective_permissions(str(current_user.id))
    
    return {
        "success": True,
        "data": effective
    }
