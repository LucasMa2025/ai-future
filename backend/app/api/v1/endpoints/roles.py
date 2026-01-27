"""
角色管理 API 端点

所有业务逻辑通过服务层实现
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from ....db.session import get_db
from ....services.role_service import RoleService
from ....middleware.auth import require_permissions
from ....models.user import User
from ....schemas.role import (
    RoleCreate,
    RoleUpdate,
    RoleResponse,
    RoleListResponse,
    AssignPermissionsRequest,
)
from ....schemas.common import MessageResponse
from ....core.exceptions import NotFoundError, AlreadyExistsError, BusinessError

router = APIRouter()


@router.get("", response_model=RoleListResponse)
async def list_roles(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    include_system: bool = Query(True),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.view")),
):
    """
    获取角色列表
    """
    role_service = RoleService(db)
    roles, total = role_service.list_roles(
        include_system=include_system,
        skip=skip,
        limit=limit,
    )
    
    # 添加用户数量
    result = []
    for role in roles:
        role_dict = {
            "id": role.id,
            "name": role.name,
            "display_name": role.display_name,
            "description": role.description,
            "risk_level_limit": role.risk_level_limit,
            "is_system": role.is_system,
            "sort_order": role.sort_order,
            "permissions": role.permissions,
            "user_count": len(role.users),
            "created_at": role.created_at,
            "updated_at": role.updated_at,
        }
        result.append(role_dict)
    
    return RoleListResponse(
        items=result,
        total=total,
        skip=skip,
        limit=limit
    )


@router.post("", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
async def create_role(
    role_data: RoleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.manage")),
):
    """
    创建角色
    """
    role_service = RoleService(db)
    
    try:
        role = role_service.create_role(
            name=role_data.name,
            display_name=role_data.display_name,
            description=role_data.description,
            risk_level_limit=role_data.risk_level_limit,
            permission_ids=role_data.permission_ids,
            sort_order=role_data.sort_order,
        )
        
        return RoleResponse(
            id=role.id,
            name=role.name,
            display_name=role.display_name,
            description=role.description,
            risk_level_limit=role.risk_level_limit,
            is_system=role.is_system,
            sort_order=role.sort_order,
            permissions=role.permissions,
            user_count=0,
            created_at=role.created_at,
            updated_at=role.updated_at,
        )
    except AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )


@router.get("/{role_id}", response_model=RoleResponse)
async def get_role(
    role_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.view")),
):
    """
    获取角色详情
    """
    role_service = RoleService(db)
    role = role_service.get_by_id(role_id)
    
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Role not found"
        )
    
    return RoleResponse(
        id=role.id,
        name=role.name,
        display_name=role.display_name,
        description=role.description,
        risk_level_limit=role.risk_level_limit,
        is_system=role.is_system,
        sort_order=role.sort_order,
        permissions=role.permissions,
        user_count=len(role.users),
        created_at=role.created_at,
        updated_at=role.updated_at,
    )


@router.put("/{role_id}", response_model=RoleResponse)
async def update_role(
    role_id: int,
    role_data: RoleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.manage")),
):
    """
    更新角色
    """
    role_service = RoleService(db)
    
    try:
        role = role_service.update_role(
            role_id=role_id,
            data=role_data.model_dump(exclude_unset=True)
        )
        
        return RoleResponse(
            id=role.id,
            name=role.name,
            display_name=role.display_name,
            description=role.description,
            risk_level_limit=role.risk_level_limit,
            is_system=role.is_system,
            sort_order=role.sort_order,
            permissions=role.permissions,
            user_count=len(role.users),
            created_at=role.created_at,
            updated_at=role.updated_at,
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )


@router.delete("/{role_id}", response_model=MessageResponse)
async def delete_role(
    role_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.manage")),
):
    """
    删除角色
    """
    role_service = RoleService(db)
    
    try:
        role_service.delete_role(role_id)
        return MessageResponse(message="Role deleted successfully")
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except BusinessError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )


@router.post("/{role_id}/permissions", response_model=RoleResponse)
async def assign_permissions(
    role_id: int,
    permissions_data: AssignPermissionsRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.manage")),
):
    """
    分配权限
    """
    role_service = RoleService(db)
    
    try:
        role = role_service.assign_permissions(
            role_id=role_id,
            permission_ids=permissions_data.permission_ids
        )
        
        return RoleResponse(
            id=role.id,
            name=role.name,
            display_name=role.display_name,
            description=role.description,
            risk_level_limit=role.risk_level_limit,
            is_system=role.is_system,
            sort_order=role.sort_order,
            permissions=role.permissions,
            user_count=len(role.users),
            created_at=role.created_at,
            updated_at=role.updated_at,
        )
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )
    except BusinessError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message
        )

