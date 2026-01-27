"""
用户管理 API 端点

所有业务逻辑通过服务层实现
"""
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from ....db.session import get_db
from ....services.user_service import UserService
from ....middleware.auth import get_current_active_user, require_permissions
from ....models.user import User
from ....schemas.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse,
    PasswordChange,
    PasswordReset,
    AssignRolesRequest,
)
from ....schemas.common import MessageResponse
from ....core.exceptions import NotFoundError, AlreadyExistsError, BusinessError

router = APIRouter()


@router.get("", response_model=UserListResponse)
async def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    is_active: Optional[bool] = None,
    role_name: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.view")),
):
    """
    获取用户列表
    """
    user_service = UserService(db)
    users, total = user_service.list_users(
        skip=skip,
        limit=limit,
        is_active=is_active,
        role_name=role_name,
        search=search,
    )
    
    return UserListResponse(
        items=users,
        total=total,
        skip=skip,
        limit=limit
    )


@router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.create")),
):
    """
    创建用户
    """
    user_service = UserService(db)
    
    try:
        user = user_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            phone=user_data.phone,
            is_active=user_data.is_active,
            role_ids=user_data.role_ids,
        )
        return user
    except AlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.message
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.view")),
):
    """
    获取用户详情
    """
    user_service = UserService(db)
    user = user_service.get_by_id(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: UUID,
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.update")),
):
    """
    更新用户
    """
    user_service = UserService(db)
    
    try:
        user = user_service.update_user(
            user_id=user_id,
            data=user_data.model_dump(exclude_unset=True),
            updated_by=current_user.id
        )
        return user
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


@router.delete("/{user_id}", response_model=MessageResponse)
async def delete_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.delete")),
):
    """
    删除用户
    """
    user_service = UserService(db)
    
    try:
        user_service.delete_user(user_id)
        return MessageResponse(message="User deleted successfully")
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


@router.post("/{user_id}/change-password", response_model=MessageResponse)
async def change_password(
    user_id: UUID,
    password_data: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    修改密码（用户自己操作）
    """
    # 只允许用户修改自己的密码
    if current_user.id != user_id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only change your own password"
        )
    
    user_service = UserService(db)
    
    try:
        user_service.change_password(
            user_id=user_id,
            old_password=password_data.old_password,
            new_password=password_data.new_password
        )
        return MessageResponse(message="Password changed successfully")
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


@router.post("/{user_id}/reset-password", response_model=MessageResponse)
async def reset_password(
    user_id: UUID,
    password_data: PasswordReset,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.update")),
):
    """
    重置密码（管理员操作）
    """
    user_service = UserService(db)
    
    try:
        user_service.reset_password(
            user_id=user_id,
            new_password=password_data.new_password,
            reset_by=current_user.id
        )
        return MessageResponse(message="Password reset successfully")
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.post("/{user_id}/roles", response_model=UserResponse)
async def assign_roles(
    user_id: UUID,
    roles_data: AssignRolesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("roles.manage")),
):
    """
    分配角色
    """
    user_service = UserService(db)
    
    try:
        user = user_service.assign_roles(
            user_id=user_id,
            role_ids=roles_data.role_ids,
            assigned_by=current_user.id
        )
        return user
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


@router.post("/{user_id}/activate", response_model=UserResponse)
async def activate_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.update")),
):
    """
    激活用户
    """
    user_service = UserService(db)
    
    try:
        user = user_service.activate_user(user_id)
        return user
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )


@router.post("/{user_id}/deactivate", response_model=UserResponse)
async def deactivate_user(
    user_id: UUID,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("users.update")),
):
    """
    停用用户
    """
    user_service = UserService(db)
    
    try:
        user = user_service.deactivate_user(user_id)
        return user
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message
        )

