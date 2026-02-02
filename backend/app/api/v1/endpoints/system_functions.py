"""
系统功能 API 端点
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....db.session import get_db
from ....services.system_function_service import SystemFunctionService
from ....core.security import get_current_user
from ....models.user import User

router = APIRouter()


# ==================== Schema 定义 ====================

class SystemFunctionBase(BaseModel):
    code: str = Field(..., description="功能代码")
    name: str = Field(..., description="功能名称")
    description: Optional[str] = None
    level: int = Field(1, ge=1, le=3, description="层级: 1=模块, 2=功能组, 3=API端点")
    parent_id: Optional[int] = None
    module: Optional[str] = None
    method: Optional[str] = None
    api_path: Optional[str] = None
    icon: Optional[str] = None
    is_visible: bool = True
    is_audited: bool = True
    audit_level: str = "normal"
    sort_order: int = 0
    is_enabled: bool = True


class SystemFunctionCreate(SystemFunctionBase):
    pass


class SystemFunctionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    is_visible: Optional[bool] = None
    is_audited: Optional[bool] = None
    audit_level: Optional[str] = None
    sort_order: Optional[int] = None
    is_enabled: Optional[bool] = None


class SystemFunctionResponse(SystemFunctionBase):
    id: int
    
    class Config:
        from_attributes = True


# ==================== API 端点 ====================

@router.get("/", summary="获取系统功能列表")
async def list_functions(
    include_disabled: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取系统功能列表"""
    service = SystemFunctionService(db)
    functions = service.get_all_functions(include_disabled)
    return {
        "success": True,
        "data": [
            {
                "id": f.id,
                "code": f.code,
                "name": f.name,
                "description": f.description,
                "level": f.level,
                "parent_id": f.parent_id,
                "module": f.module,
                "method": f.method,
                "api_path": f.api_path,
                "icon": f.icon,
                "is_visible": f.is_visible,
                "is_audited": f.is_audited,
                "audit_level": f.audit_level,
                "sort_order": f.sort_order,
                "is_enabled": f.is_enabled,
            }
            for f in functions
        ]
    }


@router.get("/tree", summary="获取系统功能树")
async def get_function_tree(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取系统功能树形结构"""
    service = SystemFunctionService(db)
    tree = service.get_function_tree()
    return {
        "success": True,
        "data": tree
    }


@router.get("/{function_id}", summary="获取功能详情")
async def get_function(
    function_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取单个功能详情"""
    service = SystemFunctionService(db)
    function = service.get_function_by_id(function_id)
    
    if not function:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="功能不存在"
        )
    
    return {
        "success": True,
        "data": {
            "id": function.id,
            "code": function.code,
            "name": function.name,
            "description": function.description,
            "level": function.level,
            "parent_id": function.parent_id,
            "module": function.module,
            "method": function.method,
            "api_path": function.api_path,
            "icon": function.icon,
            "is_visible": function.is_visible,
            "is_audited": function.is_audited,
            "audit_level": function.audit_level,
            "sort_order": function.sort_order,
            "is_enabled": function.is_enabled,
        }
    }


@router.post("/", summary="创建系统功能")
async def create_function(
    data: SystemFunctionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建系统功能"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以创建系统功能"
        )
    
    service = SystemFunctionService(db)
    
    try:
        function = service.create_function(
            code=data.code,
            name=data.name,
            level=data.level,
            parent_id=data.parent_id,
            description=data.description,
            module=data.module,
            method=data.method,
            api_path=data.api_path,
            icon=data.icon,
            is_visible=data.is_visible,
            is_audited=data.is_audited,
            audit_level=data.audit_level,
            sort_order=data.sort_order,
            is_enabled=data.is_enabled,
        )
        
        return {
            "success": True,
            "message": "功能创建成功",
            "data": {"id": function.id}
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/{function_id}", summary="更新系统功能")
async def update_function(
    function_id: int,
    data: SystemFunctionUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新系统功能"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以更新系统功能"
        )
    
    service = SystemFunctionService(db)
    
    try:
        update_data = data.model_dump(exclude_unset=True)
        function = service.update_function(function_id, **update_data)
        
        if not function:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="功能不存在"
            )
        
        return {
            "success": True,
            "message": "功能更新成功"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{function_id}", summary="删除系统功能")
async def delete_function(
    function_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除系统功能"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以删除系统功能"
        )
    
    service = SystemFunctionService(db)
    
    if not service.delete_function(function_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="功能不存在"
        )
    
    return {
        "success": True,
        "message": "功能删除成功"
    }


@router.post("/init", summary="初始化默认功能")
async def init_functions(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """初始化默认系统功能"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以初始化系统功能"
        )
    
    service = SystemFunctionService(db)
    functions = service.init_default_functions()
    
    return {
        "success": True,
        "message": f"已初始化 {len(functions)} 个功能",
        "data": {"count": len(functions)}
    }
