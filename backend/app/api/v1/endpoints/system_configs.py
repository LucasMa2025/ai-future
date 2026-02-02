"""
系统配置 API 端点
"""
from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....db.session import get_db
from ....services.system_config_service import SystemConfigService
from ....core.security import get_current_user
from ....models.user import User

router = APIRouter()


# ==================== Schema 定义 ====================

class ConfigCreate(BaseModel):
    config_key: str = Field(..., description="配置键")
    config_value: str = Field(..., description="配置值")
    value_type: str = Field("string", description="值类型")
    config_group: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    is_readonly: bool = False
    is_secret: bool = False
    default_value: Optional[str] = None


class ConfigUpdate(BaseModel):
    value: Any = Field(..., description="新的配置值")


class BatchConfigUpdate(BaseModel):
    configs: Dict[str, Any] = Field(..., description="配置键值对")


# ==================== API 端点 ====================

@router.get("/", summary="获取所有配置")
async def list_configs(
    group: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取所有系统配置"""
    service = SystemConfigService(db)
    configs = service.get_all_configs(group=group, include_secret=current_user.is_superuser)
    
    return {
        "success": True,
        "data": configs
    }


@router.get("/groups", summary="获取配置分组")
async def get_config_groups(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取配置分组信息"""
    service = SystemConfigService(db)
    groups = service.get_config_groups()
    
    return {
        "success": True,
        "data": groups
    }


@router.get("/group/{group}", summary="获取分组配置")
async def get_group_configs(
    group: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取指定分组的所有配置"""
    service = SystemConfigService(db)
    configs = service.get_configs_by_group(group)
    
    return {
        "success": True,
        "data": configs
    }


@router.get("/{config_key}", summary="获取单个配置")
async def get_config(
    config_key: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取单个配置值"""
    service = SystemConfigService(db)
    config = service.get_config(config_key)
    
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="配置不存在"
        )
    
    return {
        "success": True,
        "data": service._config_to_dict(config, current_user.is_superuser)
    }


@router.put("/{config_key}", summary="更新配置")
async def update_config(
    config_key: str,
    data: ConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """更新单个配置值"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以修改配置"
        )
    
    service = SystemConfigService(db)
    
    try:
        config = service.set_config(
            config_key,
            data.value,
            updated_by=str(current_user.id)
        )
        
        return {
            "success": True,
            "message": "配置更新成功",
            "data": service._config_to_dict(config)
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.put("/", summary="批量更新配置")
async def batch_update_configs(
    data: BatchConfigUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """批量更新配置"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以修改配置"
        )
    
    service = SystemConfigService(db)
    configs = service.batch_set_configs(
        data.configs,
        updated_by=str(current_user.id)
    )
    
    return {
        "success": True,
        "message": f"已更新 {len(configs)} 个配置"
    }


@router.post("/", summary="创建配置")
async def create_config(
    data: ConfigCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建新配置项"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以创建配置"
        )
    
    service = SystemConfigService(db)
    
    try:
        config = service.create_config(
            config_key=data.config_key,
            config_value=data.config_value,
            value_type=data.value_type,
            config_group=data.config_group,
            display_name=data.display_name,
            description=data.description,
            is_readonly=data.is_readonly,
            is_secret=data.is_secret,
            default_value=data.default_value,
        )
        
        return {
            "success": True,
            "message": "配置创建成功",
            "data": {"id": config.id}
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/init", summary="初始化默认配置")
async def init_configs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """初始化默认配置"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以初始化配置"
        )
    
    service = SystemConfigService(db)
    count = service.init_default_configs()
    
    return {
        "success": True,
        "message": f"已初始化 {count} 个配置",
        "data": {"count": count}
    }
