"""
数据备份 API 端点
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from ....db.session import get_db
from ....services.data_backup_service import DataBackupService
from ....core.security import get_current_user
from ....models.user import User

router = APIRouter()


# ==================== Schema 定义 ====================

class BackupCreate(BaseModel):
    backup_name: str = Field(..., description="备份名称")
    backup_type: str = Field("full", description="备份类型: full/tables")
    tables: Optional[List[str]] = Field(None, description="要备份的表列表")
    compress: bool = Field(True, description="是否压缩")
    description: Optional[str] = None


class BackupRestore(BaseModel):
    tables: Optional[List[str]] = Field(None, description="要恢复的表列表，不指定则恢复全部")


# ==================== API 端点 ====================

@router.get("/", summary="获取备份列表")
async def list_backups(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    status: Optional[str] = None,
    backup_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取备份列表"""
    service = DataBackupService(db)
    result = service.get_backups(
        page=page,
        page_size=page_size,
        status=status,
        backup_type=backup_type,
    )
    
    return {
        "success": True,
        "data": result
    }


@router.get("/stats", summary="获取备份统计")
async def get_backup_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取备份统计信息"""
    service = DataBackupService(db)
    stats = service.get_backup_stats()
    
    return {
        "success": True,
        "data": stats
    }


@router.get("/tables", summary="获取可备份表信息")
async def get_table_info(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取可备份表的信息"""
    service = DataBackupService(db)
    tables = service.get_table_info()
    
    return {
        "success": True,
        "data": tables
    }


@router.get("/{backup_id}", summary="获取备份详情")
async def get_backup(
    backup_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取备份详情"""
    service = DataBackupService(db)
    backup = service.get_backup_by_id(backup_id)
    
    if not backup:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="备份不存在"
        )
    
    return {
        "success": True,
        "data": service._backup_to_dict(backup)
    }


@router.post("/", summary="创建备份")
async def create_backup(
    data: BackupCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """创建数据备份"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以创建备份"
        )
    
    service = DataBackupService(db)
    
    try:
        backup = service.create_backup(
            backup_name=data.backup_name,
            backup_type=data.backup_type,
            tables=data.tables,
            compress=data.compress,
            description=data.description,
            created_by=str(current_user.id),
            async_mode=True,
        )
        
        return {
            "success": True,
            "message": "备份任务已创建，正在后台执行",
            "data": {
                "id": backup.id,
                "status": backup.status
            }
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/{backup_id}/restore", summary="恢复备份")
async def restore_backup(
    backup_id: int,
    data: BackupRestore,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """恢复备份"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以恢复备份"
        )
    
    service = DataBackupService(db)
    
    try:
        result = service.restore_backup(
            backup_id=backup_id,
            tables=data.tables,
        )
        
        return {
            "success": True,
            "message": "备份恢复完成",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{backup_id}", summary="删除备份")
async def delete_backup(
    backup_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """删除备份"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以删除备份"
        )
    
    service = DataBackupService(db)
    
    if not service.delete_backup(backup_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="备份不存在"
        )
    
    return {
        "success": True,
        "message": "备份已删除"
    }
