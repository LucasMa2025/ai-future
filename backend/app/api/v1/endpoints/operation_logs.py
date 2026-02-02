"""
操作日志 API 端点
"""
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from ....db.session import get_db
from ....services.operation_log_service import OperationLogService
from ....core.security import get_current_user
from ....models.user import User

router = APIRouter()


@router.get("/", summary="获取操作日志列表")
async def list_operation_logs(
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页数量"),
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    method: Optional[str] = None,
    path: Optional[str] = None,
    function_code: Optional[str] = None,
    is_success: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    ip_address: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    获取操作日志列表
    
    支持多种筛选条件
    """
    service = OperationLogService(db)
    
    result = service.get_logs(
        page=page,
        page_size=page_size,
        user_id=user_id,
        username=username,
        method=method,
        path=path,
        function_code=function_code,
        is_success=is_success,
        start_date=start_date,
        end_date=end_date,
        ip_address=ip_address,
    )
    
    return {
        "success": True,
        "data": result
    }


@router.get("/statistics", summary="获取日志统计")
async def get_log_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取操作日志统计信息"""
    service = OperationLogService(db)
    stats = service.get_statistics(start_date, end_date)
    
    return {
        "success": True,
        "data": stats
    }


@router.get("/export", summary="导出操作日志")
async def export_operation_logs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    username: Optional[str] = None,
    method: Optional[str] = None,
    is_success: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """导出操作日志为 CSV"""
    service = OperationLogService(db)
    
    csv_content = service.export_to_csv(
        start_date=start_date,
        end_date=end_date,
        username=username,
        method=method,
        is_success=is_success,
    )
    
    # 添加 BOM 以支持 Excel 打开中文
    csv_bytes = ("\ufeff" + csv_content).encode("utf-8")
    
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=operation_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    )


@router.get("/{log_id}", summary="获取日志详情")
async def get_operation_log(
    log_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """获取单条操作日志详情"""
    service = OperationLogService(db)
    log = service.get_log_by_id(log_id)
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="日志不存在"
        )
    
    return {
        "success": True,
        "data": service._log_to_dict(log)
    }


@router.delete("/cleanup", summary="清理旧日志")
async def cleanup_old_logs(
    days: int = Query(90, ge=1, le=365, description="保留天数"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """清理指定天数前的日志"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有超级管理员可以清理日志"
        )
    
    service = OperationLogService(db)
    deleted_count = service.cleanup_old_logs(days)
    
    return {
        "success": True,
        "message": f"已清理 {deleted_count} 条日志",
        "data": {"deleted_count": deleted_count}
    }
