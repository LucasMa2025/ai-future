"""
审计日志 API 端点

所有业务逻辑通过服务层实现
"""
from typing import Optional
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Query, HTTPException, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from ....db.session import get_db
from ....services.audit_service import AuditService
from ....middleware.auth import require_permissions
from ....models.user import User

router = APIRouter()


@router.get("/business-logs")
async def get_business_audit_logs(
    event_type: Optional[str] = None,
    event_category: Optional[str] = None,
    actor_id: Optional[UUID] = None,
    target_type: Optional[str] = None,
    target_id: Optional[str] = None,
    audit_level: Optional[str] = None,
    result: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.view")),
):
    """
    获取业务审计日志
    """
    audit_service = AuditService(db)
    
    logs, total = audit_service.get_audit_logs(
        event_type=event_type,
        event_category=event_category,
        actor_id=actor_id,
        target_type=target_type,
        target_id=target_id,
        audit_level=audit_level,
        result=result,
        start_date=start_date,
        end_date=end_date,
        skip=skip,
        limit=limit,
    )
    
    items = [
        {
            "entry_id": log.entry_id,
            "event_type": log.event_type,
            "event_category": log.event_category,
            "actor_id": str(log.actor_id) if log.actor_id else None,
            "actor_name": log.actor_name,
            "actor_ip": log.actor_ip,
            "action": log.action,
            "target_type": log.target_type,
            "target_id": log.target_id,
            "details": log.details,
            "result": log.result,
            "audit_level": log.audit_level,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]
    
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/business-logs/{entry_id}")
async def get_business_audit_log_detail(
    entry_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.view")),
):
    """
    获取审计日志详情
    """
    audit_service = AuditService(db)
    log = audit_service.get_audit_log_by_id(entry_id)
    
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit log not found"
        )
    
    return {
        "entry_id": log.entry_id,
        "event_type": log.event_type,
        "event_category": log.event_category,
        "actor_id": str(log.actor_id) if log.actor_id else None,
        "actor_name": log.actor_name,
        "actor_ip": log.actor_ip,
        "action": log.action,
        "target_type": log.target_type,
        "target_id": log.target_id,
        "target_name": log.target_name,
        "details": log.details,
        "request_data": log.request_data,
        "response_data": log.response_data,
        "result": log.result,
        "error_message": log.error_message,
        "audit_level": log.audit_level,
        "previous_hash": log.previous_hash,
        "entry_hash": log.entry_hash,
        "created_at": log.created_at.isoformat() if log.created_at else None,
    }


@router.get("/verify-chain")
async def verify_audit_chain(
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.view")),
):
    """
    验证审计日志链完整性
    """
    audit_service = AuditService(db)
    result = audit_service.verify_audit_chain(
        start_id=start_id,
        end_id=end_id,
    )
    
    return result


@router.get("/statistics")
async def get_audit_statistics(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.view")),
):
    """
    获取审计统计
    """
    audit_service = AuditService(db)
    stats = audit_service.get_audit_statistics(
        start_date=start_date,
        end_date=end_date,
    )
    
    return stats


@router.get("/export", response_class=PlainTextResponse)
async def export_audit_logs(
    start_date: datetime,
    end_date: datetime,
    format: str = Query("json", pattern="^(json|csv)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.export")),
):
    """
    导出审计日志
    """
    audit_service = AuditService(db)
    content = audit_service.export_audit_logs(
        start_date=start_date,
        end_date=end_date,
        format=format,
    )
    
    content_type = "application/json" if format == "json" else "text/csv"
    
    return PlainTextResponse(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename=audit_logs_{start_date.date()}_{end_date.date()}.{format}"
        }
    )


@router.get("/operation-logs")
async def get_operation_logs(
    user_id: Optional[UUID] = None,
    method: Optional[str] = None,
    path_pattern: Optional[str] = None,
    function_code: Optional[str] = None,
    is_success: Optional[bool] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.view")),
):
    """
    获取操作日志
    """
    audit_service = AuditService(db)
    
    logs, total = audit_service.get_operation_logs(
        user_id=user_id,
        method=method,
        path_pattern=path_pattern,
        function_code=function_code,
        is_success=is_success,
        start_date=start_date,
        end_date=end_date,
        skip=skip,
        limit=limit,
    )
    
    items = [
        {
            "id": log.id,
            "user_id": str(log.user_id) if log.user_id else None,
            "username": log.username,
            "request_id": log.request_id,
            "method": log.method,
            "path": log.path,
            "status_code": log.status_code,
            "response_time_ms": log.response_time_ms,
            "function_code": log.function_code,
            "is_success": log.is_success == 1,
            "ip_address": log.ip_address,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]
    
    return {
        "items": items,
        "total": total,
        "skip": skip,
        "limit": limit,
    }


@router.get("/user-activity/{user_id}")
async def get_user_activity(
    user_id: UUID,
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("audit.view")),
):
    """
    获取用户活动统计
    """
    audit_service = AuditService(db)
    activity = audit_service.get_user_activity(
        user_id=user_id,
        days=days,
    )
    
    return activity

