"""
审批中心 API 端点

所有业务逻辑通过服务层实现
"""
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from ....db.session import get_db
from ....services.approval_service import ApprovalService
from ....middleware.auth import get_current_active_user, require_permissions
from ....models.user import User
from ....schemas.approval import (
    InitiateApprovalRequest,
    SubmitApprovalRequest,
    ApprovalResponse,
    ApprovalListResponse,
    ApprovalSubmitResponse,
    ApprovalStatisticsResponse,
)
from ....schemas.common import MessageResponse
from ....core.exceptions import NotFoundError, BusinessError, PermissionDeniedError

router = APIRouter()


@router.get("", response_model=ApprovalListResponse)
async def list_pending_approvals(
    target_type: Optional[str] = None,
    my_pending: bool = Query(True, description="只显示我可以审批的"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取待审批列表
    """
    approval_service = ApprovalService(db)
    
    if my_pending:
        approvals = approval_service.get_pending_approvals(
            user=current_user,
            target_type=target_type
        )
    else:
        approvals = approval_service.get_pending_approvals(
            target_type=target_type
        )
    
    return ApprovalListResponse(
        items=approvals,
        total=len(approvals)
    )


@router.post("", response_model=ApprovalResponse, status_code=status.HTTP_201_CREATED)
async def initiate_approval(
    approval_data: InitiateApprovalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("approvals.view")),
):
    """
    发起审批流程
    """
    approval_service = ApprovalService(db)
    
    approval = approval_service.initiate_approval(
        target_type=approval_data.target_type,
        target_id=approval_data.target_id,
        risk_level=approval_data.risk_level,
        title=approval_data.title,
        description=approval_data.description,
        metadata=approval_data.metadata,
        initiated_by=current_user.id,
    )
    
    return approval


@router.get("/{approval_id}", response_model=ApprovalResponse)
async def get_approval(
    approval_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取审批详情
    """
    approval_service = ApprovalService(db)
    approval = approval_service.get_approval_by_id(approval_id)
    
    if not approval:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Approval not found"
        )
    
    return approval


@router.post("/{approval_id}/submit", response_model=ApprovalSubmitResponse)
async def submit_approval(
    approval_id: str,
    submit_data: SubmitApprovalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    提交审批决策
    """
    approval_service = ApprovalService(db)
    
    try:
        result = approval_service.submit_approval(
            approval_id=approval_id,
            approver=current_user,
            decision=submit_data.decision,
            comments=submit_data.comments,
        )
        
        return ApprovalSubmitResponse(**result)
        
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
    except PermissionDeniedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=e.message
        )


@router.post("/{approval_id}/cancel", response_model=MessageResponse)
async def cancel_approval(
    approval_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("approvals.view")),
):
    """
    取消审批
    """
    approval_service = ApprovalService(db)
    
    try:
        approval_service.cancel_approval(
            approval_id=approval_id,
            cancelled_by=current_user.id,
            reason="Cancelled by user"
        )
        
        return MessageResponse(message="Approval cancelled successfully")
        
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


@router.get("/statistics/summary", response_model=ApprovalStatisticsResponse)
async def get_approval_statistics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取审批统计
    """
    approval_service = ApprovalService(db)
    stats = approval_service.get_approval_statistics()
    
    return ApprovalStatisticsResponse(**stats)

