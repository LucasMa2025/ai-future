"""
审批相关 Schema
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ..core.enums import RiskLevel, ApprovalDecision


class InitiateApprovalRequest(BaseModel):
    """发起审批请求"""
    target_type: str = Field(..., description="目标类型: learning_unit, artifact, recovery_plan")
    target_id: str = Field(..., description="目标ID")
    risk_level: RiskLevel = Field(..., description="风险等级")
    title: str = Field(..., min_length=5, max_length=200)
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SubmitApprovalRequest(BaseModel):
    """提交审批请求"""
    decision: ApprovalDecision = Field(..., description="审批决策")
    comments: Optional[str] = Field(None, max_length=1000)


class ApproverInfo(BaseModel):
    """审批人信息"""
    user_id: str
    username: str
    decision: str
    comments: Optional[str] = None
    approved_at: str


class ApprovalResponse(BaseModel):
    """审批响应"""
    id: str
    target_type: str
    target_id: str
    risk_level: str
    title: str
    description: Optional[str] = None
    status: str
    required_approvers: int
    current_approvers: int
    required_roles: List[str] = []
    approver_list: List[ApproverInfo] = []
    initiated_by: Optional[str] = None
    deadline: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ApprovalListResponse(BaseModel):
    """审批列表响应"""
    items: List[ApprovalResponse]
    total: int


class ApprovalSubmitResponse(BaseModel):
    """审批提交响应"""
    status: str
    current_approvers: int
    required_approvers: int
    is_complete: bool
    approver_list: List[ApproverInfo] = []


class ApprovalStatisticsResponse(BaseModel):
    """审批统计响应"""
    total: int
    pending: int
    completed: int
    rejected: int
    by_risk_level: Dict[str, int] = {}

