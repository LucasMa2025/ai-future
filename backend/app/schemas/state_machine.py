"""
状态机相关 Schema
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ..core.enums import EventType, NLGSMState


class TriggerEventRequest(BaseModel):
    """触发事件请求"""
    event_type: EventType = Field(..., description="事件类型")
    metadata: Optional[Dict[str, Any]] = Field(default={}, description="附加数据")
    learning_unit_id: Optional[str] = None
    artifact_id: Optional[str] = None


class ForceStateRequest(BaseModel):
    """强制设置状态请求"""
    target_state: NLGSMState = Field(..., description="目标状态")
    reason: str = Field(..., min_length=10, description="操作原因")


class StateResponse(BaseModel):
    """状态响应"""
    state: str
    entered_at: Optional[datetime] = None
    trigger_event: Optional[str] = None
    trigger_source: Optional[str] = None
    iteration_count: int = 0
    metadata: Dict[str, Any] = {}
    available_transitions: List[Dict[str, str]] = []


class TransitionResponse(BaseModel):
    """状态转换响应"""
    success: bool
    from_state: str
    to_state: str
    decision: str
    reason: str
    actions_executed: List[str] = []
    duration_ms: int = 0
    error: Optional[str] = None


class TransitionHistoryItem(BaseModel):
    """状态转换历史项"""
    id: int
    from_state: str
    to_state: str
    trigger_event: str
    trigger_source: Optional[str] = None
    decision: str
    decision_reason: Optional[str] = None
    success: bool
    duration_ms: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class TransitionHistoryResponse(BaseModel):
    """状态转换历史响应"""
    items: List[TransitionHistoryItem]
    total: int
    skip: int
    limit: int

