"""
状态机 API 端点

所有业务逻辑通过服务层实现
"""
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from ....db.session import get_db
from ....services.state_machine_service import StateMachineService, Event
from ....middleware.auth import get_current_active_user, require_permissions, require_roles
from ....models.user import User
from ....schemas.state_machine import (
    TriggerEventRequest,
    ForceStateRequest,
    StateResponse,
    TransitionResponse,
    TransitionHistoryItem,
    TransitionHistoryResponse,
)
from ....core.enums import NLGSMState

router = APIRouter()


@router.get("/current", response_model=StateResponse)
async def get_current_state(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取当前系统状态
    """
    state_service = StateMachineService(db)
    
    state_info = state_service.get_current_state_info()
    available_transitions = state_service.get_available_transitions()
    
    return StateResponse(
        state=state_info["state"],
        entered_at=state_info.get("entered_at"),
        trigger_event=state_info.get("trigger_event"),
        trigger_source=state_info.get("trigger_source"),
        iteration_count=state_info.get("iteration_count", 0),
        metadata=state_info.get("metadata", {}),
        available_transitions=available_transitions,
    )


@router.post("/trigger", response_model=TransitionResponse)
async def trigger_event(
    event_data: TriggerEventRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permissions("state_machine.trigger")),
):
    """
    触发状态转换事件
    """
    state_service = StateMachineService(db)
    
    # 构建事件
    event = Event(
        event_type=event_data.event_type,
        source=current_user.username,
        metadata=event_data.metadata or {},
        learning_unit_id=event_data.learning_unit_id,
        artifact_id=event_data.artifact_id,
    )
    
    # 处理事件
    result = state_service.process_event(event)
    
    return TransitionResponse(
        success=result.success,
        from_state=result.from_state.value,
        to_state=result.to_state.value,
        decision=result.decision.value,
        reason=result.reason,
        actions_executed=result.actions_executed,
        duration_ms=result.duration_ms,
        error=result.error,
    )


@router.post("/force", response_model=TransitionResponse)
async def force_state(
    force_data: ForceStateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("admin", "governance_committee")),
):
    """
    强制设置状态（仅限管理员）
    
    用于紧急恢复或手动干预
    """
    state_service = StateMachineService(db)
    
    result = state_service.force_state(
        target_state=force_data.target_state,
        reason=force_data.reason,
        source=current_user.username,
    )
    
    return TransitionResponse(
        success=result.success,
        from_state=result.from_state.value,
        to_state=result.to_state.value,
        decision=result.decision.value,
        reason=result.reason,
        actions_executed=result.actions_executed,
        duration_ms=result.duration_ms,
        error=result.error,
    )


@router.get("/history", response_model=TransitionHistoryResponse)
async def get_transition_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    获取状态转换历史
    """
    state_service = StateMachineService(db)
    
    transitions, total = state_service.get_transition_history(
        limit=limit,
        offset=skip,
    )
    
    items = [
        TransitionHistoryItem(
            id=t.id,
            from_state=t.from_state,
            to_state=t.to_state,
            trigger_event=t.trigger_event,
            trigger_source=t.trigger_source,
            decision=t.decision,
            decision_reason=t.decision_reason,
            success=t.success,
            duration_ms=t.duration_ms,
            created_at=t.created_at,
        )
        for t in transitions
    ]
    
    return TransitionHistoryResponse(
        items=items,
        total=total,
        skip=skip,
        limit=limit,
    )


@router.get("/states", response_model=list)
async def get_all_states(
    current_user: User = Depends(get_current_active_user),
):
    """
    获取所有可能的状态
    """
    return [
        {
            "value": state.value,
            "name": state.name,
            "description": _get_state_description(state),
        }
        for state in NLGSMState
    ]


def _get_state_description(state: NLGSMState) -> str:
    """获取状态描述"""
    descriptions = {
        NLGSMState.LEARNING: "学习中 - 系统正在进行自主学习",
        NLGSMState.VALIDATION: "验证中 - 学习结果正在验证",
        NLGSMState.FROZEN: "已冻结 - 参数已锁定，等待审批",
        NLGSMState.RELEASE: "发布中 - 工件正在发布到生产",
        NLGSMState.ROLLBACK: "回滚中 - 正在恢复到之前的状态",
        NLGSMState.SAFE_HALT: "安全停机 - 系统已停止，需要人工干预",
        NLGSMState.DIAGNOSIS: "诊断中 - 正在分析问题原因",
        NLGSMState.RECOVERY_PLAN: "恢复计划 - 等待恢复方案审批",
    }
    return descriptions.get(state, "未知状态")

