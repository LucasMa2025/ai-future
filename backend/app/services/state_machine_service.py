"""
NLGSM 状态机服务

实现:
1. 状态机核心逻辑
2. 状态转换规则
3. 事件处理
"""
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import logging

from sqlalchemy.orm import Session

from ..models.state import SystemState, StateTransition
from ..core.enums import NLGSMState, EventType, Decision
from ..core.exceptions import StateMachineError, InvalidStateTransitionError

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """状态机事件"""
    event_type: EventType
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    learning_unit_id: Optional[str] = None
    artifact_id: Optional[str] = None


@dataclass
class TransitionResult:
    """状态转换结果"""
    success: bool
    from_state: NLGSMState
    to_state: NLGSMState
    decision: Decision
    reason: str
    actions_executed: List[str] = field(default_factory=list)
    duration_ms: int = 0
    error: Optional[str] = None


class StateMachineService:
    """
    NLGSM 状态机服务
    
    管理系统状态转换和事件处理
    """
    
    # 状态转换规则表
    TRANSITION_TABLE: Dict[tuple, tuple] = {
        # (当前状态, 事件) -> (目标状态, 需要的决策)
        
        # ==================== 原有转换规则 ====================
        (NLGSMState.FROZEN, EventType.AUDIT_SIGNAL): (NLGSMState.LEARNING, Decision.ALLOW),
        (NLGSMState.FROZEN, EventType.PERIODIC): (NLGSMState.LEARNING, Decision.ALLOW),
        
        (NLGSMState.LEARNING, EventType.REACH_STEPS): (NLGSMState.VALIDATION, Decision.ALLOW),
        (NLGSMState.LEARNING, EventType.ANOMALY): (NLGSMState.ROLLBACK, Decision.ROLLBACK),
        
        (NLGSMState.VALIDATION, EventType.PASS_VALIDATION): (NLGSMState.FROZEN, Decision.ALLOW),
        (NLGSMState.VALIDATION, EventType.FAIL_BUT_FIXABLE): (NLGSMState.LEARNING, Decision.ALLOW),
        (NLGSMState.VALIDATION, EventType.ANOMALY): (NLGSMState.ROLLBACK, Decision.ROLLBACK),
        
        (NLGSMState.FROZEN, EventType.HUMAN_APPROVE): (NLGSMState.RELEASE, Decision.ALLOW),
        
        (NLGSMState.RELEASE, EventType.RELEASE_COMPLETE): (NLGSMState.FROZEN, Decision.ALLOW),
        (NLGSMState.RELEASE, EventType.ANOMALY): (NLGSMState.ROLLBACK, Decision.ROLLBACK),
        
        (NLGSMState.ROLLBACK, EventType.RECOVER): (NLGSMState.FROZEN, Decision.ALLOW),
        (NLGSMState.ROLLBACK, EventType.ANOMALY): (NLGSMState.SAFE_HALT, Decision.HALT),
        
        (NLGSMState.SAFE_HALT, EventType.START_DIAGNOSIS): (NLGSMState.DIAGNOSIS, Decision.ALLOW),
        
        (NLGSMState.DIAGNOSIS, EventType.DIAGNOSIS_COMPLETE): (NLGSMState.RECOVERY_PLAN, Decision.ALLOW),
        
        (NLGSMState.RECOVERY_PLAN, EventType.PLAN_APPROVED): (NLGSMState.ROLLBACK, Decision.ALLOW),
        (NLGSMState.RECOVERY_PLAN, EventType.PLAN_RESTORE): (NLGSMState.FROZEN, Decision.ALLOW),
        (NLGSMState.RECOVERY_PLAN, EventType.PLAN_REJECTED): (NLGSMState.DIAGNOSIS, Decision.ALLOW),
        
        # ==================== 学习控制转换规则 (v4.0) ====================
        # 暂停学习: LEARNING -> PAUSED
        (NLGSMState.LEARNING, EventType.PAUSE_LEARNING): (NLGSMState.PAUSED, Decision.ALLOW),
        
        # 恢复学习: PAUSED -> LEARNING
        (NLGSMState.PAUSED, EventType.RESUME_LEARNING): (NLGSMState.LEARNING, Decision.ALLOW),
        
        # 停止学习: LEARNING/PAUSED -> FROZEN
        (NLGSMState.LEARNING, EventType.STOP_LEARNING): (NLGSMState.FROZEN, Decision.ALLOW),
        (NLGSMState.PAUSED, EventType.STOP_LEARNING): (NLGSMState.FROZEN, Decision.ALLOW),
        
        # 调整方向: LEARNING -> VALIDATION (触发重新验证)
        (NLGSMState.LEARNING, EventType.REDIRECT_LEARNING): (NLGSMState.VALIDATION, Decision.ALLOW),
        (NLGSMState.PAUSED, EventType.REDIRECT_LEARNING): (NLGSMState.VALIDATION, Decision.ALLOW),
        
        # 检查点请求: 不改变状态，但触发检查点创建
        # (通过 _execute_transition 中的特殊处理)
        
        # 回滚到检查点: LEARNING/PAUSED -> LEARNING (带检查点恢复)
        (NLGSMState.LEARNING, EventType.ROLLBACK_TO_CHECKPOINT): (NLGSMState.LEARNING, Decision.ALLOW),
        (NLGSMState.PAUSED, EventType.ROLLBACK_TO_CHECKPOINT): (NLGSMState.PAUSED, Decision.ALLOW),
        
        # PAUSED 态下的异常处理
        (NLGSMState.PAUSED, EventType.ANOMALY): (NLGSMState.ROLLBACK, Decision.ROLLBACK),
    }
    
    # 任意状态都可以触发的异常事件
    GLOBAL_ANOMALY_TRANSITIONS = {
        "critical": NLGSMState.SAFE_HALT,
        "high": NLGSMState.ROLLBACK,
        "medium": NLGSMState.DIAGNOSIS,
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_current_state(self) -> NLGSMState:
        """获取当前系统状态"""
        state_record = self.db.query(SystemState).filter(
            SystemState.is_current == True
        ).first()
        
        if not state_record:
            # 如果没有状态记录，初始化为 FROZEN
            return NLGSMState.FROZEN
        
        return NLGSMState(state_record.state)
    
    def get_current_state_info(self) -> Dict[str, Any]:
        """获取当前状态的详细信息"""
        state_record = self.db.query(SystemState).filter(
            SystemState.is_current == True
        ).first()
        
        if not state_record:
            return {
                "state": NLGSMState.FROZEN.value,
                "entered_at": None,
                "trigger_event": None,
                "iteration_count": 0,
            }
        
        return {
            "state": state_record.state,
            "entered_at": state_record.entered_at.isoformat() if state_record.entered_at else None,
            "trigger_event": state_record.trigger_event,
            "trigger_source": state_record.trigger_source,
            "iteration_count": state_record.iteration_count,
            "metadata": state_record.metadata,
        }
    
    def process_event(self, event: Event) -> TransitionResult:
        """
        处理事件并执行状态转换
        
        Args:
            event: 状态机事件
            
        Returns:
            转换结果
        """
        start_time = time.time()
        
        current_state = self.get_current_state()
        
        # 确定目标状态
        target_state, decision = self._determine_transition(current_state, event)
        
        if target_state is None:
            # 无效转换
            duration_ms = int((time.time() - start_time) * 1000)
            
            # 记录失败的转换尝试
            self._record_transition(
                from_state=current_state,
                to_state=current_state,
                event=event,
                decision=Decision.DENY,
                success=False,
                error="Invalid state transition",
                duration_ms=duration_ms
            )
            
            return TransitionResult(
                success=False,
                from_state=current_state,
                to_state=current_state,
                decision=Decision.DENY,
                reason=f"No valid transition from {current_state.value} with event {event.event_type.value}",
                duration_ms=duration_ms,
                error="Invalid state transition"
            )
        
        # 执行转换
        try:
            actions = self._execute_transition(current_state, target_state, event)
            
            # 更新状态
            self._update_state(target_state, event)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # 记录成功的转换
            self._record_transition(
                from_state=current_state,
                to_state=target_state,
                event=event,
                decision=decision,
                success=True,
                actions=actions,
                duration_ms=duration_ms
            )
            
            logger.info(
                f"State transition: {current_state.value} -> {target_state.value} "
                f"via {event.event_type.value}"
            )
            
            return TransitionResult(
                success=True,
                from_state=current_state,
                to_state=target_state,
                decision=decision,
                reason=f"Transition successful via {event.event_type.value}",
                actions_executed=actions,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            
            # 记录失败的转换
            self._record_transition(
                from_state=current_state,
                to_state=target_state,
                event=event,
                decision=decision,
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )
            
            logger.error(f"State transition failed: {e}")
            
            return TransitionResult(
                success=False,
                from_state=current_state,
                to_state=current_state,
                decision=decision,
                reason=str(e),
                duration_ms=duration_ms,
                error=str(e)
            )
    
    def get_available_transitions(self) -> List[Dict[str, Any]]:
        """获取当前状态可用的转换"""
        current_state = self.get_current_state()
        transitions = []
        
        for (state, event_type), (target_state, _) in self.TRANSITION_TABLE.items():
            if state == current_state:
                transitions.append({
                    "event_type": event_type.value,
                    "target_state": target_state.value,
                })
        
        return transitions
    
    def get_transition_history(
        self,
        limit: int = 50,
        offset: int = 0
    ) -> tuple[List[StateTransition], int]:
        """获取状态转换历史"""
        query = self.db.query(StateTransition)
        
        total = query.count()
        transitions = query.order_by(
            StateTransition.created_at.desc()
        ).offset(offset).limit(limit).all()
        
        return transitions, total
    
    def force_state(
        self,
        target_state: NLGSMState,
        reason: str,
        source: str
    ) -> TransitionResult:
        """
        强制设置状态（管理员操作）
        
        用于紧急恢复或手动干预
        """
        current_state = self.get_current_state()
        
        # 创建强制转换事件
        event = Event(
            event_type=EventType.AUDIT_SIGNAL,
            source=source,
            metadata={"forced": True, "reason": reason}
        )
        
        # 直接更新状态
        self._update_state(target_state, event)
        
        # 记录转换
        self._record_transition(
            from_state=current_state,
            to_state=target_state,
            event=event,
            decision=Decision.ALLOW,
            success=True,
            actions=["force_state"],
            duration_ms=0
        )
        
        logger.warning(
            f"Forced state change: {current_state.value} -> {target_state.value} "
            f"by {source}, reason: {reason}"
        )
        
        return TransitionResult(
            success=True,
            from_state=current_state,
            to_state=target_state,
            decision=Decision.ALLOW,
            reason=f"Forced: {reason}",
            actions_executed=["force_state"],
        )
    
    # ==================== 私有方法 ====================
    
    def _determine_transition(
        self,
        current_state: NLGSMState,
        event: Event
    ) -> tuple[Optional[NLGSMState], Decision]:
        """确定状态转换目标"""
        # 处理异常事件
        if event.event_type == EventType.ANOMALY:
            severity = event.metadata.get("severity", "low")
            if severity in self.GLOBAL_ANOMALY_TRANSITIONS:
                target = self.GLOBAL_ANOMALY_TRANSITIONS[severity]
                decision = Decision.HALT if severity == "critical" else Decision.ROLLBACK
                return target, decision
        
        # 查找常规转换
        key = (current_state, event.event_type)
        if key in self.TRANSITION_TABLE:
            return self.TRANSITION_TABLE[key]
        
        return None, Decision.DENY
    
    def _execute_transition(
        self,
        from_state: NLGSMState,
        to_state: NLGSMState,
        event: Event
    ) -> List[str]:
        """执行状态转换相关的动作"""
        actions = []
        
        # 根据转换类型执行不同动作
        if to_state == NLGSMState.LEARNING:
            actions.append("unfreeze_parameters")
            # 从暂停恢复时的特殊处理
            if from_state == NLGSMState.PAUSED:
                actions.append("resume_learning_session")
        
        elif to_state == NLGSMState.VALIDATION:
            actions.append("freeze_parameters")
            actions.append("start_validation")
            # 方向调整触发的验证
            if event.event_type == EventType.REDIRECT_LEARNING:
                actions.append("apply_new_direction")
                actions.append("create_redirect_checkpoint")
        
        elif to_state == NLGSMState.FROZEN:
            actions.append("freeze_parameters")
            if from_state == NLGSMState.VALIDATION:
                actions.append("create_snapshot")
            # 停止学习触发的冻结
            if event.event_type == EventType.STOP_LEARNING:
                actions.append("terminate_learning_session")
                actions.append("create_termination_snapshot")
        
        elif to_state == NLGSMState.RELEASE:
            actions.append("prepare_release")
        
        elif to_state == NLGSMState.ROLLBACK:
            actions.append("restore_snapshot")
            actions.append("send_notification")
        
        elif to_state == NLGSMState.SAFE_HALT:
            actions.append("freeze_all")
            actions.append("trigger_alert")
            actions.append("send_urgent_notification")
        
        elif to_state == NLGSMState.DIAGNOSIS:
            actions.append("collect_diagnostics")
        
        elif to_state == NLGSMState.RECOVERY_PLAN:
            actions.append("generate_recovery_plan")
        
        # ==================== 学习控制状态动作 (v4.0) ====================
        elif to_state == NLGSMState.PAUSED:
            actions.append("freeze_parameters")
            actions.append("pause_learning_session")
            actions.append("create_pause_checkpoint")
            actions.append("send_pause_notification")
        
        # 检查点回滚的特殊处理
        if event.event_type == EventType.ROLLBACK_TO_CHECKPOINT:
            checkpoint_id = event.metadata.get("checkpoint_id")
            if checkpoint_id:
                actions.append(f"rollback_to_checkpoint:{checkpoint_id}")
                actions.append("create_rollback_audit_log")
        
        # 检查点请求的特殊处理（不改变状态）
        if event.event_type == EventType.CHECKPOINT_REQUEST:
            actions.append("create_manual_checkpoint")
        
        return actions
    
    def _update_state(self, new_state: NLGSMState, event: Event):
        """更新系统状态"""
        # 取消当前状态
        current = self.db.query(SystemState).filter(
            SystemState.is_current == True
        ).first()
        
        if current:
            current.is_current = False
            # 计算持续时间
            if current.entered_at:
                duration = (datetime.utcnow() - current.entered_at).total_seconds()
                current.duration_seconds = int(duration)
        
        # 创建新状态
        new_state_record = SystemState(
            state=new_state.value,
            is_current=True,
            trigger_event=event.event_type.value,
            trigger_source=event.source,
            metadata=event.metadata,
            iteration_count=(current.iteration_count + 1) if current else 1,
        )
        
        self.db.add(new_state_record)
        self.db.commit()
    
    def _record_transition(
        self,
        from_state: NLGSMState,
        to_state: NLGSMState,
        event: Event,
        decision: Decision,
        success: bool,
        actions: List[str] = None,
        error: str = None,
        duration_ms: int = 0
    ):
        """记录状态转换"""
        transition = StateTransition(
            from_state=from_state.value,
            to_state=to_state.value,
            trigger_event=event.event_type.value,
            trigger_source=event.source,
            decision=decision.value,
            decision_reason=event.metadata.get("reason", ""),
            decision_evidence=event.metadata,
            actions_executed=actions or [],
            success=success,
            error_message=error,
            duration_ms=duration_ms,
        )
        
        self.db.add(transition)
        self.db.commit()

