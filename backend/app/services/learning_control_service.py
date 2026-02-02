"""
学习控制服务 v4.0

提供自学习系统的完整控制能力：
1. 暂停/恢复学习
2. 停止学习
3. 调整学习方向
4. 检查点管理
5. 学习进度监控
6. 学习过程可视化数据

架构说明:
┌─────────────────────────────────────────────────────────────────┐
│  治理系统 (Backend)                                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ LearningControlService                                     │ │
│  │  - pause_learning() / resume_learning()                    │ │
│  │  - stop_learning()                                         │ │
│  │  - redirect_learning()                                     │ │
│  │  - create_checkpoint() / rollback_to_checkpoint()          │ │
│  │  - get_learning_progress() / get_visualization_data()      │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │ 调用                               │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │ StateMachineService                                        │ │
│  │  - process_event(PAUSE_LEARNING/RESUME_LEARNING/...)       │ │
│  └─────────────────────────┬─────────────────────────────────┘ │
│                            │ 触发                               │
│  ┌─────────────────────────▼─────────────────────────────────┐ │
│  │ NotificationService                                        │ │
│  │  - WebSocket: 实时状态推送                                  │ │
│  │  - 学习进度更新                                             │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                             │
                             │ 事件回调 / WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  自学习系统 (self_learning)                                      │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ GovernanceInterface                                        │ │
│  │  - 接收治理干预                                             │ │
│  │  - 响应暂停/恢复/停止/方向调整                              │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID
from dataclasses import dataclass, field
import logging
import json

from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models.state import SystemState, StateTransition
from ..models.learning_unit import LearningUnit
from ..core.enums import (
    NLGSMState, 
    EventType, 
    Decision,
    NotificationType,
    LearningUnitStatus,
)
from ..core.exceptions import BusinessError, NotFoundError
from .state_machine_service import StateMachineService, Event, TransitionResult

if TYPE_CHECKING:
    from .notification_service import NotificationService

logger = logging.getLogger(__name__)


# ============================================================
# 数据类型定义
# ============================================================

@dataclass
class LearningSession:
    """学习会话信息"""
    session_id: str
    started_at: datetime
    state: str
    goal: Optional[str] = None
    scope: Optional[Dict[str, Any]] = None
    
    # 进度信息
    total_steps: int = 0
    completed_steps: int = 0
    current_depth: int = 0
    
    # 检查点
    checkpoints: List[str] = field(default_factory=list)
    latest_checkpoint_id: Optional[str] = None
    
    # 方向调整历史
    direction_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # 暂停信息
    is_paused: bool = False
    paused_at: Optional[datetime] = None
    pause_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "state": self.state,
            "goal": self.goal,
            "scope": self.scope,
            "progress": {
                "total_steps": self.total_steps,
                "completed_steps": self.completed_steps,
                "current_depth": self.current_depth,
                "progress_percent": (
                    self.completed_steps / self.total_steps * 100 
                    if self.total_steps > 0 else 0
                ),
            },
            "checkpoints": self.checkpoints,
            "latest_checkpoint_id": self.latest_checkpoint_id,
            "direction_changes": self.direction_changes,
            "is_paused": self.is_paused,
            "paused_at": self.paused_at.isoformat() if self.paused_at else None,
            "pause_reason": self.pause_reason,
        }


@dataclass
class Checkpoint:
    """检查点信息"""
    checkpoint_id: str
    session_id: str
    created_at: datetime
    reason: str
    
    # 状态快照
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # 学习进度快照
    progress_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "reason": self.reason,
            "state_snapshot": self.state_snapshot,
            "progress_snapshot": self.progress_snapshot,
            "metadata": self.metadata,
        }


@dataclass
class LearningVisualizationData:
    """学习过程可视化数据"""
    session_id: str
    
    # 状态流转图数据
    state_flow: List[Dict[str, Any]] = field(default_factory=list)
    
    # 学习树（探索路径）
    exploration_tree: Dict[str, Any] = field(default_factory=dict)
    
    # 时间线事件
    timeline_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 进度曲线数据
    progress_curve: List[Dict[str, Any]] = field(default_factory=list)
    
    # 检查点标记
    checkpoint_markers: List[Dict[str, Any]] = field(default_factory=list)
    
    # 方向调整标记
    direction_change_markers: List[Dict[str, Any]] = field(default_factory=list)
    
    # 统计摘要
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "state_flow": self.state_flow,
            "exploration_tree": self.exploration_tree,
            "timeline_events": self.timeline_events,
            "progress_curve": self.progress_curve,
            "checkpoint_markers": self.checkpoint_markers,
            "direction_change_markers": self.direction_change_markers,
            "statistics": self.statistics,
        }


# ============================================================
# 学习控制服务
# ============================================================

class LearningControlService:
    """
    学习控制服务
    
    提供对自学习系统的完整控制能力。
    """
    
    def __init__(
        self,
        db: Session,
        state_machine: Optional[StateMachineService] = None,
        notification_service: Optional["NotificationService"] = None,
    ):
        self.db = db
        self._state_machine = state_machine
        self._notification_service = notification_service
        
        # 当前学习会话（内存缓存）
        self._current_session: Optional[LearningSession] = None
        
        # 检查点存储（实际应持久化到数据库）
        self._checkpoints: Dict[str, Checkpoint] = {}
        
        # 会话历史
        self._session_history: List[LearningSession] = []
    
    @property
    def state_machine(self) -> StateMachineService:
        """延迟加载状态机服务"""
        if self._state_machine is None:
            self._state_machine = StateMachineService(self.db)
        return self._state_machine
    
    @property
    def notification_service(self) -> Optional["NotificationService"]:
        """获取通知服务"""
        if self._notification_service is None:
            try:
                from .notification_service import NotificationService
                self._notification_service = NotificationService(self.db)
            except ImportError:
                pass
        return self._notification_service
    
    # ==================== 学习会话管理 ====================
    
    def start_learning_session(
        self,
        goal: str,
        scope: Optional[Dict[str, Any]] = None,
        actor: str = "system",
    ) -> Dict[str, Any]:
        """
        启动学习会话
        
        Args:
            goal: 学习目标
            scope: 学习范围
            actor: 发起人
        
        Returns:
            会话信息
        """
        # 检查当前状态
        current_state = self.state_machine.get_current_state()
        if current_state not in [NLGSMState.FROZEN]:
            raise BusinessError(
                f"Cannot start learning in state {current_state.value}. Must be FROZEN.",
                code="INVALID_STATE"
            )
        
        # 创建会话
        session_id = f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        self._current_session = LearningSession(
            session_id=session_id,
            started_at=datetime.utcnow(),
            state=NLGSMState.LEARNING.value,
            goal=goal,
            scope=scope,
        )
        
        # 触发状态转换
        event = Event(
            event_type=EventType.AUDIT_SIGNAL,
            source=actor,
            metadata={
                "session_id": session_id,
                "goal": goal,
                "scope": scope,
            }
        )
        
        result = self.state_machine.process_event(event)
        
        if result.success:
            # 创建初始检查点
            self._create_checkpoint(
                session_id=session_id,
                reason="session_start",
                metadata={"goal": goal, "scope": scope}
            )
            
            logger.info(f"Learning session started: {session_id}")
            
            # 发送通知
            self._send_notification(
                NotificationType.STATE_CHANGED,
                {
                    "action": "learning_started",
                    "session_id": session_id,
                    "goal": goal,
                }
            )
        
        return {
            "success": result.success,
            "session": self._current_session.to_dict() if self._current_session else None,
            "transition": {
                "from_state": result.from_state.value,
                "to_state": result.to_state.value,
            }
        }
    
    def get_current_session(self) -> Optional[Dict[str, Any]]:
        """获取当前学习会话"""
        if self._current_session:
            return self._current_session.to_dict()
        return None
    
    # ==================== 学习控制 ====================
    
    def pause_learning(
        self,
        reason: str,
        actor: str = "system",
    ) -> Dict[str, Any]:
        """
        暂停学习
        
        Args:
            reason: 暂停原因
            actor: 操作人
        
        Returns:
            操作结果
        """
        current_state = self.state_machine.get_current_state()
        
        if current_state != NLGSMState.LEARNING:
            raise BusinessError(
                f"Cannot pause in state {current_state.value}. Must be LEARNING.",
                code="INVALID_STATE"
            )
        
        # 触发暂停事件
        event = Event(
            event_type=EventType.PAUSE_LEARNING,
            source=actor,
            metadata={
                "reason": reason,
                "session_id": self._current_session.session_id if self._current_session else None,
            }
        )
        
        result = self.state_machine.process_event(event)
        
        if result.success and self._current_session:
            self._current_session.is_paused = True
            self._current_session.paused_at = datetime.utcnow()
            self._current_session.pause_reason = reason
            self._current_session.state = NLGSMState.PAUSED.value
            
            # 创建暂停检查点
            checkpoint = self._create_checkpoint(
                session_id=self._current_session.session_id,
                reason=f"pause: {reason}",
                metadata={"pause_reason": reason, "actor": actor}
            )
            
            logger.info(f"Learning paused: {reason}")
            
            # 发送通知
            self._send_notification(
                NotificationType.LEARNING_PAUSED,
                {
                    "session_id": self._current_session.session_id,
                    "reason": reason,
                    "checkpoint_id": checkpoint.checkpoint_id,
                }
            )
        
        return {
            "success": result.success,
            "from_state": result.from_state.value,
            "to_state": result.to_state.value,
            "reason": reason,
            "checkpoint_id": checkpoint.checkpoint_id if result.success else None,
        }
    
    def resume_learning(
        self,
        actor: str = "system",
    ) -> Dict[str, Any]:
        """
        恢复学习
        
        Args:
            actor: 操作人
        
        Returns:
            操作结果
        """
        current_state = self.state_machine.get_current_state()
        
        if current_state != NLGSMState.PAUSED:
            raise BusinessError(
                f"Cannot resume in state {current_state.value}. Must be PAUSED.",
                code="INVALID_STATE"
            )
        
        # 触发恢复事件
        event = Event(
            event_type=EventType.RESUME_LEARNING,
            source=actor,
            metadata={
                "session_id": self._current_session.session_id if self._current_session else None,
            }
        )
        
        result = self.state_machine.process_event(event)
        
        if result.success and self._current_session:
            pause_duration = None
            if self._current_session.paused_at:
                pause_duration = (datetime.utcnow() - self._current_session.paused_at).total_seconds()
            
            self._current_session.is_paused = False
            self._current_session.paused_at = None
            self._current_session.pause_reason = None
            self._current_session.state = NLGSMState.LEARNING.value
            
            logger.info(f"Learning resumed after {pause_duration}s")
            
            # 发送通知
            self._send_notification(
                NotificationType.LEARNING_RESUMED,
                {
                    "session_id": self._current_session.session_id,
                    "pause_duration_seconds": pause_duration,
                }
            )
        
        return {
            "success": result.success,
            "from_state": result.from_state.value,
            "to_state": result.to_state.value,
        }
    
    def stop_learning(
        self,
        reason: str,
        actor: str = "system",
        save_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        停止学习
        
        Args:
            reason: 停止原因
            actor: 操作人
            save_progress: 是否保存进度
        
        Returns:
            操作结果
        """
        current_state = self.state_machine.get_current_state()
        
        if current_state not in [NLGSMState.LEARNING, NLGSMState.PAUSED]:
            raise BusinessError(
                f"Cannot stop in state {current_state.value}. Must be LEARNING or PAUSED.",
                code="INVALID_STATE"
            )
        
        # 保存进度
        final_checkpoint = None
        if save_progress and self._current_session:
            final_checkpoint = self._create_checkpoint(
                session_id=self._current_session.session_id,
                reason=f"stop: {reason}",
                metadata={"stop_reason": reason, "actor": actor, "final": True}
            )
        
        # 触发停止事件
        event = Event(
            event_type=EventType.STOP_LEARNING,
            source=actor,
            metadata={
                "reason": reason,
                "session_id": self._current_session.session_id if self._current_session else None,
                "final_checkpoint_id": final_checkpoint.checkpoint_id if final_checkpoint else None,
            }
        )
        
        result = self.state_machine.process_event(event)
        
        if result.success:
            # 保存会话到历史
            if self._current_session:
                self._session_history.append(self._current_session)
            
            session_id = self._current_session.session_id if self._current_session else None
            
            # 清理当前会话
            self._current_session = None
            
            logger.info(f"Learning stopped: {reason}")
            
            # 发送通知
            self._send_notification(
                NotificationType.LEARNING_STOPPED,
                {
                    "session_id": session_id,
                    "reason": reason,
                    "final_checkpoint_id": final_checkpoint.checkpoint_id if final_checkpoint else None,
                }
            )
        
        return {
            "success": result.success,
            "from_state": result.from_state.value,
            "to_state": result.to_state.value,
            "reason": reason,
            "final_checkpoint_id": final_checkpoint.checkpoint_id if final_checkpoint else None,
        }
    
    def redirect_learning(
        self,
        new_direction: str,
        reason: str,
        actor: str = "system",
        new_scope: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        调整学习方向
        
        Args:
            new_direction: 新的学习方向/目标
            reason: 调整原因
            actor: 操作人
            new_scope: 新的学习范围（可选）
        
        Returns:
            操作结果
        """
        current_state = self.state_machine.get_current_state()
        
        if current_state not in [NLGSMState.LEARNING, NLGSMState.PAUSED]:
            raise BusinessError(
                f"Cannot redirect in state {current_state.value}. Must be LEARNING or PAUSED.",
                code="INVALID_STATE"
            )
        
        old_direction = self._current_session.goal if self._current_session else None
        
        # 创建调整前检查点
        pre_redirect_checkpoint = None
        if self._current_session:
            pre_redirect_checkpoint = self._create_checkpoint(
                session_id=self._current_session.session_id,
                reason=f"pre_redirect: {reason}",
                metadata={
                    "old_direction": old_direction,
                    "new_direction": new_direction,
                }
            )
        
        # 触发方向调整事件
        event = Event(
            event_type=EventType.REDIRECT_LEARNING,
            source=actor,
            metadata={
                "new_direction": new_direction,
                "reason": reason,
                "new_scope": new_scope,
                "session_id": self._current_session.session_id if self._current_session else None,
            }
        )
        
        result = self.state_machine.process_event(event)
        
        if result.success and self._current_session:
            # 记录方向变更
            self._current_session.direction_changes.append({
                "timestamp": datetime.utcnow().isoformat(),
                "old_direction": old_direction,
                "new_direction": new_direction,
                "reason": reason,
                "actor": actor,
                "checkpoint_id": pre_redirect_checkpoint.checkpoint_id if pre_redirect_checkpoint else None,
            })
            
            # 更新目标
            self._current_session.goal = new_direction
            if new_scope:
                self._current_session.scope = new_scope
            
            logger.info(f"Learning redirected: {old_direction} -> {new_direction}")
            
            # 发送通知
            self._send_notification(
                NotificationType.LEARNING_REDIRECTED,
                {
                    "session_id": self._current_session.session_id,
                    "old_direction": old_direction,
                    "new_direction": new_direction,
                    "reason": reason,
                }
            )
        
        return {
            "success": result.success,
            "from_state": result.from_state.value,
            "to_state": result.to_state.value,
            "old_direction": old_direction,
            "new_direction": new_direction,
            "reason": reason,
            "checkpoint_id": pre_redirect_checkpoint.checkpoint_id if pre_redirect_checkpoint else None,
        }
    
    # ==================== 检查点管理 ====================
    
    def create_checkpoint(
        self,
        reason: str = "manual",
        actor: str = "system",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        手动创建检查点
        
        Args:
            reason: 创建原因
            actor: 操作人
            metadata: 额外元数据
        
        Returns:
            检查点信息
        """
        if not self._current_session:
            raise BusinessError("No active learning session", code="NO_SESSION")
        
        checkpoint = self._create_checkpoint(
            session_id=self._current_session.session_id,
            reason=reason,
            metadata={
                **(metadata or {}),
                "actor": actor,
                "manual": True,
            }
        )
        
        # 发送通知
        self._send_notification(
            NotificationType.CHECKPOINT_CREATED,
            {
                "session_id": self._current_session.session_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "reason": reason,
            }
        )
        
        return checkpoint.to_dict()
    
    def rollback_to_checkpoint(
        self,
        checkpoint_id: str,
        reason: str,
        actor: str = "system",
    ) -> Dict[str, Any]:
        """
        回滚到指定检查点
        
        Args:
            checkpoint_id: 检查点 ID
            reason: 回滚原因
            actor: 操作人
        
        Returns:
            操作结果
        """
        checkpoint = self._checkpoints.get(checkpoint_id)
        if not checkpoint:
            raise NotFoundError(f"Checkpoint not found: {checkpoint_id}")
        
        current_state = self.state_machine.get_current_state()
        
        if current_state not in [NLGSMState.LEARNING, NLGSMState.PAUSED]:
            raise BusinessError(
                f"Cannot rollback in state {current_state.value}",
                code="INVALID_STATE"
            )
        
        # 创建回滚前检查点
        pre_rollback_checkpoint = None
        if self._current_session:
            pre_rollback_checkpoint = self._create_checkpoint(
                session_id=self._current_session.session_id,
                reason=f"pre_rollback: {reason}",
                metadata={"target_checkpoint": checkpoint_id}
            )
        
        # 触发回滚事件
        event = Event(
            event_type=EventType.ROLLBACK_TO_CHECKPOINT,
            source=actor,
            metadata={
                "checkpoint_id": checkpoint_id,
                "reason": reason,
            }
        )
        
        result = self.state_machine.process_event(event)
        
        if result.success:
            # 恢复检查点状态
            self._restore_checkpoint_state(checkpoint)
            
            logger.info(f"Rolled back to checkpoint: {checkpoint_id}")
        
        return {
            "success": result.success,
            "target_checkpoint_id": checkpoint_id,
            "pre_rollback_checkpoint_id": pre_rollback_checkpoint.checkpoint_id if pre_rollback_checkpoint else None,
            "reason": reason,
        }
    
    def get_checkpoints(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取检查点列表"""
        checkpoints = list(self._checkpoints.values())
        
        if session_id:
            checkpoints = [c for c in checkpoints if c.session_id == session_id]
        
        # 按时间倒序
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        
        return [c.to_dict() for c in checkpoints[:limit]]
    
    def _create_checkpoint(
        self,
        session_id: str,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """内部方法：创建检查点"""
        checkpoint_id = f"ckpt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # 获取当前状态快照
        state_snapshot = self.state_machine.get_current_state_info()
        
        # 获取进度快照
        progress_snapshot = {}
        if self._current_session:
            progress_snapshot = {
                "completed_steps": self._current_session.completed_steps,
                "total_steps": self._current_session.total_steps,
                "current_depth": self._current_session.current_depth,
                "goal": self._current_session.goal,
            }
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            created_at=datetime.utcnow(),
            reason=reason,
            state_snapshot=state_snapshot,
            progress_snapshot=progress_snapshot,
            metadata=metadata or {},
        )
        
        self._checkpoints[checkpoint_id] = checkpoint
        
        # 更新会话的检查点列表
        if self._current_session and self._current_session.session_id == session_id:
            self._current_session.checkpoints.append(checkpoint_id)
            self._current_session.latest_checkpoint_id = checkpoint_id
        
        logger.debug(f"Checkpoint created: {checkpoint_id}")
        
        return checkpoint
    
    def _restore_checkpoint_state(self, checkpoint: Checkpoint):
        """内部方法：恢复检查点状态"""
        if self._current_session and checkpoint.progress_snapshot:
            # 恢复进度
            self._current_session.completed_steps = checkpoint.progress_snapshot.get(
                "completed_steps", self._current_session.completed_steps
            )
            self._current_session.total_steps = checkpoint.progress_snapshot.get(
                "total_steps", self._current_session.total_steps
            )
            self._current_session.current_depth = checkpoint.progress_snapshot.get(
                "current_depth", self._current_session.current_depth
            )
            self._current_session.goal = checkpoint.progress_snapshot.get(
                "goal", self._current_session.goal
            )
    
    # ==================== 进度监控 ====================
    
    def update_progress(
        self,
        completed_steps: Optional[int] = None,
        total_steps: Optional[int] = None,
        current_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        更新学习进度（由自学习系统调用）
        
        Args:
            completed_steps: 已完成步数
            total_steps: 总步数
            current_depth: 当前探索深度
        
        Returns:
            更新后的进度信息
        """
        if not self._current_session:
            raise BusinessError("No active learning session", code="NO_SESSION")
        
        if completed_steps is not None:
            self._current_session.completed_steps = completed_steps
        if total_steps is not None:
            self._current_session.total_steps = total_steps
        if current_depth is not None:
            self._current_session.current_depth = current_depth
        
        progress = {
            "session_id": self._current_session.session_id,
            "completed_steps": self._current_session.completed_steps,
            "total_steps": self._current_session.total_steps,
            "current_depth": self._current_session.current_depth,
            "progress_percent": (
                self._current_session.completed_steps / self._current_session.total_steps * 100
                if self._current_session.total_steps > 0 else 0
            ),
        }
        
        # 通过 WebSocket 推送进度更新
        if self.notification_service:
            self.notification_service.broadcast_message({
                "type": "learning_progress",
                "data": progress,
            })
        
        return progress
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """获取当前学习进度"""
        if not self._current_session:
            return {
                "active": False,
                "message": "No active learning session",
            }
        
        return {
            "active": True,
            "session_id": self._current_session.session_id,
            "state": self._current_session.state,
            "is_paused": self._current_session.is_paused,
            "goal": self._current_session.goal,
            "progress": {
                "completed_steps": self._current_session.completed_steps,
                "total_steps": self._current_session.total_steps,
                "current_depth": self._current_session.current_depth,
                "progress_percent": (
                    self._current_session.completed_steps / self._current_session.total_steps * 100
                    if self._current_session.total_steps > 0 else 0
                ),
            },
            "checkpoints_count": len(self._current_session.checkpoints),
            "direction_changes_count": len(self._current_session.direction_changes),
        }
    
    # ==================== 可视化数据 ====================
    
    def get_visualization_data(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        获取学习过程可视化数据
        
        Args:
            session_id: 会话 ID（可选，默认当前会话）
        
        Returns:
            可视化数据
        """
        target_session = None
        
        if session_id:
            # 查找指定会话
            if self._current_session and self._current_session.session_id == session_id:
                target_session = self._current_session
            else:
                for s in self._session_history:
                    if s.session_id == session_id:
                        target_session = s
                        break
        else:
            target_session = self._current_session
        
        if not target_session:
            return {
                "error": "Session not found",
                "session_id": session_id,
            }
        
        # 构建可视化数据
        viz_data = LearningVisualizationData(session_id=target_session.session_id)
        
        # 1. 状态流转图数据
        viz_data.state_flow = self._build_state_flow(target_session)
        
        # 2. 时间线事件
        viz_data.timeline_events = self._build_timeline_events(target_session)
        
        # 3. 进度曲线
        viz_data.progress_curve = self._build_progress_curve(target_session)
        
        # 4. 检查点标记
        viz_data.checkpoint_markers = self._build_checkpoint_markers(target_session)
        
        # 5. 方向调整标记
        viz_data.direction_change_markers = target_session.direction_changes
        
        # 6. 统计摘要
        viz_data.statistics = self._build_statistics(target_session)
        
        return viz_data.to_dict()
    
    def _build_state_flow(self, session: LearningSession) -> List[Dict[str, Any]]:
        """构建状态流转图数据"""
        # 从数据库获取状态转换历史
        transitions, _ = self.state_machine.get_transition_history(limit=100)
        
        flow = []
        for t in transitions:
            flow.append({
                "from_state": t.from_state,
                "to_state": t.to_state,
                "event": t.trigger_event,
                "timestamp": t.created_at.isoformat() if t.created_at else None,
                "success": t.success,
            })
        
        return flow
    
    def _build_timeline_events(self, session: LearningSession) -> List[Dict[str, Any]]:
        """构建时间线事件"""
        events = []
        
        # 会话开始
        events.append({
            "type": "session_start",
            "timestamp": session.started_at.isoformat(),
            "description": f"Learning started: {session.goal}",
        })
        
        # 检查点
        for ckpt_id in session.checkpoints:
            ckpt = self._checkpoints.get(ckpt_id)
            if ckpt:
                events.append({
                    "type": "checkpoint",
                    "timestamp": ckpt.created_at.isoformat(),
                    "description": f"Checkpoint: {ckpt.reason}",
                    "checkpoint_id": ckpt_id,
                })
        
        # 方向调整
        for change in session.direction_changes:
            events.append({
                "type": "direction_change",
                "timestamp": change["timestamp"],
                "description": f"Direction changed: {change['old_direction']} -> {change['new_direction']}",
                "reason": change["reason"],
            })
        
        # 暂停
        if session.paused_at:
            events.append({
                "type": "pause",
                "timestamp": session.paused_at.isoformat(),
                "description": f"Learning paused: {session.pause_reason}",
            })
        
        # 按时间排序
        events.sort(key=lambda e: e["timestamp"])
        
        return events
    
    def _build_progress_curve(self, session: LearningSession) -> List[Dict[str, Any]]:
        """构建进度曲线数据"""
        # 这里返回模拟数据，实际应从持久化存储读取
        # 或通过自学习系统的回调累积
        return [
            {
                "timestamp": session.started_at.isoformat(),
                "progress_percent": 0,
                "depth": 0,
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "progress_percent": (
                    session.completed_steps / session.total_steps * 100
                    if session.total_steps > 0 else 0
                ),
                "depth": session.current_depth,
            },
        ]
    
    def _build_checkpoint_markers(self, session: LearningSession) -> List[Dict[str, Any]]:
        """构建检查点标记"""
        markers = []
        for ckpt_id in session.checkpoints:
            ckpt = self._checkpoints.get(ckpt_id)
            if ckpt:
                markers.append({
                    "checkpoint_id": ckpt_id,
                    "timestamp": ckpt.created_at.isoformat(),
                    "reason": ckpt.reason,
                    "progress": ckpt.progress_snapshot,
                })
        return markers
    
    def _build_statistics(self, session: LearningSession) -> Dict[str, Any]:
        """构建统计摘要"""
        duration = None
        if session.started_at:
            end_time = datetime.utcnow()
            duration = (end_time - session.started_at).total_seconds()
        
        return {
            "duration_seconds": duration,
            "total_steps": session.total_steps,
            "completed_steps": session.completed_steps,
            "progress_percent": (
                session.completed_steps / session.total_steps * 100
                if session.total_steps > 0 else 0
            ),
            "checkpoints_count": len(session.checkpoints),
            "direction_changes_count": len(session.direction_changes),
            "current_depth": session.current_depth,
            "is_paused": session.is_paused,
        }
    
    # ==================== 历史查询 ====================
    
    def get_session_history(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """获取会话历史"""
        sessions = self._session_history[-limit:]
        return [s.to_dict() for s in reversed(sessions)]
    
    # ==================== 通知辅助 ====================
    
    def _send_notification(
        self,
        notification_type: NotificationType,
        data: Dict[str, Any],
    ):
        """发送通知"""
        if self.notification_service:
            try:
                self.notification_service.broadcast_message({
                    "type": notification_type.value,
                    "data": data,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
