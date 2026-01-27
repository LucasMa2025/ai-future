"""
治理服务

实现:
1. 学习干预（暂停、恢复、终止、回滚）
2. 检查点审查
3. 学习范围管理
4. 与自学习系统的集成
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID
from enum import Enum
import uuid
import logging

from sqlalchemy.orm import Session

from ..models.learning_unit import LearningSession, Checkpoint, LearningUnit
from ..models.user import User
from ..core.exceptions import NotFoundError, BusinessError, PermissionDeniedError


if TYPE_CHECKING:
    from .learning_unit_service import LearningUnitService
    from .notification_service import NotificationService


logger = logging.getLogger(__name__)


class InterventionType(str, Enum):
    """干预类型"""
    PAUSE = "pause"                    # 暂停学习
    RESUME = "resume"                  # 恢复学习
    TERMINATE = "terminate"            # 终止学习
    MODIFY_SCOPE = "modify_scope"      # 修改学习范围
    MODIFY_GOAL = "modify_goal"        # 修改学习目标
    ROLLBACK = "rollback"              # 回滚到检查点
    INJECT_CONSTRAINT = "inject_constraint"  # 注入约束
    REQUEST_CHECKPOINT = "request_checkpoint"  # 请求创建检查点


class InterventionPriority(str, Enum):
    """干预优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class GovernanceService:
    """
    治理服务
    
    管理自学习系统的干预和监控
    """
    
    def __init__(
        self,
        db: Session,
        lu_service: Optional["LearningUnitService"] = None,
        notification_service: Optional["NotificationService"] = None,
    ):
        self.db = db
        self._lu_service = lu_service
        self._notification_service = notification_service
        
        # 干预历史（内存）
        self._interventions: List[Dict[str, Any]] = []
        
        # 活跃的学习会话状态（内存缓存）
        self._session_states: Dict[str, Dict[str, Any]] = {}
    
    @property
    def lu_service(self) -> "LearningUnitService":
        """延迟加载 LearningUnitService"""
        if self._lu_service is None:
            from .learning_unit_service import LearningUnitService
            self._lu_service = LearningUnitService(self.db)
        return self._lu_service
    
    # ==================== 学习会话管理 ====================
    
    def create_learning_session(
        self,
        scope_id: str,
        goal: str,
        max_nl_level: str = "policy",
        allowed_levels: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> LearningSession:
        """
        创建学习会话
        
        Args:
            scope_id: 学习范围ID
            goal: 学习目标
            max_nl_level: 最大NL层级
            allowed_levels: 允许的NL层级列表
            metadata: 元数据
        
        Returns:
            创建的学习会话
        """
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        session = LearningSession(
            session_id=session_id,
            scope_id=scope_id,
            goal=goal,
            max_nl_level=max_nl_level,
            allowed_levels=allowed_levels or ["parameter", "memory", "optimizer", "policy"],
            status="active",
            metadata=metadata or {},
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        # 初始化会话状态
        self._session_states[session_id] = {
            "status": "active",
            "paused": False,
            "current_level": "parameter",
        }
        
        logger.info(f"Created learning session {session_id}")
        
        return session
    
    def get_session(self, session_id: str) -> Optional[LearningSession]:
        """获取学习会话"""
        return self.db.query(LearningSession).filter(
            LearningSession.session_id == session_id
        ).first()
    
    def get_session_or_404(self, session_id: str) -> LearningSession:
        """获取学习会话，不存在则抛出异常"""
        session = self.get_session(session_id)
        if not session:
            raise NotFoundError("LearningSession", session_id)
        return session
    
    def list_sessions(
        self,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[LearningSession], int]:
        """列出学习会话"""
        query = self.db.query(LearningSession)
        
        if status:
            query = query.filter(LearningSession.status == status)
        
        total = query.count()
        items = query.order_by(
            LearningSession.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    # ==================== 干预操作 ====================
    
    def issue_intervention(
        self,
        session_id: str,
        intervention_type: InterventionType,
        reason: str,
        issued_by: User,
        priority: InterventionPriority = InterventionPriority.NORMAL,
        target_checkpoint_id: Optional[str] = None,
        new_scope: Optional[Dict] = None,
        new_goal: Optional[str] = None,
        constraint: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        发起干预
        
        Args:
            session_id: 会话ID
            intervention_type: 干预类型
            reason: 干预原因
            issued_by: 发起人
            priority: 优先级
            target_checkpoint_id: 目标检查点（用于回滚）
            new_scope: 新的学习范围（用于修改范围）
            new_goal: 新的学习目标（用于修改目标）
            constraint: 要注入的约束
        
        Returns:
            干预结果
        """
        session = self.get_session_or_404(session_id)
        
        intervention_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        intervention = {
            "id": intervention_id,
            "session_id": session_id,
            "type": intervention_type.value,
            "reason": reason,
            "issued_by": str(issued_by.id),
            "issued_by_name": issued_by.username,
            "priority": priority.value,
            "timestamp": timestamp.isoformat(),
            "status": "pending",
            "result": None,
        }
        
        # 执行干预
        try:
            if intervention_type == InterventionType.PAUSE:
                result = self._execute_pause(session)
            
            elif intervention_type == InterventionType.RESUME:
                result = self._execute_resume(session)
            
            elif intervention_type == InterventionType.TERMINATE:
                result = self._execute_terminate(session, reason)
            
            elif intervention_type == InterventionType.MODIFY_SCOPE:
                if not new_scope:
                    raise BusinessError("new_scope required for MODIFY_SCOPE", code="MISSING_PARAM")
                result = self._execute_modify_scope(session, new_scope)
            
            elif intervention_type == InterventionType.MODIFY_GOAL:
                if not new_goal:
                    raise BusinessError("new_goal required for MODIFY_GOAL", code="MISSING_PARAM")
                result = self._execute_modify_goal(session, new_goal)
            
            elif intervention_type == InterventionType.ROLLBACK:
                if not target_checkpoint_id:
                    raise BusinessError("target_checkpoint_id required for ROLLBACK", code="MISSING_PARAM")
                result = self._execute_rollback(session, target_checkpoint_id)
            
            elif intervention_type == InterventionType.INJECT_CONSTRAINT:
                if not constraint:
                    raise BusinessError("constraint required for INJECT_CONSTRAINT", code="MISSING_PARAM")
                result = self._execute_inject_constraint(session, constraint)
            
            elif intervention_type == InterventionType.REQUEST_CHECKPOINT:
                result = self._execute_request_checkpoint(session)
            
            else:
                raise BusinessError(f"Unknown intervention type: {intervention_type}", code="UNKNOWN_TYPE")
            
            intervention["status"] = "completed"
            intervention["result"] = result
            
        except Exception as e:
            intervention["status"] = "failed"
            intervention["error"] = str(e)
            logger.error(f"Intervention {intervention_id} failed: {e}")
            raise
        
        # 记录干预
        self._interventions.append(intervention)
        
        # 更新会话的干预记录
        session.interventions = (session.interventions or []) + [intervention]
        self.db.commit()
        
        logger.info(f"Intervention {intervention_id} ({intervention_type.value}) completed for session {session_id}")
        
        return intervention
    
    def _execute_pause(self, session: LearningSession) -> Dict[str, Any]:
        """执行暂停"""
        if session.status != "active":
            raise BusinessError(f"Cannot pause session in status {session.status}", code="INVALID_STATUS")
        
        session.status = "paused"
        session.paused_at = datetime.utcnow()
        
        if session.session_id in self._session_states:
            self._session_states[session.session_id]["paused"] = True
        
        return {"action": "paused", "paused_at": session.paused_at.isoformat()}
    
    def _execute_resume(self, session: LearningSession) -> Dict[str, Any]:
        """执行恢复"""
        if session.status != "paused":
            raise BusinessError(f"Cannot resume session in status {session.status}", code="INVALID_STATUS")
        
        session.status = "active"
        
        if session.session_id in self._session_states:
            self._session_states[session.session_id]["paused"] = False
        
        return {"action": "resumed"}
    
    def _execute_terminate(self, session: LearningSession, reason: str) -> Dict[str, Any]:
        """执行终止"""
        session.status = "terminated"
        session.completed_at = datetime.utcnow()
        
        if session.session_id in self._session_states:
            self._session_states[session.session_id]["status"] = "terminated"
        
        return {"action": "terminated", "reason": reason}
    
    def _execute_modify_scope(self, session: LearningSession, new_scope: Dict) -> Dict[str, Any]:
        """执行修改范围"""
        old_scope = {
            "scope_id": session.scope_id,
            "max_nl_level": session.max_nl_level,
            "allowed_levels": session.allowed_levels,
        }
        
        if "scope_id" in new_scope:
            session.scope_id = new_scope["scope_id"]
        if "max_nl_level" in new_scope:
            session.max_nl_level = new_scope["max_nl_level"]
        if "allowed_levels" in new_scope:
            session.allowed_levels = new_scope["allowed_levels"]
        
        return {"action": "scope_modified", "old_scope": old_scope, "new_scope": new_scope}
    
    def _execute_modify_goal(self, session: LearningSession, new_goal: str) -> Dict[str, Any]:
        """执行修改目标"""
        old_goal = session.goal
        session.goal = new_goal
        
        return {"action": "goal_modified", "old_goal": old_goal, "new_goal": new_goal}
    
    def _execute_rollback(self, session: LearningSession, checkpoint_id: str) -> Dict[str, Any]:
        """执行回滚"""
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise NotFoundError("Checkpoint", checkpoint_id)
        
        if checkpoint.session_id != session.session_id:
            raise BusinessError("Checkpoint does not belong to this session", code="INVALID_CHECKPOINT")
        
        # 恢复检查点状态
        if checkpoint.state_snapshot:
            # 这里应该调用自学习系统的回滚接口
            pass
        
        return {
            "action": "rolled_back",
            "checkpoint_id": checkpoint_id,
            "checkpoint_nl_level": checkpoint.nl_level,
        }
    
    def _execute_inject_constraint(self, session: LearningSession, constraint: Dict) -> Dict[str, Any]:
        """执行注入约束"""
        # 这里应该调用自学习系统的约束注入接口
        return {"action": "constraint_injected", "constraint": constraint}
    
    def _execute_request_checkpoint(self, session: LearningSession) -> Dict[str, Any]:
        """执行请求检查点"""
        # 创建检查点
        checkpoint = self.create_checkpoint(
            session_id=session.session_id,
            nl_level=session.max_nl_level,
            knowledge_count=session.total_learning_units,
        )
        
        return {"action": "checkpoint_created", "checkpoint_id": checkpoint.checkpoint_id}
    
    # ==================== 检查点管理 ====================
    
    def create_checkpoint(
        self,
        session_id: str,
        nl_level: str,
        knowledge_count: int = 0,
        state_snapshot: Optional[Dict] = None,
    ) -> Checkpoint:
        """创建检查点"""
        checkpoint_id = f"ckpt_{uuid.uuid4().hex[:12]}"
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            session_id=session_id,
            nl_level=nl_level,
            knowledge_count=knowledge_count,
            state_snapshot=state_snapshot or {},
            review_status="pending",
        )
        
        self.db.add(checkpoint)
        
        # 更新会话的检查点计数
        session = self.get_session(session_id)
        if session:
            session.total_checkpoints = (session.total_checkpoints or 0) + 1
        
        self.db.commit()
        self.db.refresh(checkpoint)
        
        logger.info(f"Created checkpoint {checkpoint_id} for session {session_id}")
        
        return checkpoint
    
    def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """获取检查点"""
        return self.db.query(Checkpoint).filter(
            Checkpoint.checkpoint_id == checkpoint_id
        ).first()
    
    def list_checkpoints(
        self,
        session_id: Optional[str] = None,
        review_status: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[Checkpoint], int]:
        """列出检查点"""
        query = self.db.query(Checkpoint)
        
        if session_id:
            query = query.filter(Checkpoint.session_id == session_id)
        
        if review_status:
            query = query.filter(Checkpoint.review_status == review_status)
        
        total = query.count()
        items = query.order_by(
            Checkpoint.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def review_checkpoint(
        self,
        checkpoint_id: str,
        reviewer: User,
        decision: str,
        comments: Optional[str] = None,
        modified_scope: Optional[Dict] = None,
        modified_goal: Optional[str] = None,
    ) -> Checkpoint:
        """
        审查检查点
        
        Args:
            checkpoint_id: 检查点ID
            reviewer: 审查人
            decision: 决策 (continue/modify/pause/terminate)
            comments: 审查意见
            modified_scope: 修改后的范围
            modified_goal: 修改后的目标
        
        Returns:
            更新后的检查点
        """
        checkpoint = self.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise NotFoundError("Checkpoint", checkpoint_id)
        
        if checkpoint.review_status != "pending":
            raise BusinessError(
                f"Checkpoint already reviewed: {checkpoint.review_status}",
                code="ALREADY_REVIEWED"
            )
        
        checkpoint.review_status = "reviewed"
        checkpoint.reviewed_by = reviewer.id
        checkpoint.reviewed_at = datetime.utcnow()
        checkpoint.review_decision = decision
        checkpoint.review_comments = comments
        
        if modified_scope:
            checkpoint.modified_scope = modified_scope
        if modified_goal:
            checkpoint.modified_goal = modified_goal
        
        # 根据决策执行相应操作
        session = self.get_session(checkpoint.session_id)
        if session:
            if decision == "pause":
                self.issue_intervention(
                    session.session_id,
                    InterventionType.PAUSE,
                    f"Checkpoint review decision: pause. {comments or ''}",
                    reviewer,
                    InterventionPriority.HIGH,
                )
            
            elif decision == "terminate":
                self.issue_intervention(
                    session.session_id,
                    InterventionType.TERMINATE,
                    f"Checkpoint review decision: terminate. {comments or ''}",
                    reviewer,
                    InterventionPriority.CRITICAL,
                )
            
            elif decision == "modify":
                if modified_scope:
                    self.issue_intervention(
                        session.session_id,
                        InterventionType.MODIFY_SCOPE,
                        f"Checkpoint review: scope modified. {comments or ''}",
                        reviewer,
                        InterventionPriority.HIGH,
                        new_scope=modified_scope,
                    )
                if modified_goal:
                    self.issue_intervention(
                        session.session_id,
                        InterventionType.MODIFY_GOAL,
                        f"Checkpoint review: goal modified. {comments or ''}",
                        reviewer,
                        InterventionPriority.HIGH,
                        new_goal=modified_goal,
                    )
        
        self.db.commit()
        self.db.refresh(checkpoint)
        
        logger.info(f"Checkpoint {checkpoint_id} reviewed by {reviewer.username}: {decision}")
        
        return checkpoint
    
    # ==================== 查询接口 ====================
    
    def get_intervention_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取干预历史"""
        history = self._interventions
        
        if session_id:
            history = [i for i in history if i.get("session_id") == session_id]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        session = self.get_session_or_404(session_id)
        
        # 获取最近的检查点
        checkpoints, _ = self.list_checkpoints(session_id=session_id, limit=5)
        
        # 获取相关的 LU
        lus, lu_count = self.lu_service.list_learning_units(session_id=session_id, limit=10)
        
        return {
            "session_id": session_id,
            "status": session.status,
            "goal": session.goal,
            "scope_id": session.scope_id,
            "max_nl_level": session.max_nl_level,
            "allowed_levels": session.allowed_levels,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "paused_at": session.paused_at.isoformat() if session.paused_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "statistics": {
                "total_checkpoints": session.total_checkpoints,
                "total_learning_units": session.total_learning_units,
                "approved_units": session.approved_units,
                "rejected_units": session.rejected_units,
            },
            "recent_checkpoints": [
                {
                    "id": c.checkpoint_id,
                    "nl_level": c.nl_level,
                    "review_status": c.review_status,
                    "created_at": c.created_at.isoformat(),
                }
                for c in checkpoints
            ],
            "recent_learning_units": [
                {
                    "id": str(lu.id),
                    "title": lu.title,
                    "status": lu.status,
                    "risk_level": lu.risk_level,
                }
                for lu in lus
            ],
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取治理统计"""
        sessions, total_sessions = self.list_sessions()
        
        active_sessions = sum(1 for s in sessions if s.status == "active")
        paused_sessions = sum(1 for s in sessions if s.status == "paused")
        completed_sessions = sum(1 for s in sessions if s.status == "completed")
        terminated_sessions = sum(1 for s in sessions if s.status == "terminated")
        
        pending_checkpoints, _ = self.list_checkpoints(review_status="pending")
        
        return {
            "sessions": {
                "total": total_sessions,
                "active": active_sessions,
                "paused": paused_sessions,
                "completed": completed_sessions,
                "terminated": terminated_sessions,
            },
            "checkpoints": {
                "pending_review": len(pending_checkpoints),
            },
            "interventions": {
                "total": len(self._interventions),
                "by_type": self._count_interventions_by_type(),
            },
        }
    
    def _count_interventions_by_type(self) -> Dict[str, int]:
        """按类型统计干预"""
        counts = {}
        for i in self._interventions:
            t = i.get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

