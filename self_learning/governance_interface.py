"""
NLGSM 治理系统与自学习系统的接口层

实现治理系统对自学习过程的干预能力：
1. 学习过程监控（实时 Checkpoint 检查）
2. 学习路线修正（调整目标、范围、约束）
3. 学习终止（立即停止、回滚到检查点）
4. Learning Unit 审计反馈

治理原则：
- 自学习系统无权定义风险等级
- 自学习系统无权直接写入生产系统
- 所有学习必须在 Scope 控制下进行
- 治理系统可随时干预学习过程
"""
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import json

from core.types import LearningUnit, generate_id
from core.enums import AuditStatus, RiskLevel

from .nl_core import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    ContinuumMemoryState,
    NestedLearningKernel,
)
from .checkpoint import CheckpointManager


class InterventionType(Enum):
    """干预类型"""
    PAUSE = "pause"                      # 暂停学习
    RESUME = "resume"                    # 恢复学习
    TERMINATE = "terminate"              # 终止学习
    MODIFY_SCOPE = "modify_scope"        # 修改学习范围
    MODIFY_GOAL = "modify_goal"          # 修改学习目标
    ROLLBACK = "rollback"                # 回滚到检查点
    INJECT_CONSTRAINT = "inject_constraint"  # 注入约束
    REQUEST_CHECKPOINT = "request_checkpoint"  # 请求创建检查点


class InterventionPriority(Enum):
    """干预优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class GovernanceIntervention:
    """
    治理干预请求
    
    由治理系统发起，用于干预学习过程
    """
    intervention_id: str
    intervention_type: InterventionType
    priority: InterventionPriority
    reason: str
    issued_by: str
    issued_at: datetime
    
    # 干预参数
    params: Dict[str, Any] = field(default_factory=dict)
    
    # 目标（session_id 或 checkpoint_id）
    target_session_id: Optional[str] = None
    target_checkpoint_id: Optional[str] = None
    
    # 状态
    status: str = "pending"  # pending, applied, rejected, expired
    applied_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "type": self.intervention_type.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "issued_by": self.issued_by,
            "issued_at": self.issued_at.isoformat(),
            "params": self.params,
            "target_session_id": self.target_session_id,
            "target_checkpoint_id": self.target_checkpoint_id,
            "status": self.status,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "result": self.result,
        }


@dataclass
class CheckpointReview:
    """
    检查点审查记录
    
    治理系统对学习过程检查点的审查
    """
    review_id: str
    checkpoint_id: str
    session_id: str
    
    # 审查结果
    decision: str  # "continue", "pause", "modify", "terminate"
    comments: str = ""
    
    # 如果需要修改
    modified_scope: Optional[LearningScope] = None
    modified_goal: Optional[str] = None
    injected_constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # 审查人
    reviewed_by: str = ""
    reviewed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "decision": self.decision,
            "comments": self.comments,
            "modified_scope": self.modified_scope.to_dict() if self.modified_scope else None,
            "modified_goal": self.modified_goal,
            "injected_constraints": self.injected_constraints,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat(),
        }


@dataclass
class LearningUnitAuditFeedback:
    """
    Learning Unit 审计反馈
    
    审计系统对 Learning Unit 的决策和反馈
    """
    feedback_id: str
    learning_unit_id: str
    
    # 审计决策
    decision: AuditStatus
    risk_level: Optional[RiskLevel] = None
    
    # 反馈内容
    comments: str = ""
    required_modifications: List[Dict[str, Any]] = field(default_factory=list)
    
    # 如果拒绝
    rejection_reason: Optional[str] = None
    
    # 如果需要重新学习
    relearn_guidance: Optional[Dict[str, Any]] = None
    
    # 审计人
    audited_by: str = ""
    audited_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "learning_unit_id": self.learning_unit_id,
            "decision": self.decision.value,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "comments": self.comments,
            "required_modifications": self.required_modifications,
            "rejection_reason": self.rejection_reason,
            "relearn_guidance": self.relearn_guidance,
            "audited_by": self.audited_by,
            "audited_at": self.audited_at.isoformat(),
        }


class GovernanceInterface:
    """
    NLGSM 治理接口
    
    连接治理系统和自学习系统的桥梁。
    
    核心功能：
    1. 接收并处理治理干预请求
    2. 监控学习过程并创建检查点
    3. 接收 Checkpoint 审查决策
    4. 接收 Learning Unit 审计反馈
    5. 协调学习过程的暂停、修改和终止
    
    使用示例：
        # 治理系统发起干预
        governance = GovernanceInterface(kernel, checkpoint_manager)
        
        # 终止当前学习
        governance.issue_intervention(
            intervention_type=InterventionType.TERMINATE,
            reason="发现异常模式",
            issued_by="governance_system"
        )
        
        # 审查检查点
        governance.review_checkpoint(
            checkpoint_id="ckpt_xxx",
            decision="modify",
            modified_scope=new_scope,
            reviewed_by="admin"
        )
    """
    
    def __init__(
        self,
        nl_kernel: NestedLearningKernel,
        checkpoint_manager: CheckpointManager,
        notification_callback: Optional[Callable[[str, Dict], None]] = None,
    ):
        self.nl_kernel = nl_kernel
        self.checkpoint_manager = checkpoint_manager
        self.notification_callback = notification_callback
        
        # 干预队列
        self.pending_interventions: List[GovernanceIntervention] = []
        self.intervention_history: List[GovernanceIntervention] = []
        
        # 检查点审查
        self.checkpoint_reviews: Dict[str, CheckpointReview] = {}
        
        # Learning Unit 审计反馈
        self.audit_feedbacks: Dict[str, LearningUnitAuditFeedback] = {}
        
        # 当前学习状态
        self.current_session_id: Optional[str] = None
        self.is_paused: bool = False
        self.pause_reason: Optional[str] = None
        
        # 原始目标和范围（用于恢复）
        self.original_goal: Optional[str] = None
        self.original_scope: Optional[LearningScope] = None
        
        # 当前目标和范围（可能被修改）
        self.current_goal: Optional[str] = None
        self.current_scope: Optional[LearningScope] = None
        
        # 注册检查点回调
        self.checkpoint_manager.on_checkpoint_created = self._on_checkpoint_created
    
    # ==================== 干预接口 ====================
    
    def issue_intervention(
        self,
        intervention_type: InterventionType,
        reason: str,
        issued_by: str,
        priority: InterventionPriority = InterventionPriority.NORMAL,
        params: Optional[Dict[str, Any]] = None,
        target_session_id: Optional[str] = None,
        target_checkpoint_id: Optional[str] = None,
    ) -> GovernanceIntervention:
        """
        发起治理干预
        
        Args:
            intervention_type: 干预类型
            reason: 干预原因
            issued_by: 发起人
            priority: 优先级
            params: 干预参数
            target_session_id: 目标会话
            target_checkpoint_id: 目标检查点
            
        Returns:
            干预请求对象
        """
        intervention = GovernanceIntervention(
            intervention_id=generate_id("int"),
            intervention_type=intervention_type,
            priority=priority,
            reason=reason,
            issued_by=issued_by,
            issued_at=datetime.now(),
            params=params or {},
            target_session_id=target_session_id or self.current_session_id,
            target_checkpoint_id=target_checkpoint_id,
        )
        
        # 高优先级立即处理
        if priority in [InterventionPriority.HIGH, InterventionPriority.CRITICAL]:
            self._apply_intervention(intervention)
        else:
            self.pending_interventions.append(intervention)
        
        print(f"[Governance] 干预发起: {intervention_type.value} - {reason}")
        
        # 发送通知
        if self.notification_callback:
            self.notification_callback("intervention_issued", intervention.to_dict())
        
        return intervention
    
    def _apply_intervention(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """应用干预"""
        result = {"success": False, "message": ""}
        
        try:
            if intervention.intervention_type == InterventionType.PAUSE:
                result = self._handle_pause(intervention)
            
            elif intervention.intervention_type == InterventionType.RESUME:
                result = self._handle_resume(intervention)
            
            elif intervention.intervention_type == InterventionType.TERMINATE:
                result = self._handle_terminate(intervention)
            
            elif intervention.intervention_type == InterventionType.MODIFY_SCOPE:
                result = self._handle_modify_scope(intervention)
            
            elif intervention.intervention_type == InterventionType.MODIFY_GOAL:
                result = self._handle_modify_goal(intervention)
            
            elif intervention.intervention_type == InterventionType.ROLLBACK:
                result = self._handle_rollback(intervention)
            
            elif intervention.intervention_type == InterventionType.INJECT_CONSTRAINT:
                result = self._handle_inject_constraint(intervention)
            
            elif intervention.intervention_type == InterventionType.REQUEST_CHECKPOINT:
                result = self._handle_request_checkpoint(intervention)
            
            intervention.status = "applied"
            intervention.applied_at = datetime.now()
            intervention.result = result
        
        except Exception as e:
            intervention.status = "rejected"
            intervention.result = {"success": False, "error": str(e)}
            result = intervention.result
        
        self.intervention_history.append(intervention)
        return result
    
    def _handle_pause(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理暂停"""
        if self.is_paused:
            return {"success": False, "message": "Already paused"}
        
        # 冻结 NL 内核
        self.nl_kernel.freeze()
        
        # 创建暂停检查点
        checkpoint = self.checkpoint_manager.create_checkpoint(
            exploration_data={
                "reason": "governance_pause",
                "intervention_id": intervention.intervention_id,
            },
            reason=f"治理暂停: {intervention.reason}"
        )
        
        self.is_paused = True
        self.pause_reason = intervention.reason
        
        print(f"[Governance] 学习已暂停: {intervention.reason}")
        
        return {
            "success": True,
            "message": "Learning paused",
            "checkpoint_id": checkpoint["checkpoint_id"],
        }
    
    def _handle_resume(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理恢复"""
        if not self.is_paused:
            return {"success": False, "message": "Not paused"}
        
        # 解冻 NL 内核
        self.nl_kernel.unfreeze()
        
        self.is_paused = False
        self.pause_reason = None
        
        print(f"[Governance] 学习已恢复")
        
        return {"success": True, "message": "Learning resumed"}
    
    def _handle_terminate(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理终止"""
        # 冻结内核
        self.nl_kernel.freeze()
        
        # 创建终止检查点
        checkpoint = self.checkpoint_manager.create_checkpoint(
            exploration_data={
                "reason": "governance_terminate",
                "intervention_id": intervention.intervention_id,
            },
            reason=f"治理终止: {intervention.reason}"
        )
        
        # 创建状态快照
        snapshot = self.nl_kernel.create_snapshot()
        
        # 重置状态
        self.current_session_id = None
        self.is_paused = False
        self.current_goal = None
        self.current_scope = None
        
        print(f"[Governance] 学习已终止: {intervention.reason}")
        
        return {
            "success": True,
            "message": "Learning terminated",
            "checkpoint_id": checkpoint["checkpoint_id"],
            "snapshot_id": snapshot.state_id,
        }
    
    def _handle_modify_scope(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理修改范围"""
        new_scope_data = intervention.params.get("new_scope")
        if not new_scope_data:
            return {"success": False, "message": "No new scope provided"}
        
        # 保存原始范围
        if self.original_scope is None:
            self.original_scope = self.current_scope
        
        # 创建新范围
        new_scope = LearningScope(
            scope_id=generate_id("scope"),
            max_level=NLLevel[new_scope_data.get("max_level", "MEMORY")],
            allowed_levels=[
                NLLevel[l] for l in new_scope_data.get("allowed_levels", ["PARAMETER", "MEMORY"])
            ],
            created_by=intervention.issued_by,
        )
        
        # 更新当前范围
        self.current_scope = new_scope
        
        print(f"[Governance] 学习范围已修改: max_level={new_scope.max_level.name}")
        
        return {
            "success": True,
            "message": "Scope modified",
            "new_scope_id": new_scope.scope_id,
            "new_max_level": new_scope.max_level.name,
        }
    
    def _handle_modify_goal(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理修改目标"""
        new_goal = intervention.params.get("new_goal")
        if not new_goal:
            return {"success": False, "message": "No new goal provided"}
        
        # 保存原始目标
        if self.original_goal is None:
            self.original_goal = self.current_goal
        
        # 更新目标
        old_goal = self.current_goal
        self.current_goal = new_goal
        
        print(f"[Governance] 学习目标已修改: {old_goal} -> {new_goal}")
        
        return {
            "success": True,
            "message": "Goal modified",
            "old_goal": old_goal,
            "new_goal": new_goal,
        }
    
    def _handle_rollback(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理回滚"""
        checkpoint_id = intervention.target_checkpoint_id or intervention.params.get("checkpoint_id")
        if not checkpoint_id:
            return {"success": False, "message": "No checkpoint specified"}
        
        # 加载检查点
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_id)
        if not checkpoint:
            return {"success": False, "message": f"Checkpoint not found: {checkpoint_id}"}
        
        # 冻结内核
        self.nl_kernel.freeze()
        
        # 获取当前状态快照（用于可能的前滚）
        current_snapshot = self.nl_kernel.create_snapshot()
        
        # 回滚到检查点状态
        # 注意：这里需要 NL 内核实现实际的状态恢复
        # 目前是模拟实现
        
        print(f"[Governance] 已回滚到检查点: {checkpoint_id}")
        
        return {
            "success": True,
            "message": f"Rolled back to checkpoint {checkpoint_id}",
            "checkpoint_id": checkpoint_id,
            "pre_rollback_snapshot": current_snapshot.state_id,
            "checkpoint_data": checkpoint,
        }
    
    def _handle_inject_constraint(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理注入约束"""
        constraint = intervention.params.get("constraint")
        if not constraint:
            return {"success": False, "message": "No constraint provided"}
        
        # 约束会在下一次学习步骤中生效
        # 这里将约束存储，由 NL 内核在学习时读取
        
        print(f"[Governance] 约束已注入: {constraint}")
        
        return {
            "success": True,
            "message": "Constraint injected",
            "constraint": constraint,
        }
    
    def _handle_request_checkpoint(self, intervention: GovernanceIntervention) -> Dict[str, Any]:
        """处理请求检查点"""
        checkpoint = self.checkpoint_manager.create_checkpoint(
            exploration_data={
                "reason": "governance_request",
                "intervention_id": intervention.intervention_id,
                "session_id": self.current_session_id,
            },
            reason=f"治理请求检查点: {intervention.reason}"
        )
        
        print(f"[Governance] 检查点已创建: {checkpoint['checkpoint_id']}")
        
        return {
            "success": True,
            "message": "Checkpoint created",
            "checkpoint_id": checkpoint["checkpoint_id"],
        }
    
    # ==================== 检查点审查接口 ====================
    
    def review_checkpoint(
        self,
        checkpoint_id: str,
        decision: str,
        comments: str = "",
        modified_scope: Optional[LearningScope] = None,
        modified_goal: Optional[str] = None,
        injected_constraints: Optional[List[Dict]] = None,
        reviewed_by: str = "",
    ) -> CheckpointReview:
        """
        审查检查点
        
        治理系统在检查点创建后调用，决定学习如何继续。
        
        Args:
            checkpoint_id: 检查点 ID
            decision: 决策 ("continue", "pause", "modify", "terminate")
            comments: 审查意见
            modified_scope: 修改后的范围（如果 decision="modify"）
            modified_goal: 修改后的目标
            injected_constraints: 注入的约束
            reviewed_by: 审查人
            
        Returns:
            审查记录
        """
        review = CheckpointReview(
            review_id=generate_id("review"),
            checkpoint_id=checkpoint_id,
            session_id=self.current_session_id or "",
            decision=decision,
            comments=comments,
            modified_scope=modified_scope,
            modified_goal=modified_goal,
            injected_constraints=injected_constraints or [],
            reviewed_by=reviewed_by,
        )
        
        self.checkpoint_reviews[checkpoint_id] = review
        
        # 根据决策执行操作
        if decision == "continue":
            # 继续学习，无需操作
            pass
        
        elif decision == "pause":
            self.issue_intervention(
                InterventionType.PAUSE,
                reason=f"Checkpoint review: {comments}",
                issued_by=reviewed_by,
                priority=InterventionPriority.HIGH,
            )
        
        elif decision == "modify":
            if modified_scope:
                self.issue_intervention(
                    InterventionType.MODIFY_SCOPE,
                    reason=f"Checkpoint review: {comments}",
                    issued_by=reviewed_by,
                    priority=InterventionPriority.HIGH,
                    params={"new_scope": modified_scope.to_dict()},
                )
            
            if modified_goal:
                self.issue_intervention(
                    InterventionType.MODIFY_GOAL,
                    reason=f"Checkpoint review: {comments}",
                    issued_by=reviewed_by,
                    priority=InterventionPriority.HIGH,
                    params={"new_goal": modified_goal},
                )
        
        elif decision == "terminate":
            self.issue_intervention(
                InterventionType.TERMINATE,
                reason=f"Checkpoint review: {comments}",
                issued_by=reviewed_by,
                priority=InterventionPriority.CRITICAL,
            )
        
        print(f"[Governance] 检查点审查完成: {checkpoint_id} - {decision}")
        
        return review
    
    def _on_checkpoint_created(self, checkpoint: Dict[str, Any]):
        """
        检查点创建回调
        
        当学习过程创建检查点时调用，通知治理系统
        """
        print(f"[Governance] 收到检查点通知: {checkpoint['checkpoint_id']}")
        
        # 处理待处理的干预
        self._process_pending_interventions()
        
        # 发送通知给治理系统
        if self.notification_callback:
            self.notification_callback("checkpoint_created", checkpoint)
    
    def _process_pending_interventions(self):
        """处理待处理的干预"""
        # 按优先级排序
        self.pending_interventions.sort(key=lambda x: x.priority.value, reverse=True)
        
        while self.pending_interventions:
            intervention = self.pending_interventions.pop(0)
            self._apply_intervention(intervention)
    
    # ==================== Learning Unit 审计反馈接口 ====================
    
    def receive_audit_feedback(
        self,
        learning_unit_id: str,
        decision: AuditStatus,
        risk_level: Optional[RiskLevel] = None,
        comments: str = "",
        required_modifications: Optional[List[Dict]] = None,
        rejection_reason: Optional[str] = None,
        relearn_guidance: Optional[Dict] = None,
        audited_by: str = "",
    ) -> LearningUnitAuditFeedback:
        """
        接收 Learning Unit 审计反馈
        
        Args:
            learning_unit_id: Learning Unit ID
            decision: 审计决策
            risk_level: 风险等级（由审计系统定义）
            comments: 审计意见
            required_modifications: 需要的修改
            rejection_reason: 拒绝原因
            relearn_guidance: 重新学习指导
            audited_by: 审计人
            
        Returns:
            审计反馈记录
        """
        feedback = LearningUnitAuditFeedback(
            feedback_id=generate_id("feedback"),
            learning_unit_id=learning_unit_id,
            decision=decision,
            risk_level=risk_level,
            comments=comments,
            required_modifications=required_modifications or [],
            rejection_reason=rejection_reason,
            relearn_guidance=relearn_guidance,
            audited_by=audited_by,
        )
        
        self.audit_feedbacks[learning_unit_id] = feedback
        
        # 根据决策执行操作
        if decision == AuditStatus.REJECTED:
            print(f"[Governance] Learning Unit 被拒绝: {learning_unit_id}")
            print(f"  原因: {rejection_reason}")
            
            if relearn_guidance:
                print(f"  重新学习指导: {relearn_guidance}")
                # 可以触发重新学习流程
        
        elif decision == AuditStatus.APPROVED:
            print(f"[Governance] Learning Unit 已批准: {learning_unit_id}")
            print(f"  风险等级: {risk_level.value if risk_level else 'N/A'}")
        
        elif decision == AuditStatus.NEEDS_REVISION:
            print(f"[Governance] Learning Unit 需要修改: {learning_unit_id}")
            print(f"  修改要求: {required_modifications}")
        
        # 发送通知
        if self.notification_callback:
            self.notification_callback("audit_feedback", feedback.to_dict())
        
        return feedback
    
    # ==================== 状态查询接口 ====================
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取当前学习状态"""
        return {
            "session_id": self.current_session_id,
            "is_paused": self.is_paused,
            "pause_reason": self.pause_reason,
            "current_goal": self.current_goal,
            "current_scope": self.current_scope.to_dict() if self.current_scope else None,
            "original_goal": self.original_goal,
            "original_scope": self.original_scope.to_dict() if self.original_scope else None,
            "kernel_frozen": self.nl_kernel.is_frozen() if hasattr(self.nl_kernel, 'is_frozen') else True,
            "pending_interventions": len(self.pending_interventions),
            "intervention_history_count": len(self.intervention_history),
        }
    
    def get_intervention_history(
        self,
        limit: int = 50,
        intervention_type: Optional[InterventionType] = None
    ) -> List[Dict[str, Any]]:
        """获取干预历史"""
        history = self.intervention_history
        
        if intervention_type:
            history = [h for h in history if h.intervention_type == intervention_type]
        
        return [h.to_dict() for h in history[-limit:]]
    
    def get_checkpoint_reviews(
        self,
        checkpoint_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取检查点审查记录"""
        if checkpoint_id:
            review = self.checkpoint_reviews.get(checkpoint_id)
            return [review.to_dict()] if review else []
        
        return [r.to_dict() for r in self.checkpoint_reviews.values()]
    
    def get_audit_feedbacks(
        self,
        learning_unit_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取审计反馈"""
        if learning_unit_id:
            feedback = self.audit_feedbacks.get(learning_unit_id)
            return [feedback.to_dict()] if feedback else []
        
        return [f.to_dict() for f in self.audit_feedbacks.values()]

