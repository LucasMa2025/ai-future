"""
审批服务

实现:
1. Learning Unit 审批工作流
2. 多签审批
3. 审批通知
4. 与知识转移服务集成
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from uuid import UUID
import uuid
import logging

from sqlalchemy.orm import Session

from ..config import settings
from ..core.enums import (
    RiskLevel, 
    ApprovalStatus, 
    ApprovalDecision, 
    NotificationType,
    LearningUnitStatus,
)
from ..core.exceptions import NotFoundError, BusinessError, PermissionDeniedError
from ..models.user import User
from ..models.learning_unit import LearningUnit, LUAuditHistory

if TYPE_CHECKING:
    from .learning_unit_service import LearningUnitService
    from .knowledge_transfer_service import KnowledgeTransferService
    from .notification_service import NotificationService


logger = logging.getLogger(__name__)


class ApprovalService:
    """
    审批服务
    
    处理 Learning Unit 审批工作流
    """
    
    def __init__(
        self, 
        db: Session,
        lu_service: Optional["LearningUnitService"] = None,
        transfer_service: Optional["KnowledgeTransferService"] = None,
        notification_service: Optional["NotificationService"] = None,
    ):
        self.db = db
        self._lu_service = lu_service
        self._transfer_service = transfer_service
        self._notification_service = notification_service
        
        # 内存中的待审批队列
        self._pending_approvals: Dict[str, Dict] = {}
    
    @property
    def lu_service(self) -> "LearningUnitService":
        """延迟加载 LearningUnitService"""
        if self._lu_service is None:
            from .learning_unit_service import LearningUnitService
            self._lu_service = LearningUnitService(self.db)
        return self._lu_service
    
    @property
    def transfer_service(self) -> "KnowledgeTransferService":
        """延迟加载 KnowledgeTransferService"""
        if self._transfer_service is None:
            from .knowledge_transfer_service import KnowledgeTransferService
            self._transfer_service = KnowledgeTransferService(self.db)
        return self._transfer_service
    
    @property
    def notification_service(self) -> Optional["NotificationService"]:
        """延迟加载 NotificationService"""
        if self._notification_service is None:
            try:
                from .notification_service import NotificationService
                self._notification_service = NotificationService(self.db)
            except Exception:
                pass
        return self._notification_service
    
    # ==================== 风险等级配置 ====================
    
    def get_required_approvers(self, risk_level: RiskLevel) -> int:
        """获取所需审批人数"""
        if risk_level == RiskLevel.CRITICAL:
            return getattr(settings, 'MULTI_SIG_REQUIRED_CRITICAL', 3)
        elif risk_level == RiskLevel.HIGH:
            return getattr(settings, 'MULTI_SIG_REQUIRED_HIGH', 2)
        else:
            return 1
    
    def get_required_roles(self, risk_level: RiskLevel) -> List[str]:
        """获取所需审批角色"""
        if risk_level == RiskLevel.CRITICAL:
            return ["governance_committee"]
        elif risk_level == RiskLevel.HIGH:
            return ["governance_committee", "senior_engineer"]
        elif risk_level == RiskLevel.MEDIUM:
            return ["senior_engineer", "ml_engineer"]
        else:
            return ["ml_engineer", "operator"]
    
    # ==================== Learning Unit 审批 ====================
    
    def initiate_lu_approval(
        self,
        learning_unit: LearningUnit,
        initiated_by: Optional[UUID] = None,
    ) -> Dict[str, Any]:
        """
        发起 Learning Unit 审批流程
        
        Args:
            learning_unit: 待审批的 Learning Unit
            initiated_by: 发起人ID
            
        Returns:
            审批进度信息
        """
        # 确定风险等级
        risk_level = RiskLevel(learning_unit.risk_level) if learning_unit.risk_level else RiskLevel.MEDIUM
        
        approval_id = str(uuid.uuid4())
        required_approvers = self.get_required_approvers(risk_level)
        required_roles = self.get_required_roles(risk_level)
        
        deadline = datetime.utcnow() + timedelta(
            hours=getattr(settings, 'APPROVAL_TIMEOUT_HOURS', 24)
        )
        
        approval = {
            "id": approval_id,
            "target_type": "learning_unit",
            "target_id": str(learning_unit.id),
            "learning_unit_id": str(learning_unit.id),
            "risk_level": risk_level.value,
            "title": f"LU审批: {learning_unit.title}",
            "description": learning_unit.description,
            "metadata": {
                "nl_level": learning_unit.nl_level,
                "scope_id": learning_unit.scope_id,
                "constraint_count": len(learning_unit.constraints),
                "risk_score": learning_unit.risk_score,
                "risk_factors": learning_unit.risk_factors,
            },
            "status": ApprovalStatus.PENDING.value,
            "required_approvers": required_approvers,
            "required_roles": required_roles,
            "current_approvers": 0,
            "approver_list": [],
            "initiated_by": str(initiated_by) if initiated_by else None,
            "deadline": deadline.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            
            # LU 特有字段
            "auto_transfer_on_approve": True,  # 审批通过后自动转移知识
            "corrections": [],  # 修正记录
        }
        
        self._pending_approvals[approval_id] = approval
        
        # 更新 LU 状态
        learning_unit.approval_id = approval_id
        self.db.commit()
        
        # 发送通知
        self._notify_approvers(approval)
        
        logger.info(f"Initiated approval {approval_id} for LU {learning_unit.id}")
        
        return approval
    
    def submit_lu_approval(
        self,
        approval_id: str,
        approver: User,
        decision: ApprovalDecision,
        comments: Optional[str] = None,
        corrections: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        提交 Learning Unit 审批决策
        
        Args:
            approval_id: 审批ID
            approver: 审批人
            decision: 审批决策 (APPROVE/REJECT/CORRECT/TERMINATE)
            comments: 审批意见
            corrections: 修正列表（仅当 decision=CORRECT 时）
            
        Returns:
            更新后的审批状态
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            raise NotFoundError("Approval", approval_id)
        
        if approval["status"] != ApprovalStatus.PENDING.value:
            raise BusinessError(
                "Approval has already been processed",
                code="ALREADY_PROCESSED"
            )
        
        # 检查是否已审批
        for existing in approval["approver_list"]:
            if existing["user_id"] == str(approver.id):
                raise BusinessError(
                    "Already approved by this user",
                    code="DUPLICATE_APPROVAL"
                )
        
        # 检查角色权限
        approver_roles = [r.name for r in approver.roles]
        if not any(r in approval["required_roles"] for r in approver_roles):
            if not approver.is_superuser:
                raise PermissionDeniedError(
                    f"Required roles: {', '.join(approval['required_roles'])}"
                )
        
        # 检查风险等级权限
        risk_level = approval["risk_level"]
        if not self._can_approve_risk_level(approver, risk_level):
            raise PermissionDeniedError(
                f"Cannot approve {risk_level} risk level"
            )
        
        # 记录审批
        approver_entry = {
            "user_id": str(approver.id),
            "username": approver.username,
            "decision": decision.value,
            "comments": comments,
            "corrections": corrections,
            "approved_at": datetime.utcnow().isoformat(),
        }
        approval["approver_list"].append(approver_entry)
        
        # 获取 LU
        lu_id = UUID(approval["learning_unit_id"])
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        # 处理决策
        result = {
            "status": approval["status"],
            "current_approvers": approval["current_approvers"],
            "required_approvers": approval["required_approvers"],
            "is_complete": False,
            "approver_list": approval["approver_list"],
            "transfer_result": None,
        }
        
        if decision == ApprovalDecision.APPROVE:
            approval["current_approvers"] += 1
            
            # 检查是否达到要求
            if approval["current_approvers"] >= approval["required_approvers"]:
                approval["status"] = ApprovalStatus.COMPLETED.value
                approval["completed_at"] = datetime.utcnow().isoformat()
                result["is_complete"] = True
                
                # 审批通过，更新 LU 并转移知识
                self.lu_service.approve(lu_id, approver, approval_id, comments)
                
                if approval.get("auto_transfer_on_approve", True):
                    try:
                        transfer_result = self.transfer_service.transfer_to_aga(lu_id)
                        result["transfer_result"] = transfer_result
                        logger.info(f"Knowledge transferred for LU {lu_id}")
                    except Exception as e:
                        logger.error(f"Knowledge transfer failed for LU {lu_id}: {e}")
                        result["transfer_error"] = str(e)
        
        elif decision == ApprovalDecision.CORRECT:
            # 修正并通过
            if not corrections:
                raise BusinessError(
                    "Corrections required for CORRECT decision",
                    code="CORRECTIONS_REQUIRED"
                )
            
            approval["corrections"].extend(corrections)
            approval["current_approvers"] += 1
            
            if approval["current_approvers"] >= approval["required_approvers"]:
                approval["status"] = ApprovalStatus.COMPLETED.value
                approval["completed_at"] = datetime.utcnow().isoformat()
                result["is_complete"] = True
                
                # 修正 LU
                self.lu_service.correct(lu_id, approver, corrections, approval_id, comments)
                
                if approval.get("auto_transfer_on_approve", True):
                    try:
                        transfer_result = self.transfer_service.transfer_to_aga(lu_id)
                        result["transfer_result"] = transfer_result
                    except Exception as e:
                        logger.error(f"Knowledge transfer failed for LU {lu_id}: {e}")
                        result["transfer_error"] = str(e)
        
        elif decision == ApprovalDecision.REJECT:
            approval["status"] = ApprovalStatus.REJECTED.value
            approval["completed_at"] = datetime.utcnow().isoformat()
            result["is_complete"] = True
            
            # 拒绝 LU
            self.lu_service.reject(lu_id, approver, comments or "Rejected by approver")
        
        elif decision == ApprovalDecision.TERMINATE:
            approval["status"] = ApprovalStatus.REJECTED.value
            approval["completed_at"] = datetime.utcnow().isoformat()
            result["is_complete"] = True
            
            # 终止 LU
            self.lu_service.terminate(lu_id, approver, comments or "Terminated by approver")
        
        result["status"] = approval["status"]
        result["current_approvers"] = approval["current_approvers"]
        
        # 发送通知
        if result["is_complete"]:
            self._notify_approval_complete(approval, decision)
        
        return result
    
    # ==================== 通用审批（兼容原有接口） ====================
    
    def initiate_approval(
        self,
        target_type: str,
        target_id: str,
        risk_level: RiskLevel,
        title: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        initiated_by: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """
        发起审批流程（通用接口）
        """
        # 如果是 Learning Unit，使用专用方法
        if target_type == "learning_unit":
            lu = self.lu_service.get_learning_unit_or_404(UUID(target_id))
            return self.initiate_lu_approval(lu, initiated_by)
        
        # 其他类型使用通用逻辑
        approval_id = str(uuid.uuid4())
        required_approvers = self.get_required_approvers(risk_level)
        required_roles = self.get_required_roles(risk_level)
        
        deadline = datetime.utcnow() + timedelta(
            hours=getattr(settings, 'APPROVAL_TIMEOUT_HOURS', 24)
        )
        
        approval = {
            "id": approval_id,
            "target_type": target_type,
            "target_id": target_id,
            "risk_level": risk_level.value,
            "title": title,
            "description": description,
            "metadata": metadata or {},
            "status": ApprovalStatus.PENDING.value,
            "required_approvers": required_approvers,
            "required_roles": required_roles,
            "current_approvers": 0,
            "approver_list": [],
            "initiated_by": str(initiated_by) if initiated_by else None,
            "deadline": deadline.isoformat(),
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
        }
        
        self._pending_approvals[approval_id] = approval
        
        return approval
    
    def submit_approval(
        self,
        approval_id: str,
        approver: User,
        decision: ApprovalDecision,
        comments: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        提交审批决策（通用接口）
        """
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            raise NotFoundError("Approval", approval_id)
        
        # 如果是 LU 审批，使用专用方法
        if approval.get("target_type") == "learning_unit":
            return self.submit_lu_approval(approval_id, approver, decision, comments)
        
        # 通用审批逻辑
        if approval["status"] != ApprovalStatus.PENDING.value:
            raise BusinessError(
                "Approval has already been processed",
                code="ALREADY_PROCESSED"
            )
        
        # 检查是否已审批
        for existing in approval["approver_list"]:
            if existing["user_id"] == str(approver.id):
                raise BusinessError(
                    "Already approved by this user",
                    code="DUPLICATE_APPROVAL"
                )
        
        # 检查角色权限
        approver_roles = [r.name for r in approver.roles]
        if not any(r in approval["required_roles"] for r in approver_roles):
            if not approver.is_superuser:
                raise PermissionDeniedError(
                    f"Required roles: {', '.join(approval['required_roles'])}"
                )
        
        # 检查风险等级权限
        risk_level = approval["risk_level"]
        if not self._can_approve_risk_level(approver, risk_level):
            raise PermissionDeniedError(
                f"Cannot approve {risk_level} risk level"
            )
        
        # 记录审批
        approver_entry = {
            "user_id": str(approver.id),
            "username": approver.username,
            "decision": decision.value,
            "comments": comments,
            "approved_at": datetime.utcnow().isoformat(),
        }
        approval["approver_list"].append(approver_entry)
        
        # 处理决策
        if decision == ApprovalDecision.APPROVE:
            approval["current_approvers"] += 1
            
            if approval["current_approvers"] >= approval["required_approvers"]:
                approval["status"] = ApprovalStatus.COMPLETED.value
                approval["completed_at"] = datetime.utcnow().isoformat()
        
        elif decision in (ApprovalDecision.REJECT, ApprovalDecision.TERMINATE):
            approval["status"] = ApprovalStatus.REJECTED.value
            approval["completed_at"] = datetime.utcnow().isoformat()
        
        return {
            "status": approval["status"],
            "current_approvers": approval["current_approvers"],
            "required_approvers": approval["required_approvers"],
            "is_complete": approval["status"] == ApprovalStatus.COMPLETED.value,
            "approver_list": approval["approver_list"],
        }
    
    # ==================== 查询接口 ====================
    
    def get_pending_approvals(
        self,
        user: Optional[User] = None,
        target_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取待审批列表"""
        result = []
        
        for approval in self._pending_approvals.values():
            if approval["status"] != ApprovalStatus.PENDING.value:
                continue
            
            if target_type and approval["target_type"] != target_type:
                continue
            
            if user:
                already_approved = any(
                    a["user_id"] == str(user.id)
                    for a in approval["approver_list"]
                )
                if already_approved:
                    continue
                
                user_roles = [r.name for r in user.roles]
                if not user.is_superuser:
                    if not any(r in approval["required_roles"] for r in user_roles):
                        continue
            
            result.append(approval)
        
        return result
    
    def get_pending_lu_approvals(self, user: Optional[User] = None) -> List[Dict[str, Any]]:
        """获取待审批的 Learning Unit 列表"""
        return self.get_pending_approvals(user, target_type="learning_unit")
    
    def get_approval_by_id(self, approval_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取审批"""
        return self._pending_approvals.get(approval_id)
    
    def get_approval_by_lu(self, lu_id: UUID) -> Optional[Dict[str, Any]]:
        """根据 LU ID 获取审批"""
        for approval in self._pending_approvals.values():
            if approval.get("learning_unit_id") == str(lu_id):
                return approval
        return None
    
    def get_approval_by_target(
        self,
        target_type: str,
        target_id: str
    ) -> Optional[Dict[str, Any]]:
        """根据目标获取审批"""
        for approval in self._pending_approvals.values():
            if (approval["target_type"] == target_type and 
                approval["target_id"] == target_id):
                return approval
        return None
    
    def cancel_approval(
        self,
        approval_id: str,
        cancelled_by: UUID,
        reason: Optional[str] = None
    ) -> bool:
        """取消审批"""
        approval = self._pending_approvals.get(approval_id)
        if not approval:
            raise NotFoundError("Approval", approval_id)
        
        if approval["status"] != ApprovalStatus.PENDING.value:
            raise BusinessError(
                "Cannot cancel processed approval",
                code="CANNOT_CANCEL"
            )
        
        approval["status"] = "cancelled"
        approval["completed_at"] = datetime.utcnow().isoformat()
        approval["cancel_reason"] = reason
        approval["cancelled_by"] = str(cancelled_by)
        
        return True
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """获取审批统计"""
        total = len(self._pending_approvals)
        pending = sum(
            1 for a in self._pending_approvals.values()
            if a["status"] == ApprovalStatus.PENDING.value
        )
        completed = sum(
            1 for a in self._pending_approvals.values()
            if a["status"] == ApprovalStatus.COMPLETED.value
        )
        rejected = sum(
            1 for a in self._pending_approvals.values()
            if a["status"] == ApprovalStatus.REJECTED.value
        )
        
        # 按风险等级统计
        by_risk_level = {}
        for a in self._pending_approvals.values():
            level = a["risk_level"]
            by_risk_level[level] = by_risk_level.get(level, 0) + 1
        
        # LU 审批统计
        lu_approvals = [
            a for a in self._pending_approvals.values()
            if a.get("target_type") == "learning_unit"
        ]
        
        return {
            "total": total,
            "pending": pending,
            "completed": completed,
            "rejected": rejected,
            "by_risk_level": by_risk_level,
            "lu_approvals": {
                "total": len(lu_approvals),
                "pending": sum(1 for a in lu_approvals if a["status"] == ApprovalStatus.PENDING.value),
            },
        }
    
    # ==================== 私有方法 ====================
    
    def _can_approve_risk_level(self, user: User, risk_level: str) -> bool:
        """检查用户是否可以审批指定风险等级"""
        if user.is_superuser:
            return True
        
        risk_order = ["low", "medium", "high", "critical"]
        
        for role in user.roles:
            if role.risk_level_limit:
                try:
                    user_level_idx = risk_order.index(role.risk_level_limit)
                    required_idx = risk_order.index(risk_level)
                    if required_idx <= user_level_idx:
                        return True
                except ValueError:
                    continue
        
        return False
    
    def _notify_approvers(self, approval: Dict[str, Any]):
        """通知审批人"""
        if not self.notification_service:
            return
        
        try:
            # 获取需要通知的用户（具有所需角色的用户）
            # 这里简化处理，实际应该查询数据库
            logger.info(f"Notifying approvers for approval {approval['id']}")
        except Exception as e:
            logger.error(f"Failed to notify approvers: {e}")
    
    def _notify_approval_complete(self, approval: Dict[str, Any], decision: ApprovalDecision):
        """通知审批完成"""
        if not self.notification_service:
            return
        
        try:
            logger.info(
                f"Approval {approval['id']} completed with decision {decision.value}"
            )
        except Exception as e:
            logger.error(f"Failed to notify approval completion: {e}")
