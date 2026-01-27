"""
Learning Unit 服务

实现:
1. Learning Unit CRUD
2. 约束管理
3. 状态流转
4. 审计历史
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
import uuid
import logging

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..models.learning_unit import (
    LearningUnit, 
    LUConstraint, 
    LUAuditHistory,
    LearningSession,
    Checkpoint,
)
from ..models.user import User
from ..core.enums import LearningUnitStatus, RiskLevel
from ..core.exceptions import NotFoundError, BusinessError, PermissionDeniedError


logger = logging.getLogger(__name__)


class LearningUnitService:
    """
    Learning Unit 服务
    
    管理学习单元的完整生命周期
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== CRUD 操作 ====================
    
    def create_learning_unit(
        self,
        title: str,
        constraints: List[Dict[str, Any]],
        learning_session_id: Optional[str] = None,
        checkpoint_id: Optional[str] = None,
        scope_id: Optional[str] = None,
        nl_level: Optional[str] = None,
        learning_goal: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ) -> LearningUnit:
        """
        创建 Learning Unit
        
        Args:
            title: 标题
            constraints: 约束列表 [{"condition": str, "decision": str, "confidence": float}]
            learning_session_id: 学习会话ID
            checkpoint_id: 检查点ID
            scope_id: 学习范围ID
            nl_level: NL层级
            learning_goal: 学习目标
            description: 描述
            metadata: 元数据
            tags: 标签
        
        Returns:
            创建的 LearningUnit
        """
        # 创建 LU
        lu = LearningUnit(
            title=title,
            description=description,
            learning_session_id=learning_session_id,
            checkpoint_id=checkpoint_id,
            scope_id=scope_id,
            nl_level=nl_level,
            learning_goal=learning_goal,
            status=LearningUnitStatus.PENDING.value,
            metadata=metadata or {},
            tags=tags or [],
        )
        
        self.db.add(lu)
        self.db.flush()  # 获取 ID
        
        # 创建约束
        for c in constraints:
            constraint = LUConstraint(
                learning_unit_id=lu.id,
                condition=c.get("condition", ""),
                decision=c.get("decision", ""),
                confidence=c.get("confidence", 0.5),
                source_type=c.get("source_type", "llm"),
                source_evidence=c.get("source_evidence"),
            )
            self.db.add(constraint)
        
        # 记录审计历史
        self._add_audit_history(
            lu.id,
            action="create",
            to_status=LearningUnitStatus.PENDING.value,
            details={"constraint_count": len(constraints)},
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.info(f"Created LearningUnit {lu.id} with {len(constraints)} constraints")
        
        return lu
    
    def get_learning_unit(self, lu_id: UUID) -> Optional[LearningUnit]:
        """获取 Learning Unit"""
        return self.db.query(LearningUnit).filter(LearningUnit.id == lu_id).first()
    
    def get_learning_unit_or_404(self, lu_id: UUID) -> LearningUnit:
        """获取 Learning Unit，不存在则抛出异常"""
        lu = self.get_learning_unit(lu_id)
        if not lu:
            raise NotFoundError("LearningUnit", str(lu_id))
        return lu
    
    def list_learning_units(
        self,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        session_id: Optional[str] = None,
        requires_review: Optional[bool] = None,
        is_internalized: Optional[bool] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[LearningUnit], int]:
        """
        列出 Learning Units
        
        Returns:
            (列表, 总数)
        """
        query = self.db.query(LearningUnit)
        
        if status:
            query = query.filter(LearningUnit.status == status)
        
        if risk_level:
            query = query.filter(LearningUnit.risk_level == risk_level)
        
        if session_id:
            query = query.filter(LearningUnit.learning_session_id == session_id)
        
        if requires_review is not None:
            query = query.filter(LearningUnit.requires_human_review == requires_review)
        
        if is_internalized is not None:
            query = query.filter(LearningUnit.is_internalized == is_internalized)
        
        total = query.count()
        items = query.order_by(
            LearningUnit.review_priority.desc(),
            LearningUnit.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def get_pending_reviews(
        self,
        reviewer: Optional[User] = None,
        risk_level: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[LearningUnit], int]:
        """
        获取待审核的 Learning Units
        """
        query = self.db.query(LearningUnit).filter(
            LearningUnit.status.in_([
                LearningUnitStatus.PENDING.value,
                LearningUnitStatus.AUTO_CLASSIFIED.value,
                LearningUnitStatus.HUMAN_REVIEW.value,
            ]),
            LearningUnit.requires_human_review == True,
        )
        
        if risk_level:
            query = query.filter(LearningUnit.risk_level == risk_level)
        
        # 如果指定了审核人，检查其权限
        if reviewer and not reviewer.is_superuser:
            # 获取用户可审核的最高风险等级
            max_risk = self._get_user_max_risk_level(reviewer)
            risk_order = ["low", "medium", "high", "critical"]
            if max_risk in risk_order:
                allowed_risks = risk_order[:risk_order.index(max_risk) + 1]
                query = query.filter(
                    or_(
                        LearningUnit.risk_level.in_(allowed_risks),
                        LearningUnit.risk_level.is_(None)
                    )
                )
        
        total = query.count()
        items = query.order_by(
            LearningUnit.review_priority.desc(),
            LearningUnit.created_at.asc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def update_learning_unit(
        self,
        lu_id: UUID,
        updates: Dict[str, Any],
        actor: Optional[User] = None,
    ) -> LearningUnit:
        """更新 Learning Unit"""
        lu = self.get_learning_unit_or_404(lu_id)
        
        allowed_fields = [
            "title", "description", "risk_level", "risk_score",
            "risk_factors", "review_priority", "tags", "metadata"
        ]
        
        for field, value in updates.items():
            if field in allowed_fields:
                setattr(lu, field, value)
        
        self._add_audit_history(
            lu_id,
            action="update",
            actor=actor,
            details={"updated_fields": list(updates.keys())},
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        return lu
    
    def delete_learning_unit(self, lu_id: UUID, actor: Optional[User] = None) -> bool:
        """删除 Learning Unit"""
        lu = self.get_learning_unit_or_404(lu_id)
        
        if lu.is_internalized:
            raise BusinessError(
                "Cannot delete internalized LearningUnit",
                code="CANNOT_DELETE_INTERNALIZED"
            )
        
        self.db.delete(lu)
        self.db.commit()
        
        logger.info(f"Deleted LearningUnit {lu_id}")
        return True
    
    # ==================== 状态流转 ====================
    
    def auto_classify(
        self,
        lu_id: UUID,
        classification: Dict[str, Any],
        risk_level: str,
        risk_score: float,
        risk_factors: List[str],
        requires_human_review: bool = True,
    ) -> LearningUnit:
        """
        自动分类 Learning Unit
        
        由系统调用，设置风险等级和是否需要人工审核
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        if lu.status != LearningUnitStatus.PENDING.value:
            raise BusinessError(
                f"Cannot auto-classify LU in status {lu.status}",
                code="INVALID_STATUS"
            )
        
        lu.status = LearningUnitStatus.AUTO_CLASSIFIED.value
        lu.auto_classification = classification
        lu.risk_level = risk_level
        lu.risk_score = risk_score
        lu.risk_factors = risk_factors
        lu.requires_human_review = requires_human_review
        
        # 设置审核优先级
        risk_priority = {"low": 2, "medium": 5, "high": 8, "critical": 10}
        lu.review_priority = risk_priority.get(risk_level, 5)
        
        self._add_audit_history(
            lu_id,
            action="auto_classify",
            from_status=LearningUnitStatus.PENDING.value,
            to_status=LearningUnitStatus.AUTO_CLASSIFIED.value,
            details={
                "risk_level": risk_level,
                "risk_score": risk_score,
                "requires_human_review": requires_human_review,
            },
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        return lu
    
    def submit_for_review(self, lu_id: UUID, actor: Optional[User] = None) -> LearningUnit:
        """提交人工审核"""
        lu = self.get_learning_unit_or_404(lu_id)
        
        if lu.status not in [
            LearningUnitStatus.PENDING.value,
            LearningUnitStatus.AUTO_CLASSIFIED.value
        ]:
            raise BusinessError(
                f"Cannot submit for review from status {lu.status}",
                code="INVALID_STATUS"
            )
        
        old_status = lu.status
        lu.status = LearningUnitStatus.HUMAN_REVIEW.value
        lu.requires_human_review = True
        
        self._add_audit_history(
            lu_id,
            action="submit_for_review",
            actor=actor,
            from_status=old_status,
            to_status=LearningUnitStatus.HUMAN_REVIEW.value,
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        return lu
    
    def approve(
        self,
        lu_id: UUID,
        approver: User,
        approval_id: str,
        comments: Optional[str] = None,
    ) -> LearningUnit:
        """
        审批通过 Learning Unit
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        # 检查权限
        if not self._can_approve(approver, lu):
            raise PermissionDeniedError(
                f"User {approver.username} cannot approve risk level {lu.risk_level}"
            )
        
        old_status = lu.status
        lu.status = LearningUnitStatus.APPROVED.value
        lu.approval_id = approval_id
        lu.approved_by = approver.id
        lu.approved_at = datetime.utcnow()
        lu.approval_comments = comments
        
        # 更新约束状态
        for constraint in lu.constraints:
            constraint.is_approved = True
        
        self._add_audit_history(
            lu_id,
            action="approve",
            actor=approver,
            from_status=old_status,
            to_status=LearningUnitStatus.APPROVED.value,
            comments=comments,
            details={"approval_id": approval_id},
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.info(f"LearningUnit {lu_id} approved by {approver.username}")
        
        return lu
    
    def reject(
        self,
        lu_id: UUID,
        rejector: User,
        reason: str,
        comments: Optional[str] = None,
    ) -> LearningUnit:
        """
        拒绝 Learning Unit
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        old_status = lu.status
        lu.status = LearningUnitStatus.REJECTED.value
        lu.approval_comments = f"Rejected: {reason}. {comments or ''}"
        
        self._add_audit_history(
            lu_id,
            action="reject",
            actor=rejector,
            from_status=old_status,
            to_status=LearningUnitStatus.REJECTED.value,
            comments=comments,
            details={"reason": reason},
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.info(f"LearningUnit {lu_id} rejected by {rejector.username}: {reason}")
        
        return lu
    
    def correct(
        self,
        lu_id: UUID,
        corrector: User,
        corrections: List[Dict[str, Any]],
        approval_id: str,
        comments: Optional[str] = None,
    ) -> LearningUnit:
        """
        修正并通过 Learning Unit
        
        Args:
            corrections: 修正列表 [{"constraint_id": uuid, "condition": str, "decision": str}]
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        # 保存原始约束
        original_constraints = [
            {
                "id": str(c.id),
                "condition": c.condition,
                "decision": c.decision,
            }
            for c in lu.constraints
        ]
        lu.original_constraints = original_constraints
        
        # 应用修正
        constraint_map = {str(c.id): c for c in lu.constraints}
        for correction in corrections:
            constraint_id = correction.get("constraint_id")
            if constraint_id and constraint_id in constraint_map:
                c = constraint_map[constraint_id]
                c.original_condition = c.condition
                c.original_decision = c.decision
                c.condition = correction.get("condition", c.condition)
                c.decision = correction.get("decision", c.decision)
                c.is_modified = True
                c.is_approved = True
        
        old_status = lu.status
        lu.status = LearningUnitStatus.CORRECTED.value
        lu.is_corrected = True
        lu.correction_summary = comments
        lu.approval_id = approval_id
        lu.approved_by = corrector.id
        lu.approved_at = datetime.utcnow()
        lu.approval_comments = comments
        
        self._add_audit_history(
            lu_id,
            action="correct",
            actor=corrector,
            from_status=old_status,
            to_status=LearningUnitStatus.CORRECTED.value,
            comments=comments,
            details={
                "approval_id": approval_id,
                "corrections_count": len(corrections),
            },
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.info(f"LearningUnit {lu_id} corrected by {corrector.username}")
        
        return lu
    
    def terminate(
        self,
        lu_id: UUID,
        terminator: User,
        reason: str,
    ) -> LearningUnit:
        """
        终止 Learning Unit
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        old_status = lu.status
        lu.status = LearningUnitStatus.TERMINATED.value
        
        self._add_audit_history(
            lu_id,
            action="terminate",
            actor=terminator,
            from_status=old_status,
            to_status=LearningUnitStatus.TERMINATED.value,
            details={"reason": reason},
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.info(f"LearningUnit {lu_id} terminated by {terminator.username}: {reason}")
        
        return lu
    
    # ==================== AGA 内化状态 ====================
    
    def mark_internalized(
        self,
        lu_id: UUID,
        aga_slot_mapping: Dict[int, int],
        lifecycle_state: str = "probationary",
    ) -> LearningUnit:
        """
        标记 Learning Unit 已内化到 AGA
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        if lu.status not in [
            LearningUnitStatus.APPROVED.value,
            LearningUnitStatus.CORRECTED.value
        ]:
            raise BusinessError(
                f"Cannot internalize LU in status {lu.status}",
                code="INVALID_STATUS"
            )
        
        lu.is_internalized = True
        lu.internalized_at = datetime.utcnow()
        lu.aga_slot_mapping = aga_slot_mapping
        lu.lifecycle_state = lifecycle_state
        
        self._add_audit_history(
            lu_id,
            action="internalize",
            details={
                "aga_slot_mapping": aga_slot_mapping,
                "lifecycle_state": lifecycle_state,
            },
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.info(f"LearningUnit {lu_id} internalized to AGA")
        
        return lu
    
    def update_lifecycle_state(
        self,
        lu_id: UUID,
        new_state: str,
        actor: Optional[User] = None,
    ) -> LearningUnit:
        """
        更新 AGA 生命周期状态
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(
                "LU is not internalized",
                code="NOT_INTERNALIZED"
            )
        
        old_state = lu.lifecycle_state
        lu.lifecycle_state = new_state
        
        self._add_audit_history(
            lu_id,
            action="lifecycle_update",
            actor=actor,
            details={
                "from_state": old_state,
                "to_state": new_state,
            },
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        return lu
    
    def quarantine(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> LearningUnit:
        """
        隔离 Learning Unit（从 AGA 移除影响）
        """
        lu = self.get_learning_unit_or_404(lu_id)
        
        old_state = lu.lifecycle_state
        lu.lifecycle_state = "quarantined"
        
        self._add_audit_history(
            lu_id,
            action="quarantine",
            actor=actor,
            details={
                "from_state": old_state,
                "reason": reason,
            },
        )
        
        self.db.commit()
        self.db.refresh(lu)
        
        logger.warning(f"LearningUnit {lu_id} quarantined: {reason}")
        
        return lu
    
    # ==================== 约束操作 ====================
    
    def get_constraints(self, lu_id: UUID) -> List[LUConstraint]:
        """获取 Learning Unit 的所有约束"""
        return self.db.query(LUConstraint).filter(
            LUConstraint.learning_unit_id == lu_id
        ).all()
    
    def add_constraint(
        self,
        lu_id: UUID,
        condition: str,
        decision: str,
        confidence: float = 0.5,
        source_type: str = "human",
        actor: Optional[User] = None,
    ) -> LUConstraint:
        """添加约束"""
        lu = self.get_learning_unit_or_404(lu_id)
        
        constraint = LUConstraint(
            learning_unit_id=lu_id,
            condition=condition,
            decision=decision,
            confidence=confidence,
            source_type=source_type,
        )
        
        self.db.add(constraint)
        
        self._add_audit_history(
            lu_id,
            action="add_constraint",
            actor=actor,
            details={"condition": condition[:100], "decision": decision[:100]},
        )
        
        self.db.commit()
        self.db.refresh(constraint)
        
        return constraint
    
    def update_constraint(
        self,
        constraint_id: UUID,
        updates: Dict[str, Any],
        actor: Optional[User] = None,
    ) -> LUConstraint:
        """更新约束"""
        constraint = self.db.query(LUConstraint).filter(
            LUConstraint.id == constraint_id
        ).first()
        
        if not constraint:
            raise NotFoundError("LUConstraint", str(constraint_id))
        
        # 保存原始值
        if "condition" in updates and updates["condition"] != constraint.condition:
            constraint.original_condition = constraint.condition
            constraint.is_modified = True
        if "decision" in updates and updates["decision"] != constraint.decision:
            constraint.original_decision = constraint.decision
            constraint.is_modified = True
        
        for field, value in updates.items():
            if hasattr(constraint, field):
                setattr(constraint, field, value)
        
        self._add_audit_history(
            constraint.learning_unit_id,
            action="update_constraint",
            actor=actor,
            details={"constraint_id": str(constraint_id), "updates": updates},
        )
        
        self.db.commit()
        self.db.refresh(constraint)
        
        return constraint
    
    # ==================== 审计历史 ====================
    
    def get_audit_history(
        self,
        lu_id: UUID,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[LUAuditHistory], int]:
        """获取审计历史"""
        query = self.db.query(LUAuditHistory).filter(
            LUAuditHistory.learning_unit_id == lu_id
        )
        
        total = query.count()
        items = query.order_by(
            LUAuditHistory.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    # ==================== 统计 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取 Learning Unit 统计"""
        total = self.db.query(LearningUnit).count()
        
        by_status = dict(
            self.db.query(
                LearningUnit.status,
                self.db.query(LearningUnit).filter(
                    LearningUnit.status == LearningUnit.status
                ).count()
            ).group_by(LearningUnit.status).all()
        )
        
        by_risk = dict(
            self.db.query(
                LearningUnit.risk_level,
                self.db.query(LearningUnit).filter(
                    LearningUnit.risk_level == LearningUnit.risk_level
                ).count()
            ).filter(
                LearningUnit.risk_level.isnot(None)
            ).group_by(LearningUnit.risk_level).all()
        )
        
        pending_review = self.db.query(LearningUnit).filter(
            LearningUnit.requires_human_review == True,
            LearningUnit.status.in_([
                LearningUnitStatus.PENDING.value,
                LearningUnitStatus.AUTO_CLASSIFIED.value,
                LearningUnitStatus.HUMAN_REVIEW.value,
            ])
        ).count()
        
        internalized = self.db.query(LearningUnit).filter(
            LearningUnit.is_internalized == True
        ).count()
        
        return {
            "total": total,
            "by_status": by_status,
            "by_risk_level": by_risk,
            "pending_review": pending_review,
            "internalized": internalized,
        }
    
    # ==================== 私有方法 ====================
    
    def _add_audit_history(
        self,
        lu_id: UUID,
        action: str,
        actor: Optional[User] = None,
        from_status: Optional[str] = None,
        to_status: Optional[str] = None,
        comments: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        """添加审计历史记录"""
        history = LUAuditHistory(
            learning_unit_id=lu_id,
            action=action,
            actor_id=actor.id if actor else None,
            actor_name=actor.username if actor else "system",
            from_status=from_status,
            to_status=to_status,
            comments=comments,
            details=details or {},
        )
        self.db.add(history)
    
    def _can_approve(self, user: User, lu: LearningUnit) -> bool:
        """检查用户是否可以审批该 LU"""
        if user.is_superuser:
            return True
        
        max_risk = self._get_user_max_risk_level(user)
        if not max_risk:
            return False
        
        risk_order = ["low", "medium", "high", "critical"]
        if lu.risk_level not in risk_order:
            return True  # 无风险等级，允许审批
        
        if max_risk not in risk_order:
            return False
        
        return risk_order.index(lu.risk_level) <= risk_order.index(max_risk)
    
    def _get_user_max_risk_level(self, user: User) -> Optional[str]:
        """获取用户可审批的最高风险等级"""
        max_level = None
        risk_order = ["low", "medium", "high", "critical"]
        
        for role in user.roles:
            if role.risk_level_limit and role.risk_level_limit in risk_order:
                if max_level is None:
                    max_level = role.risk_level_limit
                elif risk_order.index(role.risk_level_limit) > risk_order.index(max_level):
                    max_level = role.risk_level_limit
        
        return max_level

