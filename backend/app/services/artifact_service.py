"""
Governed Artifact 管理服务

实现 NLGSM 论文中的 Governed Artifact 概念：
1. 工件创建和版本管理
2. 完整性哈希计算
3. 多签审批流程
4. 生命周期管理
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
import uuid
import hashlib
import json
import logging

from sqlalchemy.orm import Session

from ..models.artifact import Artifact, ArtifactApprover, ArtifactSnapshot, ApprovalCondition
from ..models.learning_unit import LearningUnit
from ..models.user import User
from ..core.enums import RiskLevel
from ..core.exceptions import NotFoundError, BusinessError, PermissionDeniedError


logger = logging.getLogger(__name__)


class GovernedArtifactService:
    """
    Governed Artifact 管理服务
    
    Governed Artifact 是 NLGSM 框架的核心概念：
    - 每个工件都有完整性哈希
    - 每个工件都需要经过审批流程
    - 每个工件都有完整的版本历史
    - 工件之间形成可追溯的链
    """
    
    def __init__(
        self,
        db: Session,
        notification_service=None,
    ):
        self.db = db
        self.notification_service = notification_service
        
        # 多签配置：不同风险等级需要的审批人数
        self.multi_sig_requirements = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
    
    # ==================== 工件创建 ====================
    
    def create_artifact(
        self,
        snapshot: Dict[str, Any],
        source_lu_id: Optional[UUID] = None,
        iteration: Optional[int] = None,
        nl_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        previous_artifact_id: Optional[UUID] = None,
    ) -> Artifact:
        """
        创建新的 Governed Artifact
        
        Args:
            snapshot: 工件快照数据
            source_lu_id: 来源 Learning Unit ID
            iteration: 迭代次数
            nl_state: NL 状态
            metrics: 指标
            previous_artifact_id: 前一个工件 ID（形成链）
            
        Returns:
            创建的工件
        """
        # 确定版本号
        version = 1
        if previous_artifact_id:
            prev = self.get_artifact(previous_artifact_id)
            if prev:
                version = prev.version + 1
        
        # 计算完整性哈希
        integrity_hash = self._calculate_integrity_hash({
            "snapshot": snapshot,
            "nl_state": nl_state or {},
            "version": version,
            "previous_artifact_id": str(previous_artifact_id) if previous_artifact_id else None,
        })
        
        artifact = Artifact(
            version=version,
            snapshot=snapshot,
            nl_state=nl_state or {},
            level_versions={},
            metrics=metrics or {},
            created_from_iteration=iteration,
            created_from_lu_id=source_lu_id,
            previous_artifact_id=previous_artifact_id,
            integrity_hash=integrity_hash,
            is_approved=False,
        )
        
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        
        logger.info(f"Created artifact {artifact.id} version {version}")
        
        return artifact
    
    def create_from_learning_unit(
        self,
        learning_unit: LearningUnit,
        nl_state: Optional[Dict[str, Any]] = None,
        previous_artifact_id: Optional[UUID] = None,
    ) -> Artifact:
        """从 Learning Unit 创建工件"""
        snapshot = {
            "knowledge": learning_unit.knowledge,
            "provenance": learning_unit.provenance,
            "learning_goal": learning_unit.learning_goal,
            "risk_level": learning_unit.risk_level,
            "risk_score": learning_unit.risk_score,
            "created_from_lu": str(learning_unit.id),
        }
        
        return self.create_artifact(
            snapshot=snapshot,
            source_lu_id=learning_unit.id,
            nl_state=nl_state,
            metrics={
                "risk_score": learning_unit.risk_score,
                "confidence": learning_unit.knowledge.get("confidence") if learning_unit.knowledge else None,
            },
            previous_artifact_id=previous_artifact_id,
        )
    
    # ==================== 工件查询 ====================
    
    def get_artifact(self, artifact_id: UUID) -> Optional[Artifact]:
        """获取工件"""
        return self.db.query(Artifact).filter(Artifact.id == artifact_id).first()
    
    def get_artifact_or_404(self, artifact_id: UUID) -> Artifact:
        """获取工件，不存在则抛出异常"""
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            raise NotFoundError("Artifact", str(artifact_id))
        return artifact
    
    def list_artifacts(
        self,
        is_approved: Optional[bool] = None,
        source_lu_id: Optional[UUID] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[Artifact], int]:
        """列出工件"""
        query = self.db.query(Artifact)
        
        if is_approved is not None:
            query = query.filter(Artifact.is_approved == is_approved)
        if source_lu_id:
            query = query.filter(Artifact.created_from_lu_id == source_lu_id)
        
        total = query.count()
        items = query.order_by(
            Artifact.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def get_artifact_chain(self, artifact_id: UUID) -> List[Artifact]:
        """获取工件链（从当前向前追溯）"""
        chain = []
        current = self.get_artifact(artifact_id)
        
        while current:
            chain.append(current)
            if current.previous_artifact_id:
                current = self.get_artifact(current.previous_artifact_id)
            else:
                break
        
        return chain
    
    def verify_integrity(self, artifact_id: UUID) -> Dict[str, Any]:
        """验证工件完整性"""
        artifact = self.get_artifact_or_404(artifact_id)
        
        # 重新计算哈希
        expected_hash = self._calculate_integrity_hash({
            "snapshot": artifact.snapshot,
            "nl_state": artifact.nl_state or {},
            "version": artifact.version,
            "previous_artifact_id": str(artifact.previous_artifact_id) if artifact.previous_artifact_id else None,
        })
        
        is_valid = artifact.integrity_hash == expected_hash
        
        # 验证链完整性
        chain_valid = True
        if artifact.previous_artifact_id:
            prev = self.get_artifact(artifact.previous_artifact_id)
            if prev:
                prev_result = self.verify_integrity(artifact.previous_artifact_id)
                chain_valid = prev_result["is_valid"]
        
        return {
            "artifact_id": str(artifact_id),
            "is_valid": is_valid and chain_valid,
            "hash_valid": is_valid,
            "chain_valid": chain_valid,
            "stored_hash": artifact.integrity_hash,
            "calculated_hash": expected_hash,
        }
    
    # ==================== 审批流程 ====================
    
    def initiate_approval(
        self,
        artifact_id: UUID,
        risk_level: RiskLevel,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        发起审批流程
        
        Args:
            artifact_id: 工件ID
            risk_level: 风险等级（决定需要的审批人数）
            conditions: 审批条件列表
            
        Returns:
            审批流程信息
        """
        artifact = self.get_artifact_or_404(artifact_id)
        
        if artifact.is_approved:
            raise BusinessError("Artifact already approved", code="ALREADY_APPROVED")
        
        required_approvers = self.multi_sig_requirements.get(risk_level, 1)
        
        # 记录审批条件
        if conditions:
            for condition_text in conditions:
                condition = ApprovalCondition(
                    artifact_id=artifact_id,
                    condition_text=condition_text,
                    is_met=False,
                )
                self.db.add(condition)
        
        artifact.risk_score = float(risk_level.value) if isinstance(risk_level.value, int) else 0.5
        
        self.db.commit()
        
        logger.info(
            f"Approval initiated for artifact {artifact_id}: "
            f"risk_level={risk_level.value}, required_approvers={required_approvers}"
        )
        
        return {
            "artifact_id": str(artifact_id),
            "risk_level": risk_level.value,
            "required_approvers": required_approvers,
            "current_approvers": 0,
            "conditions": conditions or [],
        }
    
    def approve(
        self,
        artifact_id: UUID,
        approver: User,
        comments: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        审批工件
        
        Args:
            artifact_id: 工件ID
            approver: 审批人
            comments: 审批意见
            
        Returns:
            审批结果
        """
        artifact = self.get_artifact_or_404(artifact_id)
        
        if artifact.is_approved:
            raise BusinessError("Artifact already approved", code="ALREADY_APPROVED")
        
        # 检查用户是否有权限审批
        risk_level = self._get_risk_level_from_score(artifact.risk_score)
        if not approver.can_approve_risk_level(risk_level.value):
            raise PermissionDeniedError(f"User cannot approve {risk_level.value} risk artifacts")
        
        # 检查是否已审批
        existing = self.db.query(ArtifactApprover).filter(
            ArtifactApprover.artifact_id == artifact_id,
            ArtifactApprover.approver_id == approver.id,
        ).first()
        
        if existing:
            raise BusinessError("User already approved this artifact", code="ALREADY_APPROVED_BY_USER")
        
        # 添加审批记录
        approval_record = ArtifactApprover(
            artifact_id=artifact_id,
            approver_id=approver.id,
            comments=comments,
        )
        self.db.add(approval_record)
        
        # 检查是否达到所需审批人数
        current_count = self.db.query(ArtifactApprover).filter(
            ArtifactApprover.artifact_id == artifact_id
        ).count() + 1  # +1 包含当前审批
        
        required_count = self.multi_sig_requirements.get(risk_level, 1)
        
        if current_count >= required_count:
            # 检查条件是否满足
            conditions = self.db.query(ApprovalCondition).filter(
                ApprovalCondition.artifact_id == artifact_id
            ).all()
            
            all_conditions_met = all(c.is_met for c in conditions)
            
            if all_conditions_met:
                artifact.is_approved = True
                artifact.approved_by = approver.id
                artifact.approval_timestamp = datetime.utcnow()
                
                logger.info(f"Artifact {artifact_id} approved with {current_count} approvers")
        
        self.db.commit()
        
        return {
            "artifact_id": str(artifact_id),
            "approver_id": str(approver.id),
            "current_approvers": current_count,
            "required_approvers": required_count,
            "is_approved": artifact.is_approved,
        }
    
    def verify_condition(
        self,
        artifact_id: UUID,
        condition_id: int,
        verifier: User,
    ) -> ApprovalCondition:
        """验证审批条件"""
        condition = self.db.query(ApprovalCondition).filter(
            ApprovalCondition.id == condition_id,
            ApprovalCondition.artifact_id == artifact_id,
        ).first()
        
        if not condition:
            raise NotFoundError("ApprovalCondition", str(condition_id))
        
        condition.is_met = True
        condition.verified_by = verifier.id
        condition.verified_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(condition)
        
        return condition
    
    def get_approval_status(self, artifact_id: UUID) -> Dict[str, Any]:
        """获取审批状态"""
        artifact = self.get_artifact_or_404(artifact_id)
        
        approvers = self.db.query(ArtifactApprover).filter(
            ArtifactApprover.artifact_id == artifact_id
        ).all()
        
        conditions = self.db.query(ApprovalCondition).filter(
            ApprovalCondition.artifact_id == artifact_id
        ).all()
        
        risk_level = self._get_risk_level_from_score(artifact.risk_score)
        required_count = self.multi_sig_requirements.get(risk_level, 1)
        
        return {
            "artifact_id": str(artifact_id),
            "is_approved": artifact.is_approved,
            "risk_level": risk_level.value,
            "required_approvers": required_count,
            "current_approvers": len(approvers),
            "approvers": [
                {
                    "user_id": str(a.approver_id),
                    "approved_at": a.approved_at.isoformat() if a.approved_at else None,
                    "comments": a.comments,
                }
                for a in approvers
            ],
            "conditions": [
                {
                    "id": c.id,
                    "text": c.condition_text,
                    "is_met": c.is_met,
                    "verified_by": str(c.verified_by) if c.verified_by else None,
                    "verified_at": c.verified_at.isoformat() if c.verified_at else None,
                }
                for c in conditions
            ],
        }
    
    # ==================== 辅助方法 ====================
    
    def _calculate_integrity_hash(self, data: Dict[str, Any]) -> str:
        """计算完整性哈希"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _get_risk_level_from_score(self, score: Optional[float]) -> RiskLevel:
        """从风险分数获取风险等级"""
        if score is None:
            return RiskLevel.LOW
        
        if score >= 0.8:
            return RiskLevel.CRITICAL
        elif score >= 0.6:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

