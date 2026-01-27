"""
知识转移服务

实现:
1. 审批通过的 Learning Unit 转移到 AGA
2. AGA 生命周期管理
3. 知识隔离和回滚
4. 与 Bridge 模块集成

架构说明:
    ApprovalService
         ↓ submit_lu_approval() 审批通过
    KnowledgeTransferService
         ↓ transfer_to_aga() 转移知识
    AGABridgeAdapter (本模块定义)
         ↓ write_learning_unit() 写入 AGA
    AGABridge (bridge 模块)
         ↓ inject_knowledge() 注入槽位
    AuxiliaryGovernedAttention (aga 核心)
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Tuple, Protocol
from uuid import UUID
from dataclasses import dataclass, field
import logging
import json

from sqlalchemy.orm import Session

from ..models.learning_unit import LearningUnit, LUConstraint
from ..models.user import User
from ..core.enums import LearningUnitStatus
from ..core.exceptions import NotFoundError, BusinessError


if TYPE_CHECKING:
    from .learning_unit_service import LearningUnitService


logger = logging.getLogger(__name__)


# ============================================================
# AGA 生命周期状态（与 AGA 模块对齐）
# ============================================================

class AGALifecycleState:
    """AGA 生命周期状态"""
    QUARANTINED = "quarantined"    # 已隔离 (r=0.0)
    PROBATIONARY = "probationary"  # 试用期 (r=0.3)
    CONFIRMED = "confirmed"        # 已确认 (r=1.0)
    DEPRECATED = "deprecated"      # 已弃用 (r=0.1)


# ============================================================
# Bridge 适配器数据类型（用于与 AGABridge 通信）
# ============================================================

@dataclass
class BridgeConstraint:
    """Bridge 约束格式"""
    condition: str
    decision: str
    confidence: float = 0.5


@dataclass
class BridgeLearningUnit:
    """Bridge Learning Unit 格式"""
    id: str
    proposed_constraints: List[BridgeConstraint]


@dataclass
class BridgeAuditApproval:
    """Bridge 审计批准格式"""
    approval_id: str
    decision: str = "approve"
    
    def verify(self) -> bool:
        """验证审计批准"""
        return self.decision == "approve"


class AGABridgeProtocol(Protocol):
    """AGA Bridge 协议接口"""
    
    def write_learning_unit(
        self,
        learning_unit: BridgeLearningUnit,
        writer_id: str,
        audit_approval: BridgeAuditApproval,
    ) -> Any:
        """写入 Learning Unit"""
        ...
    
    def confirm_learning_unit(self, lu_id: str) -> bool:
        """确认 Learning Unit"""
        ...
    
    def deprecate_learning_unit(self, lu_id: str) -> bool:
        """弃用 Learning Unit"""
        ...
    
    def quarantine_learning_unit(self, lu_id: str) -> bool:
        """隔离 Learning Unit"""
        ...
    
    def get_lu_status(self, lu_id: str) -> Optional[Dict[str, Any]]:
        """获取 LU 状态"""
        ...
    
    @property
    def lu_slot_mapping(self) -> Dict[str, Dict[int, int]]:
        """LU 到槽位的映射"""
        ...


class KnowledgeTransferService:
    """
    知识转移服务
    
    将审批通过的 Learning Unit 转移到 AGA 系统
    """
    
    def __init__(
        self,
        db: Session,
        lu_service: Optional["LearningUnitService"] = None,
        aga_bridge = None,  # AGABridge 实例
    ):
        self.db = db
        self._lu_service = lu_service
        self._aga_bridge = aga_bridge
        
        # 转移记录（内存）
        self._transfer_records: List[Dict[str, Any]] = []
    
    @property
    def lu_service(self) -> "LearningUnitService":
        """延迟加载 LearningUnitService"""
        if self._lu_service is None:
            from .learning_unit_service import LearningUnitService
            self._lu_service = LearningUnitService(self.db)
        return self._lu_service
    
    def set_aga_bridge(self, bridge):
        """设置 AGA Bridge 实例"""
        self._aga_bridge = bridge
    
    # ==================== 知识转移 ====================
    
    def transfer_to_aga(
        self,
        lu_id: UUID,
        initial_lifecycle: str = AGALifecycleState.PROBATIONARY,
    ) -> Dict[str, Any]:
        """
        将 Learning Unit 转移到 AGA
        
        Args:
            lu_id: Learning Unit ID
            initial_lifecycle: 初始生命周期状态
        
        Returns:
            转移结果
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        # 检查状态
        if lu.status not in [
            LearningUnitStatus.APPROVED.value,
            LearningUnitStatus.CORRECTED.value
        ]:
            raise BusinessError(
                f"Cannot transfer LU in status {lu.status}. Must be APPROVED or CORRECTED.",
                code="INVALID_STATUS"
            )
        
        if lu.is_internalized:
            raise BusinessError(
                f"LU {lu_id} is already internalized",
                code="ALREADY_INTERNALIZED"
            )
        
        # 获取约束
        constraints = self.lu_service.get_constraints(lu_id)
        if not constraints:
            raise BusinessError(
                f"LU {lu_id} has no constraints to transfer",
                code="NO_CONSTRAINTS"
            )
        
        # 执行转移
        transfer_result = {
            "lu_id": str(lu_id),
            "title": lu.title,
            "constraint_count": len(constraints),
            "initial_lifecycle": initial_lifecycle,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
            "aga_slot_mapping": {},
            "errors": [],
        }
        
        try:
            if self._aga_bridge:
                # 使用 AGA Bridge 进行转移
                slot_mapping = self._transfer_via_bridge(lu, constraints, initial_lifecycle)
                transfer_result["aga_slot_mapping"] = slot_mapping
                transfer_result["status"] = "success"
            else:
                # 模拟转移（无 Bridge 时）
                slot_mapping = self._simulate_transfer(lu, constraints, initial_lifecycle)
                transfer_result["aga_slot_mapping"] = slot_mapping
                transfer_result["status"] = "simulated"
                logger.warning(f"AGA Bridge not available, simulating transfer for LU {lu_id}")
            
            # 更新 LU 状态
            self.lu_service.mark_internalized(
                lu_id,
                aga_slot_mapping=slot_mapping,
                lifecycle_state=initial_lifecycle,
            )
            
        except Exception as e:
            transfer_result["status"] = "failed"
            transfer_result["errors"].append(str(e))
            logger.error(f"Transfer failed for LU {lu_id}: {e}")
            raise
        
        # 记录转移
        self._transfer_records.append(transfer_result)
        
        logger.info(f"Transferred LU {lu_id} to AGA with {len(constraints)} constraints")
        
        return transfer_result
    
    def _transfer_via_bridge(
        self,
        lu: LearningUnit,
        constraints: List[LUConstraint],
        initial_lifecycle: str,
    ) -> Dict[int, int]:
        """
        通过 AGA Bridge 转移知识
        
        将后端的 LUConstraint 模型转换为 Bridge 期望的数据格式，
        然后调用 Bridge 的 write_learning_unit 方法。
        
        Args:
            lu: 数据库中的 LearningUnit 模型
            constraints: 数据库中的 LUConstraint 列表
            initial_lifecycle: 初始生命周期状态
        
        Returns:
            layer_idx -> slot_idx 的映射
        """
        # 转换约束为 Bridge 格式
        bridge_constraints = [
            BridgeConstraint(
                condition=c.condition or "",
                decision=c.decision or "",
                confidence=c.confidence if c.confidence else 0.5,
            )
            for c in constraints
        ]
        
        # 构造 Bridge Learning Unit
        bridge_lu = BridgeLearningUnit(
            id=str(lu.id),
            proposed_constraints=bridge_constraints,
        )
        
        # 构造审计批准
        approval = BridgeAuditApproval(
            approval_id=lu.approval_id or "auto_approved",
            decision="approve",
        )
        
        # 调用 Bridge
        logger.info(f"Calling AGA Bridge for LU {lu.id} with {len(bridge_constraints)} constraints")
        
        try:
            result = self._aga_bridge.write_learning_unit(
                learning_unit=bridge_lu,
                writer_id="knowledge_transfer_service",
                audit_approval=approval,
            )
            logger.info(f"AGA Bridge returned: {result}")
        except Exception as e:
            logger.error(f"AGA Bridge write failed: {e}")
            raise
        
        # 获取槽位映射
        if hasattr(self._aga_bridge, 'lu_slot_mapping'):
            mapping = self._aga_bridge.lu_slot_mapping.get(str(lu.id), {})
            logger.info(f"Slot mapping for LU {lu.id}: {mapping}")
            return mapping
        
        return {}
    
    def _simulate_transfer(
        self,
        lu: LearningUnit,
        constraints: List[LUConstraint],
        initial_lifecycle: str,
    ) -> Dict[int, int]:
        """模拟转移（无 Bridge 时）"""
        # 模拟槽位分配
        slot_mapping = {}
        for i, constraint in enumerate(constraints):
            # 模拟分配到多层
            for layer_idx in [-4, -3, -2, -1]:
                if layer_idx not in slot_mapping:
                    slot_mapping[layer_idx] = i
        
        return slot_mapping
    
    # ==================== 生命周期管理 ====================
    
    def confirm_knowledge(self, lu_id: UUID, actor: Optional[User] = None) -> Dict[str, Any]:
        """
        确认知识（试用期 → 已确认）
        
        当知识经过验证后，提升其可靠性
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(f"LU {lu_id} is not internalized", code="NOT_INTERNALIZED")
        
        if lu.lifecycle_state == AGALifecycleState.CONFIRMED:
            raise BusinessError(f"LU {lu_id} is already confirmed", code="ALREADY_CONFIRMED")
        
        old_state = lu.lifecycle_state
        
        # 更新数据库
        self.lu_service.update_lifecycle_state(lu_id, AGALifecycleState.CONFIRMED, actor)
        
        # 更新 AGA Bridge
        if self._aga_bridge:
            try:
                self._aga_bridge.confirm_learning_unit(str(lu_id))
            except Exception as e:
                logger.error(f"Failed to confirm in AGA Bridge: {e}")
        
        logger.info(f"LU {lu_id} confirmed: {old_state} -> confirmed")
        
        return {
            "lu_id": str(lu_id),
            "old_state": old_state,
            "new_state": AGALifecycleState.CONFIRMED,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def deprecate_knowledge(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """
        弃用知识
        
        降低知识的可靠性，但不完全移除
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(f"LU {lu_id} is not internalized", code="NOT_INTERNALIZED")
        
        old_state = lu.lifecycle_state
        
        # 更新数据库
        self.lu_service.update_lifecycle_state(lu_id, AGALifecycleState.DEPRECATED, actor)
        
        # 更新 AGA Bridge
        if self._aga_bridge:
            try:
                self._aga_bridge.deprecate_learning_unit(str(lu_id))
            except Exception as e:
                logger.error(f"Failed to deprecate in AGA Bridge: {e}")
        
        logger.info(f"LU {lu_id} deprecated: {old_state} -> deprecated. Reason: {reason}")
        
        return {
            "lu_id": str(lu_id),
            "old_state": old_state,
            "new_state": AGALifecycleState.DEPRECATED,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def quarantine_knowledge(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """
        隔离知识（立即移除影响）
        
        当发现知识有问题时，立即隔离
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(f"LU {lu_id} is not internalized", code="NOT_INTERNALIZED")
        
        old_state = lu.lifecycle_state
        
        # 更新数据库
        self.lu_service.quarantine(lu_id, reason, actor)
        
        # 更新 AGA Bridge
        if self._aga_bridge:
            try:
                self._aga_bridge.quarantine_learning_unit(str(lu_id))
            except Exception as e:
                logger.error(f"Failed to quarantine in AGA Bridge: {e}")
        
        logger.warning(f"LU {lu_id} quarantined: {old_state} -> quarantined. Reason: {reason}")
        
        return {
            "lu_id": str(lu_id),
            "old_state": old_state,
            "new_state": AGALifecycleState.QUARANTINED,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def rollback_knowledge(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """
        回滚知识（等同于隔离）
        """
        return self.quarantine_knowledge(lu_id, f"Rollback: {reason}", actor)
    
    # ==================== 批量操作 ====================
    
    def batch_transfer(
        self,
        lu_ids: List[UUID],
        initial_lifecycle: str = AGALifecycleState.PROBATIONARY,
    ) -> Dict[str, Any]:
        """批量转移"""
        results = {
            "total": len(lu_ids),
            "success": 0,
            "failed": 0,
            "details": [],
        }
        
        for lu_id in lu_ids:
            try:
                result = self.transfer_to_aga(lu_id, initial_lifecycle)
                results["success"] += 1
                results["details"].append({
                    "lu_id": str(lu_id),
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "lu_id": str(lu_id),
                    "status": "failed",
                    "error": str(e),
                })
        
        return results
    
    def batch_confirm(self, lu_ids: List[UUID], actor: Optional[User] = None) -> Dict[str, Any]:
        """批量确认"""
        results = {"success": 0, "failed": 0, "details": []}
        
        for lu_id in lu_ids:
            try:
                result = self.confirm_knowledge(lu_id, actor)
                results["success"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "success"})
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "failed", "error": str(e)})
        
        return results
    
    def batch_quarantine(
        self,
        lu_ids: List[UUID],
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """批量隔离"""
        results = {"success": 0, "failed": 0, "details": []}
        
        for lu_id in lu_ids:
            try:
                result = self.quarantine_knowledge(lu_id, reason, actor)
                results["success"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "success"})
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "failed", "error": str(e)})
        
        return results
    
    # ==================== 查询接口 ====================
    
    def get_transfer_history(
        self,
        lu_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取转移历史"""
        history = self._transfer_records
        
        if lu_id:
            history = [r for r in history if r.get("lu_id") == str(lu_id)]
        
        if status:
            history = [r for r in history if r.get("status") == status]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_internalized_lus(
        self,
        lifecycle_state: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[LearningUnit], int]:
        """获取已内化的 Learning Units"""
        return self.lu_service.list_learning_units(
            is_internalized=True,
            skip=skip,
            limit=limit,
        )
    
    def get_aga_status(self, lu_id: UUID) -> Optional[Dict[str, Any]]:
        """获取 LU 在 AGA 中的状态"""
        lu = self.lu_service.get_learning_unit(lu_id)
        if not lu or not lu.is_internalized:
            return None
        
        status = {
            "lu_id": str(lu_id),
            "is_internalized": True,
            "internalized_at": lu.internalized_at.isoformat() if lu.internalized_at else None,
            "lifecycle_state": lu.lifecycle_state,
            "aga_slot_mapping": lu.aga_slot_mapping,
        }
        
        # 如果有 Bridge，获取更详细的状态
        if self._aga_bridge:
            try:
                bridge_status = self._aga_bridge.get_lu_status(str(lu_id))
                if bridge_status:
                    status["bridge_status"] = bridge_status
            except Exception as e:
                logger.error(f"Failed to get bridge status: {e}")
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 从数据库获取统计
        total_internalized = self.db.query(LearningUnit).filter(
            LearningUnit.is_internalized == True
        ).count()
        
        by_lifecycle = {}
        for state in [AGALifecycleState.PROBATIONARY, AGALifecycleState.CONFIRMED, 
                      AGALifecycleState.DEPRECATED, AGALifecycleState.QUARANTINED]:
            count = self.db.query(LearningUnit).filter(
                LearningUnit.is_internalized == True,
                LearningUnit.lifecycle_state == state,
            ).count()
            by_lifecycle[state] = count
        
        # 转移记录统计
        transfer_stats = {
            "total": len(self._transfer_records),
            "success": sum(1 for r in self._transfer_records if r["status"] == "success"),
            "failed": sum(1 for r in self._transfer_records if r["status"] == "failed"),
            "simulated": sum(1 for r in self._transfer_records if r["status"] == "simulated"),
        }
        
        return {
            "total_internalized": total_internalized,
            "by_lifecycle_state": by_lifecycle,
            "transfer_records": transfer_stats,
            "aga_bridge_available": self._aga_bridge is not None,
        }

