"""
事务管理服务

实现 NLGSM 论文中的事务性回滚机制：
1. 事务生命周期管理（创建、提交、回滚）
2. 快照管理（保存、恢复）
3. 原子性保证
4. 与状态机的集成
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from uuid import UUID
import uuid
import logging
import hashlib
import json

from sqlalchemy.orm import Session

from ..models.artifact import Artifact, ArtifactSnapshot
from ..models.transaction import Transaction
from ..core.enums import TransactionStatus, TransactionType
from ..core.exceptions import NotFoundError, BusinessError


logger = logging.getLogger(__name__)


class TransactionManager:
    """
    事务管理器
    
    提供 ACID 事务保证，支持：
    - 原子性：要么全部成功，要么全部回滚
    - 一致性：保持系统状态一致
    - 隔离性：并发事务互不干扰
    - 持久性：提交后持久保存
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._active_transactions: Dict[UUID, "TransactionContext"] = {}
    
    def begin(
        self,
        tx_type: TransactionType,
        target_artifact_id: Optional[UUID] = None,
        executor_id: Optional[UUID] = None,
        description: Optional[str] = None,
    ) -> "TransactionContext":
        """
        开始新事务
        
        Args:
            tx_type: 事务类型
            target_artifact_id: 目标工件ID
            executor_id: 执行者ID
            description: 事务描述
            
        Returns:
            事务上下文
        """
        tx_id = uuid.uuid4()
        
        # 创建事务记录
        transaction = Transaction(
            id=tx_id,
            tx_type=tx_type.value,
            target_artifact_id=target_artifact_id,
            status=TransactionStatus.PENDING.value,
            description=description,
            executed_by=executor_id,
        )
        
        self.db.add(transaction)
        self.db.flush()
        
        # 如果有目标工件，保存当前状态作为回滚点
        pre_state = None
        if target_artifact_id:
            artifact = self.db.query(Artifact).filter(
                Artifact.id == target_artifact_id
            ).first()
            if artifact:
                pre_state = self._capture_state(artifact)
                transaction.pre_state = pre_state
        
        # 创建事务上下文
        context = TransactionContext(
            transaction_id=tx_id,
            tx_type=tx_type,
            manager=self,
            pre_state=pre_state,
        )
        
        self._active_transactions[tx_id] = context
        
        logger.info(f"Transaction {tx_id} started: type={tx_type.value}")
        
        return context
    
    def commit(self, tx_id: UUID) -> Transaction:
        """
        提交事务
        
        Args:
            tx_id: 事务ID
            
        Returns:
            事务记录
        """
        context = self._active_transactions.get(tx_id)
        if not context:
            raise BusinessError(f"Transaction {tx_id} not found in active transactions", code="TX_NOT_FOUND")
        
        transaction = self.db.query(Transaction).filter(Transaction.id == tx_id).first()
        if not transaction:
            raise NotFoundError("Transaction", str(tx_id))
        
        if transaction.status != TransactionStatus.PENDING.value:
            raise BusinessError(f"Transaction {tx_id} cannot be committed: status={transaction.status}", code="INVALID_STATUS")
        
        try:
            # 执行所有挂起的操作
            for operation in context.operations:
                operation()
            
            # 更新事务状态
            transaction.status = TransactionStatus.COMMITTED.value
            transaction.completed_at = datetime.utcnow()
            
            # 保存最终状态
            if transaction.target_artifact_id:
                artifact = self.db.query(Artifact).filter(
                    Artifact.id == transaction.target_artifact_id
                ).first()
                if artifact:
                    transaction.post_state = self._capture_state(artifact)
            
            transaction.success = True
            self.db.commit()
            
            # 从活跃事务中移除
            del self._active_transactions[tx_id]
            
            logger.info(f"Transaction {tx_id} committed successfully")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Transaction {tx_id} commit failed: {e}")
            # 自动回滚
            self.rollback(tx_id, str(e))
            raise
    
    def rollback(self, tx_id: UUID, reason: Optional[str] = None) -> Transaction:
        """
        回滚事务
        
        Args:
            tx_id: 事务ID
            reason: 回滚原因
            
        Returns:
            事务记录
        """
        context = self._active_transactions.get(tx_id)
        
        transaction = self.db.query(Transaction).filter(Transaction.id == tx_id).first()
        if not transaction:
            raise NotFoundError("Transaction", str(tx_id))
        
        # 恢复到之前状态
        if transaction.pre_state and transaction.target_artifact_id:
            artifact = self.db.query(Artifact).filter(
                Artifact.id == transaction.target_artifact_id
            ).first()
            if artifact:
                self._restore_state(artifact, transaction.pre_state)
        
        # 更新事务状态
        transaction.status = TransactionStatus.ABORTED.value
        transaction.completed_at = datetime.utcnow()
        transaction.success = False
        transaction.rollback_reason = reason
        
        self.db.commit()
        
        # 从活跃事务中移除
        if tx_id in self._active_transactions:
            del self._active_transactions[tx_id]
        
        logger.warning(f"Transaction {tx_id} rolled back: {reason}")
        
        return transaction
    
    def get_transaction(self, tx_id: UUID) -> Optional[Transaction]:
        """获取事务"""
        return self.db.query(Transaction).filter(Transaction.id == tx_id).first()
    
    def list_transactions(
        self,
        status: Optional[str] = None,
        tx_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[Transaction], int]:
        """列出事务"""
        query = self.db.query(Transaction)
        
        if status:
            query = query.filter(Transaction.status == status)
        if tx_type:
            query = query.filter(Transaction.tx_type == tx_type)
        
        total = query.count()
        items = query.order_by(
            Transaction.started_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def _capture_state(self, artifact: Artifact) -> Dict[str, Any]:
        """捕获工件状态"""
        return {
            "version": artifact.version,
            "snapshot": artifact.snapshot,
            "nl_state": artifact.nl_state,
            "level_versions": artifact.level_versions,
            "metrics": artifact.metrics,
            "captured_at": datetime.utcnow().isoformat(),
        }
    
    def _restore_state(self, artifact: Artifact, state: Dict[str, Any]):
        """恢复工件状态"""
        artifact.snapshot = state.get("snapshot", {})
        artifact.nl_state = state.get("nl_state", {})
        artifact.level_versions = state.get("level_versions", {})
        artifact.metrics = state.get("metrics", {})
        # 注意：版本号不回滚，以保持历史可追溯性
        logger.info(f"Artifact {artifact.id} restored to state from {state.get('captured_at')}")


class TransactionContext:
    """
    事务上下文
    
    提供事务操作的上下文管理，支持 with 语句
    """
    
    def __init__(
        self,
        transaction_id: UUID,
        tx_type: TransactionType,
        manager: TransactionManager,
        pre_state: Optional[Dict[str, Any]] = None,
    ):
        self.transaction_id = transaction_id
        self.tx_type = tx_type
        self.manager = manager
        self.pre_state = pre_state
        self.operations: List[Callable] = []
        self.committed = False
        self.rolled_back = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # 发生异常，回滚
            if not self.rolled_back:
                self.rollback(str(exc_val))
            return False  # 不抑制异常
        else:
            # 正常退出，提交
            if not self.committed and not self.rolled_back:
                self.commit()
        return True
    
    def add_operation(self, operation: Callable):
        """添加事务操作"""
        self.operations.append(operation)
    
    def commit(self):
        """提交事务"""
        if self.committed or self.rolled_back:
            return
        
        self.manager.commit(self.transaction_id)
        self.committed = True
    
    def rollback(self, reason: Optional[str] = None):
        """回滚事务"""
        if self.committed or self.rolled_back:
            return
        
        self.manager.rollback(self.transaction_id, reason)
        self.rolled_back = True


class SnapshotManager:
    """
    快照管理器
    
    管理工件快照，支持增量和完整快照
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_snapshot(
        self,
        artifact: Artifact,
        snapshot_type: str = "full",
        description: Optional[str] = None,
    ) -> ArtifactSnapshot:
        """
        创建快照
        
        Args:
            artifact: 目标工件
            snapshot_type: 快照类型 (full, incremental)
            description: 快照描述
            
        Returns:
            创建的快照
        """
        # 计算快照数据
        snapshot_data = {
            "artifact_id": str(artifact.id),
            "version": artifact.version,
            "snapshot": artifact.snapshot,
            "nl_state": artifact.nl_state,
            "level_versions": artifact.level_versions,
            "metrics": artifact.metrics,
            "integrity_hash": artifact.integrity_hash,
            "description": description,
        }
        
        # 计算快照哈希
        snapshot_hash = self._calculate_hash(snapshot_data)
        snapshot_data["snapshot_hash"] = snapshot_hash
        
        snapshot = ArtifactSnapshot(
            artifact_id=artifact.id,
            snapshot_data=snapshot_data,
            snapshot_type=snapshot_type,
        )
        
        self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(snapshot)
        
        logger.info(f"Snapshot created for artifact {artifact.id}: type={snapshot_type}")
        
        return snapshot
    
    def restore_snapshot(self, snapshot_id: int) -> Artifact:
        """
        从快照恢复
        
        Args:
            snapshot_id: 快照ID
            
        Returns:
            恢复后的工件
        """
        snapshot = self.db.query(ArtifactSnapshot).filter(
            ArtifactSnapshot.id == snapshot_id
        ).first()
        
        if not snapshot:
            raise NotFoundError("ArtifactSnapshot", str(snapshot_id))
        
        artifact = self.db.query(Artifact).filter(
            Artifact.id == snapshot.artifact_id
        ).first()
        
        if not artifact:
            raise NotFoundError("Artifact", str(snapshot.artifact_id))
        
        # 验证快照完整性
        data = snapshot.snapshot_data
        expected_hash = data.get("snapshot_hash")
        data_copy = {k: v for k, v in data.items() if k != "snapshot_hash"}
        actual_hash = self._calculate_hash(data_copy)
        
        if expected_hash and expected_hash != actual_hash:
            raise BusinessError("Snapshot integrity check failed", code="INTEGRITY_ERROR")
        
        # 恢复状态
        artifact.snapshot = data.get("snapshot", {})
        artifact.nl_state = data.get("nl_state", {})
        artifact.level_versions = data.get("level_versions", {})
        artifact.metrics = data.get("metrics", {})
        
        self.db.commit()
        self.db.refresh(artifact)
        
        logger.info(f"Artifact {artifact.id} restored from snapshot {snapshot_id}")
        
        return artifact
    
    def list_snapshots(
        self,
        artifact_id: UUID,
        skip: int = 0,
        limit: int = 20,
    ) -> tuple[List[ArtifactSnapshot], int]:
        """列出工件快照"""
        query = self.db.query(ArtifactSnapshot).filter(
            ArtifactSnapshot.artifact_id == artifact_id
        )
        
        total = query.count()
        items = query.order_by(
            ArtifactSnapshot.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """计算数据哈希"""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


class RollbackService:
    """
    回滚服务
    
    提供高级回滚功能，支持：
    - 回滚到指定检查点
    - 回滚到指定快照
    - 回滚到指定版本
    - 级联回滚
    """
    
    def __init__(
        self,
        db: Session,
        transaction_manager: TransactionManager,
        snapshot_manager: SnapshotManager,
    ):
        self.db = db
        self.tx_manager = transaction_manager
        self.snapshot_manager = snapshot_manager
    
    def rollback_to_snapshot(
        self,
        artifact_id: UUID,
        snapshot_id: int,
        executor_id: UUID,
        reason: str,
    ) -> Dict[str, Any]:
        """
        回滚到指定快照
        
        Args:
            artifact_id: 工件ID
            snapshot_id: 快照ID
            executor_id: 执行者ID
            reason: 回滚原因
            
        Returns:
            回滚结果
        """
        with self.tx_manager.begin(
            TransactionType.ROLLBACK,
            target_artifact_id=artifact_id,
            executor_id=executor_id,
            description=f"Rollback to snapshot {snapshot_id}: {reason}",
        ) as tx:
            # 恢复快照
            artifact = self.snapshot_manager.restore_snapshot(snapshot_id)
            
            # 增加版本号
            artifact.version += 1
            
            # 创建新快照记录回滚点
            self.snapshot_manager.create_snapshot(
                artifact,
                snapshot_type="rollback",
                description=f"Post-rollback snapshot: {reason}",
            )
            
            self.db.commit()
            
            return {
                "success": True,
                "artifact_id": str(artifact_id),
                "snapshot_id": snapshot_id,
                "new_version": artifact.version,
                "transaction_id": str(tx.transaction_id),
            }
    
    def rollback_to_version(
        self,
        artifact_id: UUID,
        target_version: int,
        executor_id: UUID,
        reason: str,
    ) -> Dict[str, Any]:
        """
        回滚到指定版本
        
        Args:
            artifact_id: 工件ID
            target_version: 目标版本
            executor_id: 执行者ID
            reason: 回滚原因
            
        Returns:
            回滚结果
        """
        # 查找对应版本的快照
        snapshot = self.db.query(ArtifactSnapshot).filter(
            ArtifactSnapshot.artifact_id == artifact_id,
        ).filter(
            ArtifactSnapshot.snapshot_data["version"].astext.cast(int) == target_version
        ).first()
        
        if not snapshot:
            raise NotFoundError(f"Snapshot for version {target_version}", str(artifact_id))
        
        return self.rollback_to_snapshot(
            artifact_id=artifact_id,
            snapshot_id=snapshot.id,
            executor_id=executor_id,
            reason=reason,
        )
    
    def get_rollback_history(
        self,
        artifact_id: UUID,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """获取回滚历史"""
        transactions, _ = self.tx_manager.list_transactions(
            tx_type=TransactionType.ROLLBACK.value,
            limit=limit,
        )
        
        # 过滤目标工件
        result = []
        for tx in transactions:
            if tx.target_artifact_id == artifact_id:
                result.append({
                    "transaction_id": str(tx.id),
                    "status": tx.status,
                    "reason": tx.rollback_reason,
                    "started_at": tx.started_at.isoformat() if tx.started_at else None,
                    "completed_at": tx.completed_at.isoformat() if tx.completed_at else None,
                })
        
        return result

