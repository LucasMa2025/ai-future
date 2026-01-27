"""
Learning Unit 状态管理

实现 Learning Unit 状态的共享和异步通知机制。

设计原则：
- Learning Unit 本身带有状态
- 状态由 NLGSM 治理系统管理
- 自学习系统通过异步通知获取状态变更
- 自学习系统根据状态决定下一步操作

状态流转：
- PENDING -> AUTO_CLASSIFIED -> HUMAN_REVIEW -> APPROVED/REJECTED/CORRECTED
- APPROVED -> 可继续学习或开始新学习
- REJECTED -> 放弃 LU，开始新学习
- CORRECTED -> 调整策略，继续学习
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from threading import Lock, Event
from queue import Queue, Empty
import uuid
import logging
import json

logger = logging.getLogger(__name__)


class LUStatus(Enum):
    """Learning Unit 状态"""
    PENDING = "pending"
    AUTO_CLASSIFIED = "auto_classified"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    CORRECTED = "corrected"
    TERMINATED = "terminated"
    INTERNALIZED = "internalized"


class LUDecision(Enum):
    """治理决策"""
    CONTINUE = "continue"           # 继续学习（在此 LU 基础上）
    NEW_LEARNING = "new_learning"   # 开始新学习
    ADJUST = "adjust"               # 调整策略后继续
    STOP = "stop"                   # 停止学习


@dataclass
class LUStateChange:
    """状态变更事件"""
    lu_id: str
    old_status: Optional[LUStatus]
    new_status: LUStatus
    decision: Optional[LUDecision] = None
    reason: str = ""
    changed_by: str = "system"
    changed_at: datetime = field(default_factory=datetime.now)
    
    # 调整参数（如果 decision == ADJUST）
    adjustment_params: Dict[str, Any] = field(default_factory=dict)
    
    # 继续学习参数（如果 decision == CONTINUE）
    continue_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lu_id": self.lu_id,
            "old_status": self.old_status.value if self.old_status else None,
            "new_status": self.new_status.value,
            "decision": self.decision.value if self.decision else None,
            "reason": self.reason,
            "changed_by": self.changed_by,
            "changed_at": self.changed_at.isoformat(),
            "adjustment_params": self.adjustment_params,
            "continue_params": self.continue_params,
        }


@dataclass
class SharedLearningUnit:
    """
    共享的 Learning Unit
    
    状态由 NLGSM 治理系统管理，自学习系统可读取。
    """
    id: str
    title: str
    learning_goal: str
    
    # 状态（由治理系统管理）
    status: LUStatus = LUStatus.PENDING
    
    # 知识内容
    knowledge: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    
    # 链式学习信息
    parent_lu_id: Optional[str] = None
    chain_depth: int = 0
    chain_root_id: Optional[str] = None
    
    # 风险信息（由治理系统设置）
    risk_level: Optional[str] = None
    risk_score: float = 0.0
    
    # 审批信息
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # 调整信息（如果被调整）
    adjustment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 内化状态
    is_internalized: bool = False
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "learning_goal": self.learning_goal,
            "status": self.status.value,
            "knowledge": self.knowledge,
            "constraints": self.constraints,
            "parent_lu_id": self.parent_lu_id,
            "chain_depth": self.chain_depth,
            "chain_root_id": self.chain_root_id,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "adjustment_history": self.adjustment_history,
            "is_internalized": self.is_internalized,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def can_continue_learning(self) -> bool:
        """是否可以继续学习"""
        return (
            self.status in [LUStatus.APPROVED, LUStatus.CORRECTED, LUStatus.INTERNALIZED]
            and self.is_internalized
        )
    
    def get_context_for_continuation(self) -> Dict[str, Any]:
        """获取继续学习的上下文"""
        return {
            "parent_lu_id": self.id,
            "parent_title": self.title,
            "parent_goal": self.learning_goal,
            "parent_knowledge": self.knowledge,
            "parent_constraints": self.constraints,
            "chain_depth": self.chain_depth + 1,
            "chain_root_id": self.chain_root_id or self.id,
        }


class LUStateManager:
    """
    Learning Unit 状态管理器
    
    核心功能：
    1. 管理 LU 状态
    2. 发送状态变更通知
    3. 支持订阅/取消订阅
    
    使用方式：
    - NLGSM 治理系统调用 update_status() 更新状态
    - 自学习系统订阅状态变更通知
    """
    
    def __init__(self):
        self._lus: Dict[str, SharedLearningUnit] = {}
        self._lock = Lock()
        
        # 订阅者
        self._subscribers: Dict[str, Callable[[LUStateChange], None]] = {}
        
        # 状态变更历史
        self._state_history: List[LUStateChange] = []
        
        # 待处理的状态变更队列（用于异步通知）
        self._change_queue: Queue = Queue()
    
    def register_lu(self, lu: SharedLearningUnit) -> None:
        """注册 Learning Unit"""
        with self._lock:
            self._lus[lu.id] = lu
            logger.info(f"LU registered: {lu.id}")
    
    def get_lu(self, lu_id: str) -> Optional[SharedLearningUnit]:
        """获取 Learning Unit"""
        with self._lock:
            return self._lus.get(lu_id)
    
    def update_status(
        self,
        lu_id: str,
        new_status: LUStatus,
        decision: Optional[LUDecision] = None,
        reason: str = "",
        changed_by: str = "system",
        adjustment_params: Optional[Dict[str, Any]] = None,
        continue_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[LUStateChange]:
        """
        更新 LU 状态（由治理系统调用）
        
        Args:
            lu_id: Learning Unit ID
            new_status: 新状态
            decision: 治理决策
            reason: 变更原因
            changed_by: 变更人
            adjustment_params: 调整参数
            continue_params: 继续学习参数
            
        Returns:
            状态变更事件
        """
        with self._lock:
            lu = self._lus.get(lu_id)
            if not lu:
                logger.warning(f"LU not found: {lu_id}")
                return None
            
            old_status = lu.status
            lu.status = new_status
            lu.updated_at = datetime.now()
            
            # 如果是审批通过
            if new_status == LUStatus.APPROVED:
                lu.approved_at = datetime.now()
                lu.approved_by = changed_by
            
            # 如果是调整
            if decision == LUDecision.ADJUST and adjustment_params:
                lu.adjustment_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "params": adjustment_params,
                    "reason": reason,
                })
            
            # 创建状态变更事件
            change = LUStateChange(
                lu_id=lu_id,
                old_status=old_status,
                new_status=new_status,
                decision=decision,
                reason=reason,
                changed_by=changed_by,
                adjustment_params=adjustment_params or {},
                continue_params=continue_params or {},
            )
            
            self._state_history.append(change)
            
            # 放入队列用于异步通知
            self._change_queue.put(change)
        
        # 通知订阅者（在锁外执行）
        self._notify_subscribers(change)
        
        logger.info(f"LU status updated: {lu_id} {old_status.value} -> {new_status.value}")
        
        return change
    
    def subscribe(
        self,
        subscriber_id: str,
        callback: Callable[[LUStateChange], None],
    ) -> None:
        """
        订阅状态变更通知
        
        Args:
            subscriber_id: 订阅者 ID
            callback: 回调函数
        """
        with self._lock:
            self._subscribers[subscriber_id] = callback
            logger.info(f"Subscriber added: {subscriber_id}")
    
    def unsubscribe(self, subscriber_id: str) -> None:
        """取消订阅"""
        with self._lock:
            if subscriber_id in self._subscribers:
                del self._subscribers[subscriber_id]
                logger.info(f"Subscriber removed: {subscriber_id}")
    
    def _notify_subscribers(self, change: LUStateChange) -> None:
        """通知所有订阅者"""
        with self._lock:
            subscribers = list(self._subscribers.items())
        
        for subscriber_id, callback in subscribers:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Subscriber {subscriber_id} callback failed: {e}")
    
    def get_pending_changes(self, timeout: float = 0.1) -> List[LUStateChange]:
        """获取待处理的状态变更（用于轮询模式）"""
        changes = []
        while True:
            try:
                change = self._change_queue.get(timeout=timeout)
                changes.append(change)
            except Empty:
                break
        return changes
    
    def get_continuable_lus(self, max_chain_depth: int = 5) -> List[SharedLearningUnit]:
        """获取可继续学习的 LU"""
        with self._lock:
            return [
                lu for lu in self._lus.values()
                if lu.can_continue_learning()
                and lu.chain_depth < max_chain_depth
            ]
    
    def get_lus_by_status(self, status: LUStatus) -> List[SharedLearningUnit]:
        """按状态获取 LU"""
        with self._lock:
            return [lu for lu in self._lus.values() if lu.status == status]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            status_counts = {}
            for lu in self._lus.values():
                status = lu.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                "total_lus": len(self._lus),
                "by_status": status_counts,
                "total_changes": len(self._state_history),
                "subscribers": len(self._subscribers),
            }


class SelfLearningStateHandler:
    """
    自学习系统状态处理器
    
    处理来自治理系统的状态变更通知，决定下一步操作。
    """
    
    def __init__(
        self,
        state_manager: LUStateManager,
        learner_id: str = "default",
    ):
        self.state_manager = state_manager
        self.learner_id = learner_id
        
        # 当前正在处理的 LU
        self.current_lu_id: Optional[str] = None
        
        # 待处理的决策
        self.pending_decisions: Queue = Queue()
        
        # 注册订阅
        self.state_manager.subscribe(
            f"learner_{learner_id}",
            self._on_state_change,
        )
    
    def _on_state_change(self, change: LUStateChange) -> None:
        """
        处理状态变更通知
        
        根据状态和决策决定下一步操作：
        - APPROVED + CONTINUE: 继续学习
        - APPROVED + NEW_LEARNING: 开始新学习
        - REJECTED + NEW_LEARNING: 放弃 LU，开始新学习
        - CORRECTED + ADJUST: 调整策略后继续
        """
        # 只处理与当前 LU 相关的变更
        if self.current_lu_id and change.lu_id != self.current_lu_id:
            return
        
        logger.info(f"[Learner {self.learner_id}] Received state change: {change.lu_id} -> {change.new_status.value}")
        
        # 将决策放入队列
        self.pending_decisions.put(change)
    
    def wait_for_decision(self, timeout: float = None) -> Optional[LUStateChange]:
        """
        等待治理决策
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
            
        Returns:
            状态变更事件，超时返回 None
        """
        try:
            return self.pending_decisions.get(timeout=timeout)
        except Empty:
            return None
    
    def get_next_action(self, change: LUStateChange) -> Dict[str, Any]:
        """
        根据状态变更获取下一步操作
        
        Returns:
            {
                "action": "continue" | "new_learning" | "adjust" | "stop",
                "params": {...}
            }
        """
        if change.decision == LUDecision.CONTINUE:
            lu = self.state_manager.get_lu(change.lu_id)
            if lu and lu.can_continue_learning():
                return {
                    "action": "continue",
                    "params": {
                        **lu.get_context_for_continuation(),
                        **change.continue_params,
                    },
                }
        
        elif change.decision == LUDecision.NEW_LEARNING:
            return {
                "action": "new_learning",
                "params": change.continue_params,
            }
        
        elif change.decision == LUDecision.ADJUST:
            return {
                "action": "adjust",
                "params": change.adjustment_params,
            }
        
        elif change.decision == LUDecision.STOP:
            return {
                "action": "stop",
                "params": {"reason": change.reason},
            }
        
        # 默认根据状态决定
        if change.new_status == LUStatus.APPROVED:
            return {"action": "continue", "params": {}}
        elif change.new_status == LUStatus.REJECTED:
            return {"action": "new_learning", "params": {}}
        elif change.new_status == LUStatus.CORRECTED:
            return {"action": "adjust", "params": change.adjustment_params}
        elif change.new_status == LUStatus.TERMINATED:
            return {"action": "stop", "params": {}}
        
        return {"action": "wait", "params": {}}
    
    def set_current_lu(self, lu_id: str) -> None:
        """设置当前正在处理的 LU"""
        self.current_lu_id = lu_id
    
    def clear_current_lu(self) -> None:
        """清除当前 LU"""
        self.current_lu_id = None
    
    def cleanup(self) -> None:
        """清理资源"""
        self.state_manager.unsubscribe(f"learner_{self.learner_id}")

