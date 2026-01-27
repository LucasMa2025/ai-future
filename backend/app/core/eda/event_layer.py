"""
EDA 事件层

负责事件的感知、路由和分发
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Set
from uuid import UUID, uuid4
import asyncio
import logging
from collections import defaultdict
import threading


logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """事件类型"""
    # 学习相关
    LEARNING_STARTED = "learning_started"
    LEARNING_STEP_COMPLETED = "learning_step_completed"
    LEARNING_PAUSED = "learning_paused"
    LEARNING_RESUMED = "learning_resumed"
    LEARNING_TERMINATED = "learning_terminated"
    
    # LU 相关
    LU_CREATED = "lu_created"
    LU_SUBMITTED = "lu_submitted"
    LU_CLASSIFIED = "lu_classified"
    LU_APPROVED = "lu_approved"
    LU_REJECTED = "lu_rejected"
    LU_CORRECTED = "lu_corrected"
    LU_INTERNALIZED = "lu_internalized"
    
    # 状态机相关
    STATE_CHANGED = "state_changed"
    STATE_TRANSITION_FAILED = "state_transition_failed"
    
    # 异常相关
    ANOMALY_DETECTED = "anomaly_detected"
    ANOMALY_RESOLVED = "anomaly_resolved"
    
    # 审批相关
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_COMPLETED = "approval_completed"
    APPROVAL_TIMEOUT = "approval_timeout"
    
    # 系统相关
    SYSTEM_HEALTH_CHECK = "system_health_check"
    CHECKPOINT_CREATED = "checkpoint_created"
    ROLLBACK_INITIATED = "rollback_initiated"
    ROLLBACK_COMPLETED = "rollback_completed"
    
    # 诊断恢复
    DIAGNOSIS_STARTED = "diagnosis_started"
    DIAGNOSIS_COMPLETED = "diagnosis_completed"
    RECOVERY_PLAN_CREATED = "recovery_plan_created"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"


class EventPriority(int, Enum):
    """事件优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Event:
    """事件"""
    id: UUID = field(default_factory=uuid4)
    event_type: EventType = EventType.SYSTEM_HEALTH_CHECK
    priority: EventPriority = EventPriority.NORMAL
    
    # 来源
    source: str = ""
    source_id: Optional[UUID] = None
    
    # 数据
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    correlation_id: Optional[UUID] = None  # 关联多个事件
    causation_id: Optional[UUID] = None    # 因果链追踪
    
    # 时间
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 处理状态
    processed: bool = False
    processing_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "event_type": self.event_type.value,
            "priority": self.priority.value,
            "source": self.source,
            "source_id": str(self.source_id) if self.source_id else None,
            "payload": self.payload,
            "correlation_id": str(self.correlation_id) if self.correlation_id else None,
            "causation_id": str(self.causation_id) if self.causation_id else None,
            "timestamp": self.timestamp.isoformat(),
            "processed": self.processed,
        }


class EventHandler:
    """事件处理器基类"""
    
    def __init__(self, name: str, event_types: List[EventType]):
        self.name = name
        self.event_types = event_types
    
    def can_handle(self, event: Event) -> bool:
        """检查是否可以处理此事件"""
        return event.event_type in self.event_types
    
    def handle(self, event: Event) -> Dict[str, Any]:
        """处理事件"""
        raise NotImplementedError


class EventBus:
    """
    事件总线
    
    负责事件的发布、订阅和分发
    支持同步和异步处理
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._async_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_queue: asyncio.Queue = None
        self._event_history: List[Event] = []
        self._max_history_size: int = 1000
        self._lock = threading.Lock()
        
        # 事件过滤器
        self._filters: List[Callable[[Event], bool]] = []
        
        # 事件转换器
        self._transformers: List[Callable[[Event], Event]] = []
    
    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ):
        """订阅事件"""
        with self._lock:
            self._handlers[event_type].append(handler)
            logger.debug(f"Handler {handler.name} subscribed to {event_type.value}")
    
    def subscribe_async(
        self,
        event_type: EventType,
        callback: Callable,
    ):
        """订阅异步事件"""
        with self._lock:
            self._async_handlers[event_type].append(callback)
    
    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
    ):
        """取消订阅"""
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
    
    def add_filter(self, filter_fn: Callable[[Event], bool]):
        """添加事件过滤器"""
        self._filters.append(filter_fn)
    
    def add_transformer(self, transformer_fn: Callable[[Event], Event]):
        """添加事件转换器"""
        self._transformers.append(transformer_fn)
    
    def publish(self, event: Event) -> List[Dict[str, Any]]:
        """
        同步发布事件
        
        Args:
            event: 事件
            
        Returns:
            处理结果列表
        """
        # 应用过滤器
        for filter_fn in self._filters:
            if not filter_fn(event):
                logger.debug(f"Event {event.id} filtered out")
                return []
        
        # 应用转换器
        for transformer in self._transformers:
            event = transformer(event)
        
        # 记录事件
        self._record_event(event)
        
        # 获取处理器
        handlers = self._handlers.get(event.event_type, [])
        
        results = []
        for handler in handlers:
            try:
                result = handler.handle(event)
                results.append({
                    "handler": handler.name,
                    "success": True,
                    "result": result,
                })
            except Exception as e:
                logger.error(f"Handler {handler.name} failed: {e}")
                results.append({
                    "handler": handler.name,
                    "success": False,
                    "error": str(e),
                })
        
        event.processed = True
        event.processing_results = results
        
        logger.info(f"Event {event.event_type.value} published, {len(results)} handlers executed")
        
        return results
    
    async def publish_async(self, event: Event):
        """异步发布事件"""
        # 应用过滤器
        for filter_fn in self._filters:
            if not filter_fn(event):
                return
        
        # 应用转换器
        for transformer in self._transformers:
            event = transformer(event)
        
        # 记录事件
        self._record_event(event)
        
        # 获取异步处理器
        handlers = self._async_handlers.get(event.event_type, [])
        
        tasks = [handler(event) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        event.processed = True
    
    def _record_event(self, event: Event):
        """记录事件历史"""
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history_size:
                self._event_history = self._event_history[-self._max_history_size // 2:]
    
    def get_event_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
    ) -> List[Event]:
        """获取事件历史"""
        history = self._event_history
        
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        
        return sorted(history, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        event_counts = defaultdict(int)
        for event in self._event_history:
            event_counts[event.event_type.value] += 1
        
        return {
            "total_events": len(self._event_history),
            "event_counts": dict(event_counts),
            "handlers_count": sum(len(h) for h in self._handlers.values()),
            "async_handlers_count": sum(len(h) for h in self._async_handlers.values()),
        }


# 全局事件总线实例
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """获取全局事件总线"""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus

