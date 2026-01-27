"""
追踪系统

提供分布式追踪支持
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from enum import Enum
import threading
import logging
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class SpanStatus(str, Enum):
    """Span 状态"""
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


@dataclass
class SpanContext:
    """Span 上下文"""
    trace_id: UUID = field(default_factory=uuid4)
    span_id: UUID = field(default_factory=uuid4)
    parent_span_id: Optional[UUID] = None
    
    # 采样
    sampled: bool = True
    
    # Baggage（跨服务传递的键值对）
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """追踪 Span"""
    name: str = ""
    context: SpanContext = field(default_factory=SpanContext)
    
    # 时间
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # 状态
    status: SpanStatus = SpanStatus.UNSET
    status_message: str = ""
    
    # 属性
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # 事件
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 链接
    links: List[SpanContext] = field(default_factory=list)
    
    def set_attribute(self, key: str, value: Any):
        """设置属性"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """添加事件"""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "attributes": attributes or {},
        })
    
    def set_status(self, status: SpanStatus, message: str = ""):
        """设置状态"""
        self.status = status
        self.status_message = message
    
    def end(self):
        """结束 Span"""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": str(self.context.trace_id),
            "span_id": str(self.context.span_id),
            "parent_span_id": str(self.context.parent_span_id) if self.context.parent_span_id else None,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "events": self.events,
        }


class Tracer:
    """
    追踪器
    
    创建和管理 Span
    """
    
    def __init__(self, service_name: str = "nlgsm"):
        self.service_name = service_name
        self._active_spans: Dict[int, List[Span]] = {}  # thread_id -> span stack
        self._completed_spans: List[Span] = []
        self._max_completed_spans: int = 1000
        self._lock = threading.Lock()
        
        # 采样率
        self._sample_rate: float = 1.0
        
        # 导出器
        self._exporters: List[callable] = []
    
    @contextmanager
    def start_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        开始新的 Span（上下文管理器）
        
        用法:
            with tracer.start_span("operation_name") as span:
                span.set_attribute("key", "value")
                # ... 执行操作 ...
        """
        span = self._create_span(name, parent, attributes)
        self._push_span(span)
        
        try:
            yield span
            if span.status == SpanStatus.UNSET:
                span.set_status(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.add_event("exception", {
                "type": type(e).__name__,
                "message": str(e),
            })
            raise
        finally:
            span.end()
            self._pop_span()
            self._record_span(span)
    
    def _create_span(
        self,
        name: str,
        parent: Optional[SpanContext] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """创建 Span"""
        # 确定父上下文
        if parent is None:
            current = self.get_current_span()
            if current:
                parent = current.context
        
        # 创建上下文
        context = SpanContext(
            trace_id=parent.trace_id if parent else uuid4(),
            span_id=uuid4(),
            parent_span_id=parent.span_id if parent else None,
            sampled=self._should_sample(),
        )
        
        span = Span(
            name=name,
            context=context,
            attributes={
                "service.name": self.service_name,
                **(attributes or {}),
            },
        )
        
        return span
    
    def _should_sample(self) -> bool:
        """判断是否采样"""
        import random
        return random.random() < self._sample_rate
    
    def _push_span(self, span: Span):
        """将 Span 压入当前线程的栈"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id not in self._active_spans:
                self._active_spans[thread_id] = []
            self._active_spans[thread_id].append(span)
    
    def _pop_span(self):
        """从当前线程的栈中弹出 Span"""
        thread_id = threading.get_ident()
        with self._lock:
            if thread_id in self._active_spans and self._active_spans[thread_id]:
                self._active_spans[thread_id].pop()
    
    def get_current_span(self) -> Optional[Span]:
        """获取当前 Span"""
        thread_id = threading.get_ident()
        with self._lock:
            spans = self._active_spans.get(thread_id, [])
            return spans[-1] if spans else None
    
    def _record_span(self, span: Span):
        """记录完成的 Span"""
        if not span.context.sampled:
            return
        
        with self._lock:
            self._completed_spans.append(span)
            if len(self._completed_spans) > self._max_completed_spans:
                self._completed_spans = self._completed_spans[-self._max_completed_spans // 2:]
        
        # 导出
        for exporter in self._exporters:
            try:
                exporter(span)
            except Exception as e:
                logger.error(f"Span export failed: {e}")
    
    def add_exporter(self, exporter: callable):
        """添加导出器"""
        self._exporters.append(exporter)
    
    def set_sample_rate(self, rate: float):
        """设置采样率"""
        self._sample_rate = max(0.0, min(1.0, rate))
    
    def get_trace(self, trace_id: UUID) -> List[Span]:
        """获取完整的 trace"""
        with self._lock:
            return [
                s for s in self._completed_spans
                if s.context.trace_id == trace_id
            ]
    
    def get_recent_spans(self, limit: int = 100) -> List[Span]:
        """获取最近的 Span"""
        with self._lock:
            return sorted(
                self._completed_spans,
                key=lambda s: s.start_time,
                reverse=True
            )[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            spans = self._completed_spans
        
        if not spans:
            return {
                "total_spans": 0,
                "avg_duration_ms": 0,
                "error_rate": 0,
            }
        
        total = len(spans)
        errors = sum(1 for s in spans if s.status == SpanStatus.ERROR)
        durations = [s.duration_ms for s in spans if s.duration_ms]
        
        return {
            "total_spans": total,
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0,
            "error_rate": errors / total if total > 0 else 0,
            "sample_rate": self._sample_rate,
        }


# 全局追踪器
_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """获取全局追踪器"""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer

