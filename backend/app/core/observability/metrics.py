"""
指标收集和导出

支持 Prometheus 格式的指标导出
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import threading
import time
import logging


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """指标值"""
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class Counter:
    """
    计数器指标
    
    只能增加，适合统计请求数、错误数等
    """
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def inc(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """增加计数"""
        labels = labels or {}
        key = self._labels_to_key(labels)
        
        with self._lock:
            if key not in self._values:
                self._values[key] = 0
            self._values[key] += value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取当前值"""
        labels = labels or {}
        key = self._labels_to_key(labels)
        return self._values.get(key, 0)
    
    def _labels_to_key(self, labels: Dict[str, str]) -> tuple:
        """将标签转换为键"""
        return tuple(sorted(labels.items()))
    
    def collect(self) -> List[MetricValue]:
        """收集所有值"""
        values = []
        with self._lock:
            for key, value in self._values.items():
                labels = dict(key)
                values.append(MetricValue(value=value, labels=labels))
        return values


class Gauge:
    """
    仪表盘指标
    
    可以增加或减少，适合统计当前连接数、队列长度等
    """
    
    def __init__(self, name: str, description: str = "", labels: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self._values: Dict[tuple, float] = {}
        self._lock = threading.Lock()
    
    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        """设置值"""
        labels = labels or {}
        key = self._labels_to_key(labels)
        
        with self._lock:
            self._values[key] = value
    
    def inc(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """增加"""
        labels = labels or {}
        key = self._labels_to_key(labels)
        
        with self._lock:
            if key not in self._values:
                self._values[key] = 0
            self._values[key] += value
    
    def dec(self, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """减少"""
        self.inc(-value, labels)
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """获取当前值"""
        labels = labels or {}
        key = self._labels_to_key(labels)
        return self._values.get(key, 0)
    
    def _labels_to_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))
    
    def collect(self) -> List[MetricValue]:
        values = []
        with self._lock:
            for key, value in self._values.items():
                labels = dict(key)
                values.append(MetricValue(value=value, labels=labels))
        return values


class Histogram:
    """
    直方图指标
    
    统计值的分布，适合统计延迟、请求大小等
    """
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, float("inf")]
    
    def __init__(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ):
        self.name = name
        self.description = description
        self.label_names = labels or []
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        
        self._bucket_counts: Dict[tuple, Dict[float, int]] = {}
        self._sums: Dict[tuple, float] = {}
        self._counts: Dict[tuple, int] = {}
        self._lock = threading.Lock()
    
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None):
        """观察一个值"""
        labels = labels or {}
        key = self._labels_to_key(labels)
        
        with self._lock:
            if key not in self._bucket_counts:
                self._bucket_counts[key] = {b: 0 for b in self.buckets}
                self._sums[key] = 0
                self._counts[key] = 0
            
            self._sums[key] += value
            self._counts[key] += 1
            
            for bucket in self.buckets:
                if value <= bucket:
                    self._bucket_counts[key][bucket] += 1
    
    def time(self, labels: Optional[Dict[str, str]] = None):
        """计时器上下文管理器"""
        return _HistogramTimer(self, labels)
    
    def _labels_to_key(self, labels: Dict[str, str]) -> tuple:
        return tuple(sorted(labels.items()))
    
    def collect(self) -> List[MetricValue]:
        """收集所有值"""
        values = []
        with self._lock:
            for key, buckets in self._bucket_counts.items():
                labels = dict(key)
                
                # 添加桶计数
                for bucket, count in buckets.items():
                    bucket_labels = {**labels, "le": str(bucket)}
                    values.append(MetricValue(value=count, labels=bucket_labels))
                
                # 添加总和
                sum_labels = {**labels}
                values.append(MetricValue(
                    value=self._sums.get(key, 0),
                    labels={**sum_labels, "_type": "sum"}
                ))
                
                # 添加计数
                values.append(MetricValue(
                    value=self._counts.get(key, 0),
                    labels={**sum_labels, "_type": "count"}
                ))
        
        return values


class _HistogramTimer:
    """直方图计时器"""
    
    def __init__(self, histogram: Histogram, labels: Optional[Dict[str, str]] = None):
        self.histogram = histogram
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.histogram.observe(duration, self.labels)


class MetricsCollector:
    """
    指标收集器
    
    管理所有指标，提供 Prometheus 格式的导出
    """
    
    def __init__(self, prefix: str = "nlgsm"):
        self.prefix = prefix
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        
        # 初始化默认指标
        self._init_default_metrics()
    
    def _init_default_metrics(self):
        """初始化默认指标"""
        # 学习相关
        self.register_counter(
            "learning_units_total",
            "Total number of learning units created",
            ["status", "risk_level"]
        )
        self.register_counter(
            "learning_steps_total",
            "Total number of learning steps executed",
            ["session_id"]
        )
        
        # 状态机相关
        self.register_counter(
            "state_transitions_total",
            "Total number of state transitions",
            ["from_state", "to_state", "event_type"]
        )
        self.register_gauge(
            "current_state",
            "Current state machine state (encoded as number)",
            ["state"]
        )
        
        # 异常检测
        self.register_counter(
            "anomalies_detected_total",
            "Total number of anomalies detected",
            ["detector_type", "severity"]
        )
        self.register_gauge(
            "anomaly_score",
            "Current anomaly score",
            ["detector_type"]
        )
        
        # 审批相关
        self.register_counter(
            "approvals_total",
            "Total number of approvals",
            ["decision", "risk_level"]
        )
        self.register_gauge(
            "pending_approvals",
            "Number of pending approvals",
            ["risk_level"]
        )
        
        # 性能指标
        self.register_histogram(
            "request_duration_seconds",
            "Request duration in seconds",
            ["endpoint", "method"]
        )
        self.register_histogram(
            "learning_step_duration_seconds",
            "Learning step duration in seconds",
            ["nl_level"]
        )
        
        # 系统健康
        self.register_gauge(
            "system_healthy",
            "System health status (1=healthy, 0=unhealthy)",
            ["component"]
        )
        self.register_gauge(
            "active_learners",
            "Number of active learner threads",
        )
        self.register_gauge(
            "pending_lu_count",
            "Number of pending learning units",
        )
    
    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Counter:
        """注册计数器"""
        full_name = f"{self.prefix}_{name}"
        counter = Counter(full_name, description, labels)
        self._counters[name] = counter
        return counter
    
    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
    ) -> Gauge:
        """注册仪表盘"""
        full_name = f"{self.prefix}_{name}"
        gauge = Gauge(full_name, description, labels)
        self._gauges[name] = gauge
        return gauge
    
    def register_histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """注册直方图"""
        full_name = f"{self.prefix}_{name}"
        histogram = Histogram(full_name, description, labels, buckets)
        self._histograms[name] = histogram
        return histogram
    
    def counter(self, name: str) -> Optional[Counter]:
        """获取计数器"""
        return self._counters.get(name)
    
    def gauge(self, name: str) -> Optional[Gauge]:
        """获取仪表盘"""
        return self._gauges.get(name)
    
    def histogram(self, name: str) -> Optional[Histogram]:
        """获取直方图"""
        return self._histograms.get(name)
    
    def export_prometheus(self) -> str:
        """
        导出 Prometheus 格式的指标
        
        Returns:
            Prometheus 格式的文本
        """
        lines = []
        
        # 导出计数器
        for name, counter in self._counters.items():
            lines.append(f"# HELP {counter.name} {counter.description}")
            lines.append(f"# TYPE {counter.name} counter")
            for value in counter.collect():
                labels_str = self._format_labels(value.labels)
                lines.append(f"{counter.name}{labels_str} {value.value}")
        
        # 导出仪表盘
        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {gauge.name} {gauge.description}")
            lines.append(f"# TYPE {gauge.name} gauge")
            for value in gauge.collect():
                labels_str = self._format_labels(value.labels)
                lines.append(f"{gauge.name}{labels_str} {value.value}")
        
        # 导出直方图
        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {histogram.name} {histogram.description}")
            lines.append(f"# TYPE {histogram.name} histogram")
            for value in histogram.collect():
                labels = value.labels.copy()
                metric_type = labels.pop("_type", None)
                
                if metric_type == "sum":
                    labels_str = self._format_labels(labels)
                    lines.append(f"{histogram.name}_sum{labels_str} {value.value}")
                elif metric_type == "count":
                    labels_str = self._format_labels(labels)
                    lines.append(f"{histogram.name}_count{labels_str} {value.value}")
                else:
                    labels_str = self._format_labels(labels)
                    lines.append(f"{histogram.name}_bucket{labels_str} {value.value}")
        
        return "\n".join(lines)
    
    def _format_labels(self, labels: Dict[str, str]) -> str:
        """格式化标签"""
        if not labels:
            return ""
        
        pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(pairs) + "}"
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标（JSON 格式）"""
        result = {
            "counters": {},
            "gauges": {},
            "histograms": {},
        }
        
        for name, counter in self._counters.items():
            values = counter.collect()
            result["counters"][name] = [
                {"value": v.value, "labels": v.labels}
                for v in values
            ]
        
        for name, gauge in self._gauges.items():
            values = gauge.collect()
            result["gauges"][name] = [
                {"value": v.value, "labels": v.labels}
                for v in values
            ]
        
        for name, histogram in self._histograms.items():
            values = histogram.collect()
            result["histograms"][name] = [
                {"value": v.value, "labels": v.labels}
                for v in values
            ]
        
        return result


# 全局指标收集器
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector

