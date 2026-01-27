"""
可观测性系统

提供 NLGSM 系统的监控和可观测性支持：
1. 指标收集和导出 (Prometheus 格式)
2. 健康检查
3. 告警管理
4. 追踪支持
"""

from .metrics import (
    MetricsCollector,
    Counter,
    Gauge,
    Histogram,
    get_metrics_collector,
)
from .health import (
    HealthChecker,
    HealthStatus,
    HealthCheckResult,
    ComponentHealth,
)
from .alerting import (
    AlertManager,
    Alert,
    AlertSeverity,
    AlertRule,
)
from .tracing import (
    Tracer,
    Span,
    SpanContext,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "Counter",
    "Gauge",
    "Histogram",
    "get_metrics_collector",
    # Health
    "HealthChecker",
    "HealthStatus",
    "HealthCheckResult",
    "ComponentHealth",
    # Alerting
    "AlertManager",
    "Alert",
    "AlertSeverity",
    "AlertRule",
    # Tracing
    "Tracer",
    "Span",
    "SpanContext",
]

