"""
可观测性服务

整合监控、健康检查、告警和追踪功能
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
import logging

from sqlalchemy.orm import Session

from ..core.observability.metrics import MetricsCollector, get_metrics_collector
from ..core.observability.health import HealthChecker, HealthCheckResult, get_health_checker
from ..core.observability.alerting import AlertManager, Alert, AlertSeverity, get_alert_manager
from ..core.observability.tracing import Tracer, get_tracer


logger = logging.getLogger(__name__)


class ObservabilityService:
    """
    可观测性服务
    
    提供统一的可观测性接口
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.metrics = get_metrics_collector()
        self.health = get_health_checker()
        self.alerts = get_alert_manager()
        self.tracer = get_tracer()
        
        # 注册自定义健康检查
        self._register_health_checks()
    
    def _register_health_checks(self):
        """注册自定义健康检查"""
        # 数据库健康检查
        from ..core.observability.health import ComponentHealth, HealthStatus
        
        def check_db():
            try:
                self.db.execute("SELECT 1")
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection is healthy",
                )
            except Exception as e:
                return ComponentHealth(
                    name="database",
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )
        
        self.health.register("database", check_db)
    
    # ==================== 指标接口 ====================
    
    def get_metrics_prometheus(self) -> str:
        """获取 Prometheus 格式的指标"""
        return self.metrics.export_prometheus()
    
    def get_metrics_json(self) -> Dict[str, Any]:
        """获取 JSON 格式的指标"""
        return self.metrics.get_all_metrics()
    
    def record_metric(
        self,
        metric_type: str,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ):
        """
        记录指标
        
        Args:
            metric_type: 指标类型 (counter, gauge, histogram)
            name: 指标名称
            value: 值
            labels: 标签
        """
        if metric_type == "counter":
            counter = self.metrics.counter(name)
            if counter:
                counter.inc(value, labels)
        elif metric_type == "gauge":
            gauge = self.metrics.gauge(name)
            if gauge:
                gauge.set(value, labels)
        elif metric_type == "histogram":
            histogram = self.metrics.histogram(name)
            if histogram:
                histogram.observe(value, labels)
    
    # ==================== 健康检查接口 ====================
    
    def check_health(self, use_cache: bool = True) -> Dict[str, Any]:
        """执行健康检查"""
        result = self.health.check(use_cache)
        return result.to_dict()
    
    def check_component_health(self, component: str) -> Optional[Dict[str, Any]]:
        """检查单个组件健康状态"""
        result = self.health.check_component(component)
        if result:
            return {
                "name": result.name,
                "status": result.status.value,
                "message": result.message,
                "details": result.details,
                "response_time_ms": result.response_time_ms,
            }
        return None
    
    def get_health_components(self) -> List[str]:
        """获取所有健康检查组件"""
        return self.health.get_component_names()
    
    # ==================== 告警接口 ====================
    
    def fire_alert(
        self,
        name: str,
        message: str,
        severity: str = "warning",
        source: str = "",
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """触发告警"""
        sev = AlertSeverity(severity)
        alert = self.alerts.fire_alert(
            name=name,
            message=message,
            severity=sev,
            source=source,
            labels=labels,
            channels=["log", "websocket"],
        )
        return alert.to_dict()
    
    def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """列出告警"""
        alerts = self.alerts.list_alerts(status, severity, limit)
        return [a.to_dict() for a in alerts]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        alerts = self.alerts.get_active_alerts()
        return [a.to_dict() for a in alerts]
    
    def acknowledge_alert(self, alert_id: UUID, user_id: UUID) -> Optional[Dict[str, Any]]:
        """确认告警"""
        alert = self.alerts.acknowledge(alert_id, user_id)
        return alert.to_dict() if alert else None
    
    def resolve_alert(self, alert_id: UUID) -> Optional[Dict[str, Any]]:
        """解决告警"""
        alert = self.alerts.resolve(alert_id)
        return alert.to_dict() if alert else None
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        return self.alerts.get_statistics()
    
    # ==================== 追踪接口 ====================
    
    def get_trace(self, trace_id: UUID) -> List[Dict[str, Any]]:
        """获取完整的 trace"""
        spans = self.tracer.get_trace(trace_id)
        return [s.to_dict() for s in spans]
    
    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的 trace"""
        spans = self.tracer.get_recent_spans(limit)
        
        # 按 trace_id 分组
        traces: Dict[str, List] = {}
        for span in spans:
            trace_id = str(span.context.trace_id)
            if trace_id not in traces:
                traces[trace_id] = []
            traces[trace_id].append(span.to_dict())
        
        return [
            {"trace_id": tid, "spans": spans}
            for tid, spans in traces.items()
        ]
    
    def get_tracing_statistics(self) -> Dict[str, Any]:
        """获取追踪统计"""
        return self.tracer.get_statistics()
    
    # ==================== 综合统计 ====================
    
    def get_overview(self) -> Dict[str, Any]:
        """
        获取系统概览
        
        包含健康状态、活跃告警、关键指标
        """
        health = self.health.check()
        active_alerts = self.alerts.get_active_alerts()
        
        # 获取关键指标
        lu_counter = self.metrics.counter("learning_units_total")
        pending_gauge = self.metrics.gauge("pending_lu_count")
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": {
                "overall_status": health.overall_status.value,
                "components_count": len(health.components),
                "unhealthy_components": [
                    name for name, c in health.components.items()
                    if c.status.value == "unhealthy"
                ],
            },
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": sum(
                    1 for a in active_alerts
                    if a.severity == AlertSeverity.CRITICAL
                ),
                "recent_alerts": [a.to_dict() for a in active_alerts[:5]],
            },
            "metrics": {
                "learning_units_total": lu_counter.get() if lu_counter else 0,
                "pending_lu_count": pending_gauge.get() if pending_gauge else 0,
            },
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        获取仪表板数据
        
        用于前端仪表板展示
        """
        overview = self.get_overview()
        alert_stats = self.alerts.get_statistics()
        trace_stats = self.tracer.get_statistics()
        
        return {
            **overview,
            "alert_statistics": alert_stats,
            "tracing_statistics": trace_stats,
        }

