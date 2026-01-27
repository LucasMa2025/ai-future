"""
告警管理系统

提供告警规则、告警发送和告警管理功能
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from uuid import UUID, uuid4
import threading
import logging


logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """告警状态"""
    FIRING = "firing"       # 触发中
    RESOLVED = "resolved"   # 已解决
    SILENCED = "silenced"   # 已静默
    ACKNOWLEDGED = "acknowledged"  # 已确认


@dataclass
class Alert:
    """告警"""
    id: UUID = field(default_factory=uuid4)
    
    # 基本信息
    name: str = ""
    severity: AlertSeverity = AlertSeverity.WARNING
    message: str = ""
    
    # 来源
    source: str = ""
    rule_id: Optional[str] = None
    
    # 标签和注解
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # 状态
    status: AlertStatus = AlertStatus.FIRING
    
    # 时间
    started_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[UUID] = None
    
    # 通知
    notified: bool = False
    notification_channels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "source": self.source,
            "rule_id": self.rule_id,
            "labels": self.labels,
            "annotations": self.annotations,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str = ""
    
    # 条件
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    threshold: Optional[float] = None
    comparison: str = "gt"  # gt, lt, eq, ne, ge, le
    
    # 触发配置
    for_duration: timedelta = timedelta(seconds=0)  # 持续时间
    
    # 告警配置
    severity: AlertSeverity = AlertSeverity.WARNING
    message_template: str = ""
    
    # 标签
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # 通知
    notification_channels: List[str] = field(default_factory=list)
    
    # 状态
    enabled: bool = True
    
    # 冷却
    cooldown: timedelta = timedelta(minutes=5)
    last_fired: Optional[datetime] = None


class AlertManager:
    """
    告警管理器
    
    管理告警规则、触发告警、发送通知
    """
    
    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[UUID, Alert] = {}
        self._pending_conditions: Dict[str, datetime] = {}  # 规则ID -> 首次满足条件时间
        self._notification_handlers: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # 初始化默认规则
        self._init_default_rules()
        
        # 注册默认通知渠道
        self._register_default_channels()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        # 高错误率告警
        self.add_rule(AlertRule(
            id="high_error_rate",
            name="High Error Rate",
            description="Error rate exceeds threshold",
            threshold=0.1,
            comparison="gt",
            for_duration=timedelta(minutes=1),
            severity=AlertSeverity.ERROR,
            message_template="Error rate is {value:.2%}, exceeds threshold {threshold:.2%}",
            notification_channels=["log", "websocket"],
        ))
        
        # 学习系统停止告警
        self.add_rule(AlertRule(
            id="learning_system_stopped",
            name="Learning System Stopped",
            description="Learning system has stopped unexpectedly",
            severity=AlertSeverity.CRITICAL,
            message_template="Learning system has stopped unexpectedly",
            notification_channels=["log", "websocket", "email"],
        ))
        
        # 审批超时告警
        self.add_rule(AlertRule(
            id="approval_timeout",
            name="Approval Timeout",
            description="Approval request has timed out",
            for_duration=timedelta(hours=24),
            severity=AlertSeverity.WARNING,
            message_template="Approval request {approval_id} has been pending for over 24 hours",
            notification_channels=["log", "websocket"],
        ))
        
        # 安全停机告警
        self.add_rule(AlertRule(
            id="safe_halt_triggered",
            name="Safe Halt Triggered",
            description="System has entered safe halt state",
            severity=AlertSeverity.CRITICAL,
            message_template="System has entered safe halt state: {reason}",
            notification_channels=["log", "websocket", "email"],
        ))
    
    def _register_default_channels(self):
        """注册默认通知渠道"""
        self.register_channel("log", self._notify_log)
        self.register_channel("websocket", self._notify_websocket)
        self.register_channel("email", self._notify_email)
    
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self._rules[rule.id] = rule
    
    def remove_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self._rules:
            del self._rules[rule_id]
    
    def enable_rule(self, rule_id: str):
        """启用规则"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
    
    def disable_rule(self, rule_id: str):
        """禁用规则"""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
    
    def register_channel(self, name: str, handler: Callable):
        """注册通知渠道"""
        self._notification_handlers[name] = handler
    
    def evaluate(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        评估指标值
        
        Args:
            metric_name: 指标名称
            value: 指标值
            labels: 标签
        """
        labels = labels or {}
        
        for rule_id, rule in self._rules.items():
            if not rule.enabled:
                continue
            
            # 检查条件
            triggered = self._check_condition(rule, value)
            
            if triggered:
                # 检查持续时间
                if rule.for_duration.total_seconds() > 0:
                    if rule_id not in self._pending_conditions:
                        self._pending_conditions[rule_id] = datetime.utcnow()
                        continue
                    
                    elapsed = datetime.utcnow() - self._pending_conditions[rule_id]
                    if elapsed < rule.for_duration:
                        continue
                
                # 检查冷却
                if rule.last_fired:
                    elapsed = datetime.utcnow() - rule.last_fired
                    if elapsed < rule.cooldown:
                        continue
                
                # 触发告警
                self._fire_alert(rule, value, labels)
                rule.last_fired = datetime.utcnow()
                
                # 清除待处理状态
                if rule_id in self._pending_conditions:
                    del self._pending_conditions[rule_id]
            
            else:
                # 条件不满足，清除待处理状态
                if rule_id in self._pending_conditions:
                    del self._pending_conditions[rule_id]
    
    def fire_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        source: str = "",
        labels: Optional[Dict[str, str]] = None,
        channels: Optional[List[str]] = None,
    ) -> Alert:
        """
        直接触发告警
        
        Args:
            name: 告警名称
            message: 告警消息
            severity: 严重程度
            source: 来源
            labels: 标签
            channels: 通知渠道
            
        Returns:
            创建的告警
        """
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            source=source,
            labels=labels or {},
            notification_channels=channels or ["log"],
        )
        
        with self._lock:
            self._alerts[alert.id] = alert
        
        # 发送通知
        self._send_notifications(alert)
        
        logger.warning(f"Alert fired: {name} - {message}")
        
        return alert
    
    def _check_condition(self, rule: AlertRule, value: float) -> bool:
        """检查规则条件"""
        if rule.threshold is None:
            return False
        
        if rule.comparison == "gt":
            return value > rule.threshold
        elif rule.comparison == "lt":
            return value < rule.threshold
        elif rule.comparison == "ge":
            return value >= rule.threshold
        elif rule.comparison == "le":
            return value <= rule.threshold
        elif rule.comparison == "eq":
            return value == rule.threshold
        elif rule.comparison == "ne":
            return value != rule.threshold
        
        return False
    
    def _fire_alert(self, rule: AlertRule, value: float, labels: Dict[str, str]):
        """根据规则触发告警"""
        message = rule.message_template.format(
            value=value,
            threshold=rule.threshold,
            **labels,
        )
        
        alert = Alert(
            name=rule.name,
            severity=rule.severity,
            message=message,
            source="rule_engine",
            rule_id=rule.id,
            labels={**rule.labels, **labels},
            annotations=rule.annotations,
            notification_channels=rule.notification_channels,
        )
        
        with self._lock:
            self._alerts[alert.id] = alert
        
        self._send_notifications(alert)
    
    def _send_notifications(self, alert: Alert):
        """发送通知"""
        for channel in alert.notification_channels:
            handler = self._notification_handlers.get(channel)
            if handler:
                try:
                    handler(alert)
                    alert.notified = True
                except Exception as e:
                    logger.error(f"Failed to send notification via {channel}: {e}")
    
    def acknowledge(self, alert_id: UUID, user_id: UUID) -> Optional[Alert]:
        """确认告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = user_id
        return alert
    
    def resolve(self, alert_id: UUID) -> Optional[Alert]:
        """解决告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
        return alert
    
    def silence(self, alert_id: UUID) -> Optional[Alert]:
        """静默告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.status = AlertStatus.SILENCED
        return alert
    
    def get_alert(self, alert_id: UUID) -> Optional[Alert]:
        """获取告警"""
        return self._alerts.get(alert_id)
    
    def list_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """列出告警"""
        alerts = list(self._alerts.values())
        
        if status:
            alerts = [a for a in alerts if a.status.value == status]
        if severity:
            alerts = [a for a in alerts if a.severity.value == severity]
        
        return sorted(alerts, key=lambda a: a.started_at, reverse=True)[:limit]
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [
            a for a in self._alerts.values()
            if a.status == AlertStatus.FIRING
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        alerts = list(self._alerts.values())
        
        severity_counts = {}
        status_counts = {}
        
        for alert in alerts:
            severity_counts[alert.severity.value] = \
                severity_counts.get(alert.severity.value, 0) + 1
            status_counts[alert.status.value] = \
                status_counts.get(alert.status.value, 0) + 1
        
        return {
            "total_alerts": len(alerts),
            "active_alerts": len(self.get_active_alerts()),
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "rules_count": len(self._rules),
        }
    
    # ==================== 默认通知处理器 ====================
    
    def _notify_log(self, alert: Alert):
        """日志通知"""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.name} - {alert.message}")
    
    def _notify_websocket(self, alert: Alert):
        """WebSocket 通知"""
        # 实际实现中应该通过 WebSocket 发送
        pass
    
    def _notify_email(self, alert: Alert):
        """邮件通知"""
        # 实际实现中应该发送邮件
        pass


# 全局告警管理器
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager

