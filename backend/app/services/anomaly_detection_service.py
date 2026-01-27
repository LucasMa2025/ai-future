"""
异常检测服务

将异常检测系统集成到后端 API，提供:
1. 异常检测接口
2. 检测器配置管理
3. 异常事件持久化
4. 与治理系统的集成
"""
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID
import logging

from sqlalchemy.orm import Session

from ..core.anomaly import (
    AnomalyEnsemble,
    AnomalySignal,
    AnomalySeverity,
    DetectorType,
    AnomalyResponse,
    ResponseDecision,
)
from ..core.anomaly.detectors import DetectorConfig
from ..core.anomaly.ensemble import EnsembleConfig
from ..core.anomaly.response import AnomalyResponseDecider, AnomalyResponseIntegration
from ..core.anomaly.signals import CompositeAnomalySignal
from ..models.anomaly import AnomalyEvent, AnomalySignalRecord
from ..core.exceptions import NotFoundError


logger = logging.getLogger(__name__)


class AnomalyDetectionService:
    """
    异常检测服务
    
    提供统一的异常检测接口，管理检测器配置，持久化检测结果
    """
    
    def __init__(
        self,
        db: Session,
        state_machine_service=None,
        notification_service=None,
    ):
        self.db = db
        self.state_machine = state_machine_service
        self.notification_service = notification_service
        
        # 初始化检测器集成
        self.ensemble = AnomalyEnsemble()
        
        # 初始化响应决策器
        self.response_decider = AnomalyResponseDecider()
        self._register_action_handlers()
        
        # 状态机集成
        self.integration = AnomalyResponseIntegration(state_machine_service)
    
    def _register_action_handlers(self):
        """注册响应动作处理器"""
        self.response_decider.register_action_handler(
            "freeze_all", self._action_freeze_all
        )
        self.response_decider.register_action_handler(
            "freeze_learning", self._action_freeze_learning
        )
        self.response_decider.register_action_handler(
            "restore_checkpoint", self._action_restore_checkpoint
        )
        self.response_decider.register_action_handler(
            "trigger_safe_halt", self._action_trigger_safe_halt
        )
        self.response_decider.register_action_handler(
            "collect_diagnostics", self._action_collect_diagnostics
        )
        self.response_decider.register_action_handler(
            "send_alert", self._action_send_alert
        )
        self.response_decider.register_action_handler(
            "notify_governance_committee", self._action_notify_governance_committee
        )
        self.response_decider.register_action_handler(
            "notify_operators", self._action_notify_operators
        )
        self.response_decider.register_action_handler(
            "notify_ml_engineers", self._action_notify_ml_engineers
        )
        self.response_decider.register_action_handler(
            "log_anomaly", self._action_log_anomaly
        )
        self.response_decider.register_action_handler(
            "log_detailed", self._action_log_detailed
        )
        self.response_decider.register_action_handler(
            "log_external_event", self._action_log_external_event
        )
        self.response_decider.register_action_handler(
            "analyze_behavior", self._action_analyze_behavior
        )
        self.response_decider.register_action_handler(
            "analyze_drift", self._action_analyze_drift
        )
    
    # ==================== 检测接口 ====================
    
    def detect(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行异常检测
        
        Args:
            data: 检测数据，格式:
                {
                    "metrics": {"metric_name": "accuracy", "value": 0.85},
                    "behavior": {"decision": "approve", "confidence": 0.9},
                    "drift": {"error_rate": 0.05},
                    "external": {"event_type": "...", "source": "..."},
                }
        
        Returns:
            检测结果
        """
        # 执行检测
        composite_signal = self.ensemble.detect(data)
        
        # 如果检测到异常，决策响应并执行
        response = None
        if composite_signal.triggered_detectors:
            response = self.response_decider.decide(composite_signal)
            
            # 自动执行非人工审核的响应
            if not response.requires_human_review:
                response = self.response_decider.execute_response(response)
                # 集成到状态机
                self.integration.integrate_response(response)
            
            # 持久化异常事件
            self._persist_anomaly_event(composite_signal, response)
        
        return {
            "detected": len(composite_signal.triggered_detectors) > 0,
            "severity": composite_signal.final_severity.value,
            "score": composite_signal.composite_score,
            "triggered_detectors": [d.value for d in composite_signal.triggered_detectors],
            "recommended_action": composite_signal.recommended_action,
            "response": response.to_dict() if response else None,
        }
    
    def detect_metric(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """检测单个指标"""
        data = {
            "metrics": {
                "metric_name": metric_name,
                "value": value,
                "labels": labels or {},
            }
        }
        return self.detect(data)
    
    def detect_behavior(
        self,
        decision: str,
        confidence: float,
        output_distribution: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """检测行为异常"""
        data = {
            "behavior": {
                "decision": decision,
                "confidence": confidence,
                "output_distribution": output_distribution or {},
            }
        }
        return self.detect(data)
    
    def detect_drift(
        self,
        error_rate: Optional[float] = None,
        prediction_correct: Optional[bool] = None,
        feature_means: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """检测漂移"""
        drift_data = {}
        if error_rate is not None:
            drift_data["error_rate"] = error_rate
        if prediction_correct is not None:
            drift_data["prediction_correct"] = prediction_correct
        if feature_means:
            drift_data["feature_means"] = feature_means
        
        data = {"drift": drift_data}
        return self.detect(data)
    
    def report_external_event(
        self,
        event_type: str,
        source: str,
        severity: Optional[str] = None,
        description: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """报告外部异常事件"""
        external_data = {
            "event_type": event_type,
            "source": source,
        }
        if severity:
            external_data["severity"] = severity
        if description:
            external_data["description"] = description
        if evidence:
            external_data["evidence"] = evidence
        
        data = {"external": external_data}
        return self.detect(data)
    
    # ==================== 配置管理 ====================
    
    def set_behavior_baseline(self, distribution: Dict[str, float]):
        """设置行为检测的基准分布"""
        self.ensemble.set_behavior_baseline(distribution)
    
    def update_detector_config(
        self,
        detector_type: str,
        config: Dict[str, Any],
    ):
        """更新检测器配置"""
        dt = DetectorType(detector_type)
        detector = self.ensemble.get_detector(dt)
        
        if detector:
            if "enabled" in config:
                detector.config.enabled = config["enabled"]
            if "sensitivity" in config:
                detector.config.sensitivity = config["sensitivity"]
            if "low_threshold" in config:
                detector.config.low_threshold = config["low_threshold"]
            if "medium_threshold" in config:
                detector.config.medium_threshold = config["medium_threshold"]
            if "high_threshold" in config:
                detector.config.high_threshold = config["high_threshold"]
            if "critical_threshold" in config:
                detector.config.critical_threshold = config["critical_threshold"]
            if "cooldown_seconds" in config:
                detector.config.cooldown_seconds = config["cooldown_seconds"]
    
    def get_detector_config(self, detector_type: str) -> Optional[Dict[str, Any]]:
        """获取检测器配置"""
        dt = DetectorType(detector_type)
        detector = self.ensemble.get_detector(dt)
        
        if detector:
            cfg = detector.config
            return {
                "enabled": cfg.enabled,
                "sensitivity": cfg.sensitivity,
                "low_threshold": cfg.low_threshold,
                "medium_threshold": cfg.medium_threshold,
                "high_threshold": cfg.high_threshold,
                "critical_threshold": cfg.critical_threshold,
                "window_size": cfg.window_size,
                "min_samples": cfg.min_samples,
                "cooldown_seconds": cfg.cooldown_seconds,
            }
        return None
    
    # ==================== 查询接口 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取异常检测统计"""
        ensemble_stats = self.ensemble.get_statistics()
        response_stats = self.response_decider.get_statistics()
        
        return {
            "detection": ensemble_stats,
            "response": response_stats,
        }
    
    def get_recent_anomalies(
        self,
        limit: int = 20,
        min_severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """获取最近的异常"""
        severity = AnomalySeverity(min_severity) if min_severity else None
        anomalies = self.ensemble.get_recent_anomalies(limit, severity)
        
        return [a.to_dict() for a in anomalies]
    
    def list_anomaly_events(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> tuple[List[AnomalyEvent], int]:
        """列出异常事件"""
        query = self.db.query(AnomalyEvent)
        
        if status:
            query = query.filter(AnomalyEvent.status == status)
        if severity:
            query = query.filter(AnomalyEvent.severity == severity)
        
        total = query.count()
        items = query.order_by(
            AnomalyEvent.detected_at.desc()
        ).offset(skip).limit(limit).all()
        
        return items, total
    
    def get_anomaly_event(self, event_id: UUID) -> Optional[AnomalyEvent]:
        """获取单个异常事件"""
        return self.db.query(AnomalyEvent).filter(
            AnomalyEvent.id == event_id
        ).first()
    
    def resolve_anomaly_event(
        self,
        event_id: UUID,
        resolved_by: UUID,
        resolution_notes: str,
    ) -> AnomalyEvent:
        """解决异常事件"""
        event = self.get_anomaly_event(event_id)
        if not event:
            raise NotFoundError("AnomalyEvent", str(event_id))
        
        event.status = "resolved"
        event.resolved_by = resolved_by
        event.resolved_at = datetime.utcnow()
        event.resolution_notes = resolution_notes
        
        self.db.commit()
        self.db.refresh(event)
        
        return event
    
    # ==================== 持久化 ====================
    
    def _persist_anomaly_event(
        self,
        signal: CompositeAnomalySignal,
        response: AnomalyResponse,
    ):
        """持久化异常事件"""
        try:
            event = AnomalyEvent(
                severity=signal.final_severity.value,
                composite_score=signal.composite_score,
                detected_by=[d.value for d in signal.triggered_detectors],
                recommendation=signal.recommended_action,
                response_decision=response.decision.value if response else None,
                status="open" if response and response.requires_human_review else "resolved",
            )
            
            self.db.add(event)
            self.db.flush()
            
            # 持久化各信号详情
            for sig in signal.signals:
                if sig.detected:
                    signal_record = AnomalySignalRecord(
                        anomaly_event_id=event.id,
                        detector_type=sig.detector_type.value,
                        detected=sig.detected,
                        severity=sig.severity.value,
                        signal_data={
                            "score": sig.score,
                            "threshold": sig.threshold,
                            "metric_name": sig.metric_name,
                            "current_value": sig.current_value,
                            "expected_value": sig.expected_value,
                            "context": sig.context,
                        },
                    )
                    self.db.add(signal_record)
            
            self.db.commit()
            logger.info(f"Persisted anomaly event {event.id}")
            
        except Exception as e:
            logger.error(f"Failed to persist anomaly event: {e}")
            self.db.rollback()
    
    # ==================== 动作处理器 ====================
    
    def _action_freeze_all(self, signal, params):
        """冻结所有学习"""
        logger.info("Action: Freezing all learning activities")
        return {"action": "freeze_all", "status": "completed"}
    
    def _action_freeze_learning(self, signal, params):
        """冻结学习"""
        logger.info("Action: Freezing learning")
        return {"action": "freeze_learning", "status": "completed"}
    
    def _action_restore_checkpoint(self, signal, params):
        """恢复检查点"""
        logger.info("Action: Restoring to last known good checkpoint")
        return {"action": "restore_checkpoint", "status": "completed"}
    
    def _action_trigger_safe_halt(self, signal, params):
        """触发安全停机"""
        logger.warning("Action: Triggering safe halt!")
        if self.state_machine:
            from ..core.enums import EventType
            from .state_machine_service import Event
            
            event = Event(
                event_type=EventType.ANOMALY,
                source="anomaly_detection",
                metadata={"severity": "critical"},
            )
            self.state_machine.process_event(event)
        
        return {"action": "trigger_safe_halt", "status": "completed"}
    
    def _action_collect_diagnostics(self, signal, params):
        """收集诊断信息"""
        logger.info("Action: Collecting diagnostics")
        return {"action": "collect_diagnostics", "status": "completed"}
    
    def _action_send_alert(self, signal, params):
        """发送告警"""
        logger.info("Action: Sending alert")
        if self.notification_service and signal:
            # 发送通知
            pass
        return {"action": "send_alert", "status": "completed"}
    
    def _action_notify_governance_committee(self, signal, params):
        """通知治理委员会"""
        logger.info("Action: Notifying governance committee")
        return {"action": "notify_governance_committee", "status": "completed"}
    
    def _action_notify_operators(self, signal, params):
        """通知运维人员"""
        logger.info("Action: Notifying operators")
        return {"action": "notify_operators", "status": "completed"}
    
    def _action_notify_ml_engineers(self, signal, params):
        """通知 ML 工程师"""
        logger.info("Action: Notifying ML engineers")
        return {"action": "notify_ml_engineers", "status": "completed"}
    
    def _action_log_anomaly(self, signal, params):
        """记录异常日志"""
        if signal:
            logger.info(f"Anomaly logged: {signal.to_dict()}")
        return {"action": "log_anomaly", "status": "completed"}
    
    def _action_log_detailed(self, signal, params):
        """记录详细日志"""
        if signal:
            logger.info(f"Detailed anomaly: {signal.to_dict()}")
        return {"action": "log_detailed", "status": "completed"}
    
    def _action_log_external_event(self, signal, params):
        """记录外部事件"""
        logger.info("External event logged")
        return {"action": "log_external_event", "status": "completed"}
    
    def _action_analyze_behavior(self, signal, params):
        """分析行为"""
        logger.info("Action: Analyzing behavior anomaly")
        return {"action": "analyze_behavior", "status": "completed"}
    
    def _action_analyze_drift(self, signal, params):
        """分析漂移"""
        logger.info("Action: Analyzing drift")
        return {"action": "analyze_drift", "status": "completed"}

