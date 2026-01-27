"""
异常信号定义

定义异常检测产生的信号类型和结构
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4


class DetectorType(str, Enum):
    """检测器类型"""
    METRIC = "metric"           # 指标异常
    BEHAVIOR = "behavior"       # 行为异常
    DRIFT = "drift"             # 漂移检测
    EXTERNAL = "external"       # 外部异常


class AnomalySeverity(str, Enum):
    """异常严重程度"""
    LOW = "low"           # 低：仅记录日志
    MEDIUM = "medium"     # 中：需要人工关注
    HIGH = "high"         # 高：触发回滚
    CRITICAL = "critical" # 严重：紧急停机


@dataclass
class AnomalySignal:
    """
    异常信号
    
    由各类检测器产生，包含异常的详细信息
    """
    # 基础信息
    id: UUID = field(default_factory=uuid4)
    detector_type: DetectorType = DetectorType.METRIC
    detected: bool = False
    severity: AnomalySeverity = AnomalySeverity.LOW
    
    # 检测结果
    score: float = 0.0  # 异常分数 0-1
    threshold: float = 0.5  # 触发阈值
    confidence: float = 1.0  # 检测置信度
    
    # 详细信息
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    expected_value: Optional[float] = None
    deviation: Optional[float] = None
    
    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None
    
    # 时间戳
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    # 来源追溯
    source_component: Optional[str] = None
    learning_unit_id: Optional[UUID] = None
    artifact_id: Optional[UUID] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "detector_type": self.detector_type.value,
            "detected": self.detected,
            "severity": self.severity.value,
            "score": self.score,
            "threshold": self.threshold,
            "confidence": self.confidence,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "expected_value": self.expected_value,
            "deviation": self.deviation,
            "context": self.context,
            "detected_at": self.detected_at.isoformat(),
            "source_component": self.source_component,
            "learning_unit_id": str(self.learning_unit_id) if self.learning_unit_id else None,
            "artifact_id": str(self.artifact_id) if self.artifact_id else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalySignal":
        """从字典创建"""
        return cls(
            id=UUID(data["id"]) if "id" in data else uuid4(),
            detector_type=DetectorType(data.get("detector_type", "metric")),
            detected=data.get("detected", False),
            severity=AnomalySeverity(data.get("severity", "low")),
            score=data.get("score", 0.0),
            threshold=data.get("threshold", 0.5),
            confidence=data.get("confidence", 1.0),
            metric_name=data.get("metric_name"),
            current_value=data.get("current_value"),
            expected_value=data.get("expected_value"),
            deviation=data.get("deviation"),
            context=data.get("context", {}),
            detected_at=datetime.fromisoformat(data["detected_at"]) if "detected_at" in data else datetime.utcnow(),
            source_component=data.get("source_component"),
            learning_unit_id=UUID(data["learning_unit_id"]) if data.get("learning_unit_id") else None,
            artifact_id=UUID(data["artifact_id"]) if data.get("artifact_id") else None,
        )


@dataclass
class CompositeAnomalySignal:
    """
    复合异常信号
    
    由多个检测器的信号聚合而成
    """
    id: UUID = field(default_factory=uuid4)
    signals: List[AnomalySignal] = field(default_factory=list)
    
    # 聚合结果
    composite_score: float = 0.0
    final_severity: AnomalySeverity = AnomalySeverity.LOW
    triggered_detectors: List[DetectorType] = field(default_factory=list)
    
    # 推荐响应
    recommended_action: Optional[str] = None
    
    # 时间
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_signal(self, signal: AnomalySignal):
        """添加信号并更新聚合结果"""
        self.signals.append(signal)
        
        if signal.detected:
            if signal.detector_type not in self.triggered_detectors:
                self.triggered_detectors.append(signal.detector_type)
            
            # 更新复合分数（加权平均）
            self._update_composite_score()
            
            # 更新最终严重程度（取最高）
            self._update_severity()
    
    def _update_composite_score(self):
        """更新复合异常分数"""
        if not self.signals:
            self.composite_score = 0.0
            return
        
        # 权重：行为异常和漂移检测权重更高
        weights = {
            DetectorType.METRIC: 1.0,
            DetectorType.BEHAVIOR: 1.5,
            DetectorType.DRIFT: 1.5,
            DetectorType.EXTERNAL: 2.0,
        }
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for signal in self.signals:
            if signal.detected:
                w = weights.get(signal.detector_type, 1.0)
                weighted_sum += signal.score * signal.confidence * w
                total_weight += w
        
        self.composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _update_severity(self):
        """更新最终严重程度"""
        severity_order = [
            AnomalySeverity.LOW,
            AnomalySeverity.MEDIUM,
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL,
        ]
        
        max_severity = AnomalySeverity.LOW
        
        for signal in self.signals:
            if signal.detected:
                if severity_order.index(signal.severity) > severity_order.index(max_severity):
                    max_severity = signal.severity
        
        self.final_severity = max_severity
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "signals": [s.to_dict() for s in self.signals],
            "composite_score": self.composite_score,
            "final_severity": self.final_severity.value,
            "triggered_detectors": [d.value for d in self.triggered_detectors],
            "recommended_action": self.recommended_action,
            "created_at": self.created_at.isoformat(),
        }

