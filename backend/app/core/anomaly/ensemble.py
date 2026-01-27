"""
异常检测集成器

将多个检测器的结果聚合，产生综合的异常判断和响应建议
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
import logging

from .detectors import (
    AnomalyDetector,
    MetricAnomalyDetector,
    BehaviorAnomalyDetector,
    DriftDetector,
    ExternalAnomalyDetector,
    DetectorConfig,
)
from .signals import (
    AnomalySignal,
    CompositeAnomalySignal,
    AnomalySeverity,
    DetectorType,
)


logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """集成器配置"""
    # 各检测器权重
    weights: Dict[DetectorType, float] = field(default_factory=lambda: {
        DetectorType.METRIC: 1.0,
        DetectorType.BEHAVIOR: 1.5,
        DetectorType.DRIFT: 1.5,
        DetectorType.EXTERNAL: 2.0,
    })
    
    # 聚合策略
    aggregation_strategy: str = "max"  # max, weighted_avg, voting
    
    # 触发阈值
    trigger_threshold: float = 0.5
    
    # 最少需要触发的检测器数量
    min_triggered_detectors: int = 1


class AnomalyEnsemble:
    """
    异常检测集成器
    
    协调多个检测器的运行，聚合结果，产生统一的异常响应
    
    使用示例:
    ```python
    ensemble = AnomalyEnsemble()
    
    # 检测
    result = ensemble.detect({
        "metrics": {"metric_name": "accuracy", "value": 0.75},
        "behavior": {"decision": "approve", "confidence": 0.9},
        "drift": {"error_rate": 0.1},
    })
    
    if result.final_severity >= AnomalySeverity.HIGH:
        # 触发告警或回滚
        pass
    ```
    """
    
    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        detector_config: Optional[DetectorConfig] = None,
    ):
        self.config = config or EnsembleConfig()
        detector_cfg = detector_config or DetectorConfig()
        
        # 初始化各检测器
        self.detectors: Dict[DetectorType, AnomalyDetector] = {
            DetectorType.METRIC: MetricAnomalyDetector(detector_cfg),
            DetectorType.BEHAVIOR: BehaviorAnomalyDetector(detector_cfg),
            DetectorType.DRIFT: DriftDetector(detector_cfg),
            DetectorType.EXTERNAL: ExternalAnomalyDetector(detector_cfg),
        }
        
        # 检测历史
        self._detection_history: List[CompositeAnomalySignal] = []
    
    def detect(self, data: Dict[str, Any]) -> CompositeAnomalySignal:
        """
        运行所有检测器并聚合结果
        
        Args:
            data: 包含各检测器所需数据的字典
                {
                    "metrics": {...},    # 给 MetricAnomalyDetector
                    "behavior": {...},   # 给 BehaviorAnomalyDetector
                    "drift": {...},      # 给 DriftDetector
                    "external": {...},   # 给 ExternalAnomalyDetector
                }
        
        Returns:
            复合异常信号
        """
        composite = CompositeAnomalySignal()
        
        # 运行各检测器
        detector_data_map = {
            DetectorType.METRIC: data.get("metrics", {}),
            DetectorType.BEHAVIOR: data.get("behavior", {}),
            DetectorType.DRIFT: data.get("drift", {}),
            DetectorType.EXTERNAL: data.get("external", {}),
        }
        
        for detector_type, detector in self.detectors.items():
            detector_data = detector_data_map.get(detector_type, {})
            
            if not detector_data:
                continue
            
            try:
                signal = detector.detect(detector_data)
                composite.add_signal(signal)
            except Exception as e:
                logger.error(f"Detector {detector_type.value} failed: {e}")
                # 创建错误信号
                error_signal = AnomalySignal(
                    detector_type=detector_type,
                    detected=False,
                    severity=AnomalySeverity.LOW,
                    context={"error": str(e)},
                )
                composite.add_signal(error_signal)
        
        # 计算复合分数
        self._aggregate_scores(composite)
        
        # 生成推荐动作
        composite.recommended_action = self._recommend_action(composite)
        
        # 记录历史
        self._detection_history.append(composite)
        if len(self._detection_history) > 1000:
            self._detection_history = self._detection_history[-500:]
        
        return composite
    
    def detect_single(
        self,
        detector_type: DetectorType,
        data: Dict[str, Any]
    ) -> AnomalySignal:
        """运行单个检测器"""
        if detector_type not in self.detectors:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        return self.detectors[detector_type].detect(data)
    
    def _aggregate_scores(self, composite: CompositeAnomalySignal):
        """聚合各检测器的分数"""
        if self.config.aggregation_strategy == "max":
            # 取最高分
            max_score = 0.0
            for signal in composite.signals:
                if signal.detected:
                    max_score = max(max_score, signal.score)
            composite.composite_score = max_score
        
        elif self.config.aggregation_strategy == "weighted_avg":
            # 加权平均
            total_weight = 0.0
            weighted_sum = 0.0
            
            for signal in composite.signals:
                if signal.detected:
                    weight = self.config.weights.get(signal.detector_type, 1.0)
                    weighted_sum += signal.score * weight
                    total_weight += weight
            
            composite.composite_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        elif self.config.aggregation_strategy == "voting":
            # 投票机制
            votes = len([s for s in composite.signals if s.detected])
            total_detectors = len(composite.signals)
            composite.composite_score = votes / total_detectors if total_detectors > 0 else 0.0
    
    def _recommend_action(self, composite: CompositeAnomalySignal) -> str:
        """
        基于复合信号推荐响应动作
        
        按照 NLGSM 论文的响应矩阵:
        - LOG: 仅记录
        - DIAGNOSE: 需要诊断
        - ROLLBACK: 需要回滚
        - HALT: 紧急停机
        """
        triggered_count = len(composite.triggered_detectors)
        severity = composite.final_severity
        score = composite.composite_score
        
        # 检查是否达到最低触发数量
        if triggered_count < self.config.min_triggered_detectors:
            return "LOG"
        
        # 根据严重程度决定动作
        if severity == AnomalySeverity.CRITICAL:
            return "HALT"
        
        elif severity == AnomalySeverity.HIGH:
            # 高严重度且多个检测器触发
            if triggered_count >= 2:
                return "ROLLBACK"
            return "DIAGNOSE"
        
        elif severity == AnomalySeverity.MEDIUM:
            if triggered_count >= 3:
                return "ROLLBACK"
            elif triggered_count >= 2:
                return "DIAGNOSE"
            return "LOG"
        
        else:
            return "LOG"
    
    def get_detector(self, detector_type: DetectorType) -> Optional[AnomalyDetector]:
        """获取特定检测器"""
        return self.detectors.get(detector_type)
    
    def set_behavior_baseline(self, distribution: Dict[str, float]):
        """设置行为检测器的基准分布"""
        behavior_detector = self.detectors.get(DetectorType.BEHAVIOR)
        if isinstance(behavior_detector, BehaviorAnomalyDetector):
            behavior_detector.set_baseline(distribution)
    
    def register_external_event_type(self, event_type: str, severity: AnomalySeverity):
        """注册外部事件类型"""
        external_detector = self.detectors.get(DetectorType.EXTERNAL)
        if isinstance(external_detector, ExternalAnomalyDetector):
            external_detector.register_event_type(event_type, severity)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取检测统计"""
        total_detections = len(self._detection_history)
        triggered_count = sum(
            1 for c in self._detection_history if len(c.triggered_detectors) > 0
        )
        
        severity_counts = {s.value: 0 for s in AnomalySeverity}
        action_counts: Dict[str, int] = {}
        
        for composite in self._detection_history:
            if composite.triggered_detectors:
                severity_counts[composite.final_severity.value] += 1
            if composite.recommended_action:
                action_counts[composite.recommended_action] = \
                    action_counts.get(composite.recommended_action, 0) + 1
        
        return {
            "total_detections": total_detections,
            "anomalies_detected": triggered_count,
            "detection_rate": triggered_count / total_detections if total_detections > 0 else 0,
            "severity_distribution": severity_counts,
            "action_distribution": action_counts,
        }
    
    def get_recent_anomalies(
        self,
        limit: int = 20,
        min_severity: Optional[AnomalySeverity] = None
    ) -> List[CompositeAnomalySignal]:
        """获取最近的异常"""
        anomalies = [c for c in self._detection_history if len(c.triggered_detectors) > 0]
        
        if min_severity:
            severity_order = list(AnomalySeverity)
            min_index = severity_order.index(min_severity)
            anomalies = [
                c for c in anomalies
                if severity_order.index(c.final_severity) >= min_index
            ]
        
        return sorted(anomalies, key=lambda x: x.created_at, reverse=True)[:limit]

