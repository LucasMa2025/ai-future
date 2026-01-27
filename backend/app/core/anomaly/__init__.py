"""
多维异常检测系统

实现 NLGSM 论文定义的四种异常检测器：
1. MetricAnomalyDetector: 指标异常检测（准确率、损失波动等）
2. BehaviorAnomalyDetector: 行为异常检测（输出分布偏移）
3. DriftDetector: 漂移检测（概念漂移、数据漂移）
4. ExternalAnomalyDetector: 外部异常检测（安全事件、合规问题）

参考论文 Section 4: Multi-Dimensional Anomaly Detection
"""

from .detectors import (
    AnomalyDetector,
    MetricAnomalyDetector,
    BehaviorAnomalyDetector,
    DriftDetector,
    ExternalAnomalyDetector,
)
from .ensemble import AnomalyEnsemble
from .signals import AnomalySignal, AnomalySeverity, DetectorType
from .response import AnomalyResponse, ResponseDecision

__all__ = [
    # 检测器
    "AnomalyDetector",
    "MetricAnomalyDetector",
    "BehaviorAnomalyDetector",
    "DriftDetector",
    "ExternalAnomalyDetector",
    # 集成
    "AnomalyEnsemble",
    # 信号
    "AnomalySignal",
    "AnomalySeverity",
    "DetectorType",
    # 响应
    "AnomalyResponse",
    "ResponseDecision",
]

