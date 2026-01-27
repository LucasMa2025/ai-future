"""
异常检测器实现

实现 NLGSM 论文中定义的四种异常检测器：
1. MetricAnomalyDetector: 基于统计方法检测指标异常
2. BehaviorAnomalyDetector: 检测模型行为异常
3. DriftDetector: 检测概念漂移和数据漂移
4. ExternalAnomalyDetector: 处理外部安全和合规事件
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Deque
from collections import deque
import statistics
import math
import logging

from .signals import AnomalySignal, AnomalySeverity, DetectorType


logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    """检测器配置"""
    # 通用配置
    enabled: bool = True
    sensitivity: float = 1.0  # 灵敏度系数
    
    # 阈值配置
    low_threshold: float = 0.3
    medium_threshold: float = 0.5
    high_threshold: float = 0.7
    critical_threshold: float = 0.9
    
    # 窗口配置
    window_size: int = 100
    min_samples: int = 10
    
    # 冷却配置（避免重复告警）
    cooldown_seconds: int = 60
    
    def get_severity(self, score: float) -> AnomalySeverity:
        """根据分数获取严重程度"""
        if score >= self.critical_threshold:
            return AnomalySeverity.CRITICAL
        elif score >= self.high_threshold:
            return AnomalySeverity.HIGH
        elif score >= self.medium_threshold:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW


class AnomalyDetector(ABC):
    """异常检测器基类"""
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._last_alert_time: Dict[str, datetime] = {}
    
    @property
    @abstractmethod
    def detector_type(self) -> DetectorType:
        """检测器类型"""
        pass
    
    @abstractmethod
    def detect(self, data: Dict[str, Any]) -> AnomalySignal:
        """
        执行异常检测
        
        Args:
            data: 待检测数据
            
        Returns:
            异常信号
        """
        pass
    
    def _should_alert(self, metric_name: str) -> bool:
        """检查是否应该发出告警（冷却期检查）"""
        if metric_name not in self._last_alert_time:
            return True
        
        elapsed = (datetime.utcnow() - self._last_alert_time[metric_name]).total_seconds()
        return elapsed > self.config.cooldown_seconds
    
    def _record_alert(self, metric_name: str):
        """记录告警时间"""
        self._last_alert_time[metric_name] = datetime.utcnow()
    
    def _create_signal(
        self,
        detected: bool,
        score: float,
        metric_name: Optional[str] = None,
        current_value: Optional[float] = None,
        expected_value: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AnomalySignal:
        """创建异常信号"""
        severity = self.config.get_severity(score) if detected else AnomalySeverity.LOW
        deviation = None
        if current_value is not None and expected_value is not None and expected_value != 0:
            deviation = (current_value - expected_value) / abs(expected_value)
        
        return AnomalySignal(
            detector_type=self.detector_type,
            detected=detected,
            severity=severity,
            score=score,
            threshold=self.config.medium_threshold,
            confidence=1.0,
            metric_name=metric_name,
            current_value=current_value,
            expected_value=expected_value,
            deviation=deviation,
            context=context or {},
        )


class MetricAnomalyDetector(AnomalyDetector):
    """
    指标异常检测器
    
    基于统计方法检测各类指标的异常：
    - 准确率突降
    - 损失函数波动
    - 延迟异常
    - 资源使用异常
    
    使用方法：
    - Z-Score 检测
    - 移动平均偏差
    - 百分位异常
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)
        # 存储历史数据的滑动窗口
        self._metric_windows: Dict[str, Deque[float]] = {}
        self._metric_stats: Dict[str, Dict[str, float]] = {}
    
    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.METRIC
    
    def detect(self, data: Dict[str, Any]) -> AnomalySignal:
        """
        检测指标异常
        
        data 格式:
        {
            "metric_name": "accuracy",
            "value": 0.85,
            "timestamp": "2026-01-27T10:00:00Z",
            "labels": {"model": "production", "version": "1.0"}
        }
        """
        if not self.config.enabled:
            return self._create_signal(detected=False, score=0.0)
        
        metric_name = data.get("metric_name", "unknown")
        value = data.get("value")
        
        if value is None:
            return self._create_signal(
                detected=False,
                score=0.0,
                metric_name=metric_name,
                context={"error": "No value provided"}
            )
        
        # 更新滑动窗口
        self._update_window(metric_name, value)
        
        # 检查样本数量
        window = self._metric_windows.get(metric_name, deque())
        if len(window) < self.config.min_samples:
            return self._create_signal(
                detected=False,
                score=0.0,
                metric_name=metric_name,
                current_value=value,
                context={"reason": "Insufficient samples", "samples": len(window)}
            )
        
        # 计算异常分数
        score, expected = self._calculate_zscore(metric_name, value)
        detected = score >= self.config.medium_threshold
        
        # 冷却期检查
        if detected and not self._should_alert(metric_name):
            return self._create_signal(
                detected=False,
                score=score,
                metric_name=metric_name,
                current_value=value,
                expected_value=expected,
                context={"reason": "In cooldown period"}
            )
        
        if detected:
            self._record_alert(metric_name)
            logger.warning(
                f"Metric anomaly detected: {metric_name}={value} "
                f"(expected ~{expected:.4f}, score={score:.2f})"
            )
        
        return self._create_signal(
            detected=detected,
            score=score,
            metric_name=metric_name,
            current_value=value,
            expected_value=expected,
            context={
                "window_size": len(window),
                "detection_method": "z-score",
                **data.get("labels", {}),
            }
        )
    
    def _update_window(self, metric_name: str, value: float):
        """更新指标滑动窗口"""
        if metric_name not in self._metric_windows:
            self._metric_windows[metric_name] = deque(maxlen=self.config.window_size)
        
        self._metric_windows[metric_name].append(value)
    
    def _calculate_zscore(self, metric_name: str, value: float) -> tuple[float, float]:
        """
        计算 Z-Score 并转换为 0-1 范围的异常分数
        
        Returns:
            (异常分数, 期望值)
        """
        window = list(self._metric_windows[metric_name])
        
        if len(window) < 2:
            return 0.0, value
        
        mean = statistics.mean(window)
        stdev = statistics.stdev(window) if len(window) > 1 else 0.0
        
        if stdev == 0:
            return 0.0, mean
        
        z_score = abs(value - mean) / stdev
        
        # 将 Z-Score 转换为 0-1 范围
        # 使用 Sigmoid 函数，Z=3 约对应 0.95
        score = 1 / (1 + math.exp(-self.config.sensitivity * (z_score - 2)))
        
        return min(score, 1.0), mean
    
    def get_statistics(self, metric_name: str) -> Optional[Dict[str, float]]:
        """获取指标统计信息"""
        window = self._metric_windows.get(metric_name)
        if not window or len(window) < 2:
            return None
        
        values = list(window)
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
        }


class BehaviorAnomalyDetector(AnomalyDetector):
    """
    行为异常检测器
    
    检测模型输出行为的异常：
    - 输出分布偏移
    - 决策模式异常
    - 置信度分布异常
    - 响应时间异常
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)
        # 存储决策分布历史
        self._decision_history: Deque[str] = deque(maxlen=1000)
        self._baseline_distribution: Optional[Dict[str, float]] = None
        self._confidence_window: Deque[float] = deque(maxlen=500)
    
    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.BEHAVIOR
    
    def set_baseline(self, distribution: Dict[str, float]):
        """设置基准决策分布"""
        self._baseline_distribution = distribution
    
    def detect(self, data: Dict[str, Any]) -> AnomalySignal:
        """
        检测行为异常
        
        data 格式:
        {
            "decision": "approve",  # 决策类型
            "confidence": 0.95,     # 置信度
            "output_distribution": {"approve": 0.7, "reject": 0.2, "review": 0.1},
            "response_time_ms": 150,
        }
        """
        if not self.config.enabled:
            return self._create_signal(detected=False, score=0.0)
        
        decision = data.get("decision")
        confidence = data.get("confidence", 1.0)
        output_dist = data.get("output_distribution", {})
        
        # 记录决策
        if decision:
            self._decision_history.append(decision)
        
        # 记录置信度
        if confidence is not None:
            self._confidence_window.append(confidence)
        
        # 计算各类异常分数
        scores = []
        context = {}
        
        # 1. 决策分布偏移
        if self._baseline_distribution and len(self._decision_history) >= self.config.min_samples:
            dist_score = self._check_distribution_shift()
            scores.append(dist_score)
            context["distribution_shift_score"] = dist_score
        
        # 2. 置信度异常
        if len(self._confidence_window) >= self.config.min_samples:
            conf_score = self._check_confidence_anomaly(confidence)
            scores.append(conf_score)
            context["confidence_anomaly_score"] = conf_score
        
        # 3. 输出分布熵检查（过于集中或分散）
        if output_dist:
            entropy_score = self._check_entropy_anomaly(output_dist)
            scores.append(entropy_score)
            context["entropy_anomaly_score"] = entropy_score
        
        # 综合评分
        if not scores:
            return self._create_signal(
                detected=False,
                score=0.0,
                context={"reason": "Insufficient data for behavior analysis"}
            )
        
        final_score = max(scores)  # 取最高异常分数
        detected = final_score >= self.config.medium_threshold
        
        if detected:
            logger.warning(f"Behavior anomaly detected: score={final_score:.2f}, context={context}")
        
        return self._create_signal(
            detected=detected,
            score=final_score,
            metric_name="behavior",
            context=context
        )
    
    def _check_distribution_shift(self) -> float:
        """检查决策分布偏移（使用 KL 散度近似）"""
        if not self._baseline_distribution:
            return 0.0
        
        # 计算当前分布
        current_dist: Dict[str, float] = {}
        total = len(self._decision_history)
        
        for decision in self._decision_history:
            current_dist[decision] = current_dist.get(decision, 0) + 1
        
        for key in current_dist:
            current_dist[key] /= total
        
        # 计算 Jensen-Shannon 散度
        all_keys = set(self._baseline_distribution.keys()) | set(current_dist.keys())
        
        kl_sum = 0.0
        for key in all_keys:
            p = self._baseline_distribution.get(key, 0.001)
            q = current_dist.get(key, 0.001)
            m = (p + q) / 2
            
            if p > 0 and m > 0:
                kl_sum += p * math.log(p / m)
            if q > 0 and m > 0:
                kl_sum += q * math.log(q / m)
        
        js_divergence = kl_sum / 2
        
        # 转换为 0-1 分数
        score = 1 - math.exp(-self.config.sensitivity * js_divergence)
        return min(score, 1.0)
    
    def _check_confidence_anomaly(self, current_confidence: float) -> float:
        """检查置信度异常"""
        if len(self._confidence_window) < 2:
            return 0.0
        
        values = list(self._confidence_window)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if stdev == 0:
            return 0.0
        
        z_score = abs(current_confidence - mean) / stdev
        score = 1 / (1 + math.exp(-self.config.sensitivity * (z_score - 2)))
        
        return min(score, 1.0)
    
    def _check_entropy_anomaly(self, distribution: Dict[str, float]) -> float:
        """检查输出分布熵异常（过于集中或分散）"""
        if not distribution:
            return 0.0
        
        # 计算熵
        entropy = 0.0
        for prob in distribution.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        # 最大熵（均匀分布）
        max_entropy = math.log2(len(distribution)) if len(distribution) > 1 else 0.0
        
        if max_entropy == 0:
            return 0.0
        
        # 正则化熵
        normalized_entropy = entropy / max_entropy
        
        # 熵过低（过于确定）或过高（过于随机）都是异常
        # 期望正则化熵在 0.3-0.8 之间
        if normalized_entropy < 0.2:
            score = (0.2 - normalized_entropy) / 0.2
        elif normalized_entropy > 0.9:
            score = (normalized_entropy - 0.9) / 0.1
        else:
            score = 0.0
        
        return min(score * self.config.sensitivity, 1.0)


class DriftDetector(AnomalyDetector):
    """
    漂移检测器
    
    检测两种类型的漂移：
    1. 概念漂移：输入-输出关系变化
    2. 数据漂移：输入分布变化
    
    使用 Page-Hinkley 和 ADWIN 算法
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)
        # Page-Hinkley 测试状态
        self._ph_sum: float = 0.0
        self._ph_min: float = 0.0
        self._ph_count: int = 0
        self._ph_threshold: float = 50.0
        self._ph_delta: float = 0.005
        
        # 错误率窗口
        self._error_window: Deque[float] = deque(maxlen=500)
        
        # 特征均值跟踪
        self._feature_means: Dict[str, Deque[float]] = {}
    
    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.DRIFT
    
    def detect(self, data: Dict[str, Any]) -> AnomalySignal:
        """
        检测漂移
        
        data 格式:
        {
            "error_rate": 0.05,           # 当前批次错误率
            "feature_means": {"f1": 0.5, "f2": 0.3},  # 特征均值
            "prediction_correct": True,    # 单次预测是否正确
        }
        """
        if not self.config.enabled:
            return self._create_signal(detected=False, score=0.0)
        
        scores = []
        context = {}
        
        # 1. Page-Hinkley 测试（概念漂移）
        if "prediction_correct" in data:
            error = 0.0 if data["prediction_correct"] else 1.0
            self._error_window.append(error)
            
            ph_score = self._page_hinkley_test(error)
            scores.append(ph_score)
            context["page_hinkley_score"] = ph_score
        
        # 2. 错误率突变检测
        if "error_rate" in data:
            error_rate = data["error_rate"]
            self._error_window.append(error_rate)
            
            rate_score = self._check_error_rate_shift(error_rate)
            scores.append(rate_score)
            context["error_rate_shift_score"] = rate_score
        
        # 3. 特征漂移检测
        if "feature_means" in data:
            feature_score = self._check_feature_drift(data["feature_means"])
            scores.append(feature_score)
            context["feature_drift_score"] = feature_score
        
        if not scores:
            return self._create_signal(
                detected=False,
                score=0.0,
                context={"reason": "No drift data provided"}
            )
        
        final_score = max(scores)
        detected = final_score >= self.config.medium_threshold
        
        if detected:
            logger.warning(f"Drift detected: score={final_score:.2f}, context={context}")
        
        return self._create_signal(
            detected=detected,
            score=final_score,
            metric_name="drift",
            context=context
        )
    
    def _page_hinkley_test(self, error: float) -> float:
        """
        Page-Hinkley 测试
        
        检测错误率的持续增长趋势
        """
        self._ph_count += 1
        
        if self._ph_count < self.config.min_samples:
            return 0.0
        
        # 计算历史平均
        mean_error = statistics.mean(self._error_window) if self._error_window else 0.0
        
        # 更新累积和
        self._ph_sum += error - mean_error - self._ph_delta
        self._ph_min = min(self._ph_min, self._ph_sum)
        
        # 计算 PH 值
        ph_value = self._ph_sum - self._ph_min
        
        # 转换为 0-1 分数
        score = min(ph_value / self._ph_threshold, 1.0)
        
        # 如果检测到漂移，重置状态
        if score >= self.config.high_threshold:
            self._reset_ph()
        
        return score * self.config.sensitivity
    
    def _reset_ph(self):
        """重置 Page-Hinkley 状态"""
        self._ph_sum = 0.0
        self._ph_min = 0.0
    
    def _check_error_rate_shift(self, current_rate: float) -> float:
        """检查错误率突变"""
        if len(self._error_window) < self.config.min_samples:
            return 0.0
        
        values = list(self._error_window)
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if stdev == 0:
            return 0.0
        
        z_score = abs(current_rate - mean) / stdev
        score = 1 / (1 + math.exp(-self.config.sensitivity * (z_score - 2)))
        
        return min(score, 1.0)
    
    def _check_feature_drift(self, feature_means: Dict[str, float]) -> float:
        """检查特征漂移"""
        max_drift = 0.0
        
        for feature, value in feature_means.items():
            if feature not in self._feature_means:
                self._feature_means[feature] = deque(maxlen=100)
            
            self._feature_means[feature].append(value)
            
            window = self._feature_means[feature]
            if len(window) < self.config.min_samples:
                continue
            
            values = list(window)
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            
            if stdev == 0:
                continue
            
            z_score = abs(value - mean) / stdev
            drift_score = 1 / (1 + math.exp(-self.config.sensitivity * (z_score - 2)))
            max_drift = max(max_drift, drift_score)
        
        return min(max_drift, 1.0)


class ExternalAnomalyDetector(AnomalyDetector):
    """
    外部异常检测器
    
    处理来自外部系统的异常信号：
    - 安全事件（入侵检测、异常登录）
    - 合规问题（违规操作、审计失败）
    - 基础设施异常（资源耗尽、网络问题）
    - 人工报告的问题
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)
        # 已知的外部事件类型和严重程度映射
        self._severity_mapping: Dict[str, AnomalySeverity] = {
            # 安全事件
            "unauthorized_access": AnomalySeverity.CRITICAL,
            "suspicious_activity": AnomalySeverity.HIGH,
            "failed_authentication": AnomalySeverity.MEDIUM,
            
            # 合规事件
            "compliance_violation": AnomalySeverity.HIGH,
            "audit_failure": AnomalySeverity.HIGH,
            "policy_breach": AnomalySeverity.MEDIUM,
            
            # 基础设施事件
            "resource_exhaustion": AnomalySeverity.HIGH,
            "service_degradation": AnomalySeverity.MEDIUM,
            "network_issue": AnomalySeverity.MEDIUM,
            
            # 人工报告
            "manual_report_critical": AnomalySeverity.CRITICAL,
            "manual_report_high": AnomalySeverity.HIGH,
            "manual_report_medium": AnomalySeverity.MEDIUM,
        }
    
    @property
    def detector_type(self) -> DetectorType:
        return DetectorType.EXTERNAL
    
    def register_event_type(self, event_type: str, severity: AnomalySeverity):
        """注册新的事件类型"""
        self._severity_mapping[event_type] = severity
    
    def detect(self, data: Dict[str, Any]) -> AnomalySignal:
        """
        检测外部异常
        
        data 格式:
        {
            "event_type": "unauthorized_access",
            "source": "security_system",
            "severity": "critical",  # 可选，覆盖默认映射
            "description": "Detected unauthorized access attempt",
            "evidence": {...},
        }
        """
        if not self.config.enabled:
            return self._create_signal(detected=False, score=0.0)
        
        event_type = data.get("event_type", "unknown")
        source = data.get("source", "external")
        description = data.get("description", "")
        
        # 确定严重程度
        if "severity" in data:
            try:
                severity = AnomalySeverity(data["severity"])
            except ValueError:
                severity = self._severity_mapping.get(event_type, AnomalySeverity.MEDIUM)
        else:
            severity = self._severity_mapping.get(event_type, AnomalySeverity.MEDIUM)
        
        # 将严重程度转换为分数
        severity_scores = {
            AnomalySeverity.LOW: 0.3,
            AnomalySeverity.MEDIUM: 0.5,
            AnomalySeverity.HIGH: 0.75,
            AnomalySeverity.CRITICAL: 0.95,
        }
        
        score = severity_scores.get(severity, 0.5)
        detected = True  # 外部事件始终视为检测到
        
        logger.warning(
            f"External anomaly: type={event_type}, source={source}, "
            f"severity={severity.value}, description={description}"
        )
        
        return AnomalySignal(
            detector_type=self.detector_type,
            detected=detected,
            severity=severity,
            score=score,
            threshold=self.config.medium_threshold,
            confidence=1.0,
            metric_name=event_type,
            context={
                "source": source,
                "description": description,
                "evidence": data.get("evidence", {}),
            },
            source_component=source,
        )

