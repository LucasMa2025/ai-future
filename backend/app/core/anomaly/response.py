"""
异常响应决策

根据异常检测结果自动决策响应动作，并与 NLGSM 状态机集成
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from uuid import UUID, uuid4
import logging

from .signals import CompositeAnomalySignal, AnomalySeverity, DetectorType


logger = logging.getLogger(__name__)


class ResponseDecision(str, Enum):
    """响应决策类型"""
    LOG = "log"              # 仅记录
    ALERT = "alert"          # 发送告警
    DIAGNOSE = "diagnose"    # 进入诊断模式
    ROLLBACK = "rollback"    # 回滚到安全状态
    HALT = "halt"            # 紧急停机


@dataclass
class AnomalyResponse:
    """
    异常响应
    
    包含决策、动作和执行状态
    """
    id: UUID = field(default_factory=uuid4)
    
    # 触发源
    anomaly_signal: Optional[CompositeAnomalySignal] = None
    
    # 决策
    decision: ResponseDecision = ResponseDecision.LOG
    reason: str = ""
    
    # 动作列表
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 执行状态
    executed: bool = False
    executed_at: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    
    # 人工干预
    requires_human_review: bool = False
    human_reviewed: bool = False
    reviewer_id: Optional[UUID] = None
    review_decision: Optional[str] = None
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": str(self.id),
            "decision": self.decision.value,
            "reason": self.reason,
            "actions": self.actions,
            "executed": self.executed,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_result": self.execution_result,
            "requires_human_review": self.requires_human_review,
            "human_reviewed": self.human_reviewed,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ResponseRule:
    """响应规则"""
    name: str
    condition: Callable[[CompositeAnomalySignal], bool]
    decision: ResponseDecision
    actions: List[str]
    requires_human_review: bool = False
    priority: int = 0


class AnomalyResponseDecider:
    """
    异常响应决策器
    
    根据复合异常信号自动决策响应动作
    
    决策流程:
    1. 评估异常严重程度
    2. 匹配响应规则
    3. 生成响应动作
    4. 与状态机集成
    """
    
    def __init__(self):
        self._rules: List[ResponseRule] = self._init_default_rules()
        self._action_handlers: Dict[str, Callable] = {}
        self._response_history: List[AnomalyResponse] = []
    
    def _init_default_rules(self) -> List[ResponseRule]:
        """初始化默认响应规则"""
        return [
            # 最高优先级：严重异常触发停机
            ResponseRule(
                name="critical_halt",
                condition=lambda s: s.final_severity == AnomalySeverity.CRITICAL,
                decision=ResponseDecision.HALT,
                actions=["freeze_all", "trigger_safe_halt", "notify_governance_committee"],
                requires_human_review=True,
                priority=100,
            ),
            
            # 高优先级：多检测器触发回滚
            ResponseRule(
                name="multi_detector_rollback",
                condition=lambda s: (
                    len(s.triggered_detectors) >= 2 and
                    s.final_severity >= AnomalySeverity.HIGH
                ),
                decision=ResponseDecision.ROLLBACK,
                actions=["freeze_learning", "restore_checkpoint", "notify_operators"],
                requires_human_review=True,
                priority=90,
            ),
            
            # 高严重度：回滚
            ResponseRule(
                name="high_severity_rollback",
                condition=lambda s: s.final_severity == AnomalySeverity.HIGH,
                decision=ResponseDecision.ROLLBACK,
                actions=["freeze_learning", "restore_checkpoint", "notify_operators"],
                requires_human_review=False,
                priority=80,
            ),
            
            # 行为异常：进入诊断
            ResponseRule(
                name="behavior_anomaly_diagnose",
                condition=lambda s: DetectorType.BEHAVIOR in s.triggered_detectors,
                decision=ResponseDecision.DIAGNOSE,
                actions=["collect_diagnostics", "analyze_behavior", "notify_ml_engineers"],
                requires_human_review=False,
                priority=60,
            ),
            
            # 漂移检测：进入诊断
            ResponseRule(
                name="drift_detected_diagnose",
                condition=lambda s: DetectorType.DRIFT in s.triggered_detectors,
                decision=ResponseDecision.DIAGNOSE,
                actions=["collect_diagnostics", "analyze_drift", "notify_ml_engineers"],
                requires_human_review=False,
                priority=50,
            ),
            
            # 中等严重度：告警
            ResponseRule(
                name="medium_severity_alert",
                condition=lambda s: s.final_severity == AnomalySeverity.MEDIUM,
                decision=ResponseDecision.ALERT,
                actions=["send_alert", "log_detailed"],
                requires_human_review=False,
                priority=30,
            ),
            
            # 外部事件：根据类型处理
            ResponseRule(
                name="external_event_alert",
                condition=lambda s: DetectorType.EXTERNAL in s.triggered_detectors,
                decision=ResponseDecision.ALERT,
                actions=["send_alert", "log_external_event"],
                requires_human_review=True,
                priority=40,
            ),
            
            # 默认：记录日志
            ResponseRule(
                name="default_log",
                condition=lambda s: True,
                decision=ResponseDecision.LOG,
                actions=["log_anomaly"],
                requires_human_review=False,
                priority=0,
            ),
        ]
    
    def register_rule(self, rule: ResponseRule):
        """注册自定义响应规则"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)
    
    def register_action_handler(self, action_name: str, handler: Callable):
        """注册动作处理器"""
        self._action_handlers[action_name] = handler
    
    def decide(self, signal: CompositeAnomalySignal) -> AnomalyResponse:
        """
        决策响应动作
        
        Args:
            signal: 复合异常信号
            
        Returns:
            异常响应
        """
        # 如果没有检测到异常，直接返回 LOG
        if not signal.triggered_detectors:
            return AnomalyResponse(
                anomaly_signal=signal,
                decision=ResponseDecision.LOG,
                reason="No anomaly detected",
                actions=[{"name": "log_check", "status": "skipped"}],
            )
        
        # 匹配规则
        matched_rule: Optional[ResponseRule] = None
        for rule in self._rules:
            try:
                if rule.condition(signal):
                    matched_rule = rule
                    break
            except Exception as e:
                logger.error(f"Rule {rule.name} evaluation failed: {e}")
                continue
        
        if not matched_rule:
            # 使用默认规则
            matched_rule = self._rules[-1]
        
        # 创建响应
        response = AnomalyResponse(
            anomaly_signal=signal,
            decision=matched_rule.decision,
            reason=f"Matched rule: {matched_rule.name}",
            requires_human_review=matched_rule.requires_human_review,
        )
        
        # 生成动作列表
        for action_name in matched_rule.actions:
            response.actions.append({
                "name": action_name,
                "status": "pending",
                "params": {},
            })
        
        logger.info(
            f"Anomaly response decided: decision={matched_rule.decision.value}, "
            f"rule={matched_rule.name}, actions={matched_rule.actions}"
        )
        
        # 记录历史
        self._response_history.append(response)
        
        return response
    
    def execute_response(self, response: AnomalyResponse) -> AnomalyResponse:
        """
        执行响应动作
        
        Args:
            response: 异常响应
            
        Returns:
            更新后的响应
        """
        if response.executed:
            logger.warning(f"Response {response.id} already executed")
            return response
        
        results = []
        all_success = True
        
        for action in response.actions:
            action_name = action["name"]
            handler = self._action_handlers.get(action_name)
            
            if handler:
                try:
                    result = handler(response.anomaly_signal, action.get("params", {}))
                    action["status"] = "completed"
                    action["result"] = result
                except Exception as e:
                    action["status"] = "failed"
                    action["error"] = str(e)
                    all_success = False
                    logger.error(f"Action {action_name} failed: {e}")
            else:
                # 没有处理器，标记为跳过
                action["status"] = "skipped"
                action["reason"] = "No handler registered"
            
            results.append(action)
        
        response.executed = True
        response.executed_at = datetime.utcnow()
        response.execution_result = {
            "success": all_success,
            "actions": results,
        }
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取响应统计"""
        decision_counts = {d.value: 0 for d in ResponseDecision}
        
        for response in self._response_history:
            decision_counts[response.decision.value] += 1
        
        return {
            "total_responses": len(self._response_history),
            "decision_distribution": decision_counts,
            "pending_review": sum(
                1 for r in self._response_history
                if r.requires_human_review and not r.human_reviewed
            ),
        }


class AnomalyResponseIntegration:
    """
    异常响应与 NLGSM 状态机的集成
    
    将异常响应动作转换为状态机事件
    """
    
    def __init__(self, state_machine_service=None):
        self.state_machine = state_machine_service
    
    def integrate_response(self, response: AnomalyResponse) -> Dict[str, Any]:
        """
        将响应集成到状态机
        
        Args:
            response: 异常响应
            
        Returns:
            集成结果
        """
        if not self.state_machine:
            return {"integrated": False, "reason": "No state machine configured"}
        
        result = {"integrated": True, "events": []}
        
        # 根据决策类型触发相应事件
        if response.decision == ResponseDecision.HALT:
            # 触发紧急停机事件
            event_result = self._trigger_event(
                "ANOMALY",
                {"severity": "critical", "response_id": str(response.id)}
            )
            result["events"].append(event_result)
        
        elif response.decision == ResponseDecision.ROLLBACK:
            # 触发回滚事件
            event_result = self._trigger_event(
                "ANOMALY",
                {"severity": "high", "response_id": str(response.id)}
            )
            result["events"].append(event_result)
        
        elif response.decision == ResponseDecision.DIAGNOSE:
            # 触发诊断事件
            event_result = self._trigger_event(
                "START_DIAGNOSIS",
                {"response_id": str(response.id)}
            )
            result["events"].append(event_result)
        
        return result
    
    def _trigger_event(self, event_type: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """触发状态机事件"""
        try:
            from ..enums import EventType
            from ...services.state_machine_service import Event
            
            event = Event(
                event_type=EventType(event_type.lower()),
                source="anomaly_response",
                metadata=metadata,
            )
            
            result = self.state_machine.process_event(event)
            return {
                "event_type": event_type,
                "success": result.success,
                "to_state": result.to_state.value if result.to_state else None,
            }
        except Exception as e:
            logger.error(f"Failed to trigger event {event_type}: {e}")
            return {
                "event_type": event_type,
                "success": False,
                "error": str(e),
            }

