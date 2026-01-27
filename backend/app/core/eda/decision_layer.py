"""
EDA 决策层

负责根据事件做出决策
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from uuid import UUID, uuid4
import logging

from .event_layer import Event, EventType, EventPriority


logger = logging.getLogger(__name__)


class Decision(str, Enum):
    """决策类型"""
    ALLOW = "allow"           # 允许继续
    DENY = "deny"             # 拒绝
    ESCALATE = "escalate"     # 升级到人工
    ROLLBACK = "rollback"     # 回滚
    HALT = "halt"             # 停机
    DIAGNOSE = "diagnose"     # 进入诊断
    CONTINUE = "continue"     # 继续学习
    RETRY = "retry"           # 重试
    SKIP = "skip"             # 跳过


@dataclass
class DecisionContext:
    """决策上下文"""
    event: Event
    current_state: str = ""
    risk_level: str = "low"
    
    # 附加信息
    learning_unit_id: Optional[UUID] = None
    artifact_id: Optional[UUID] = None
    
    # 历史信息
    recent_decisions: List["DecisionResult"] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionResult:
    """决策结果"""
    id: UUID = field(default_factory=uuid4)
    
    # 决策
    decision: Decision = Decision.ALLOW
    confidence: float = 1.0
    
    # 原因
    reason: str = ""
    matched_rule: Optional[str] = None
    
    # 建议动作
    recommended_actions: List[str] = field(default_factory=list)
    action_params: Dict[str, Any] = field(default_factory=dict)
    
    # 约束
    constraints: List[str] = field(default_factory=list)
    
    # 审计
    requires_approval: bool = False
    approvers_required: int = 0
    
    # 时间
    decided_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "decision": self.decision.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "matched_rule": self.matched_rule,
            "recommended_actions": self.recommended_actions,
            "action_params": self.action_params,
            "requires_approval": self.requires_approval,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass
class DecisionRule:
    """决策规则"""
    id: str
    name: str
    description: str = ""
    
    # 条件
    event_types: List[EventType] = field(default_factory=list)
    condition: Optional[Callable[[DecisionContext], bool]] = None
    
    # 决策
    decision: Decision = Decision.ALLOW
    confidence: float = 1.0
    
    # 动作
    actions: List[str] = field(default_factory=list)
    action_params_fn: Optional[Callable[[DecisionContext], Dict]] = None
    
    # 审批要求
    requires_approval: bool = False
    approvers_required: int = 0
    
    # 优先级（越高越先匹配）
    priority: int = 0
    
    # 启用状态
    enabled: bool = True
    
    def matches(self, context: DecisionContext) -> bool:
        """检查规则是否匹配"""
        if not self.enabled:
            return False
        
        # 检查事件类型
        if self.event_types and context.event.event_type not in self.event_types:
            return False
        
        # 检查条件
        if self.condition:
            try:
                return self.condition(context)
            except Exception as e:
                logger.error(f"Rule {self.id} condition evaluation failed: {e}")
                return False
        
        return True


class DecisionEngine:
    """
    决策引擎
    
    根据事件和上下文，应用规则做出决策
    """
    
    def __init__(self):
        self._rules: List[DecisionRule] = []
        self._decision_history: List[DecisionResult] = []
        self._max_history_size: int = 1000
        
        # 默认规则
        self._init_default_rules()
    
    def _init_default_rules(self):
        """初始化默认规则"""
        # 严重异常 -> 停机
        self.add_rule(DecisionRule(
            id="critical_anomaly_halt",
            name="Critical Anomaly Halt",
            description="严重异常触发停机",
            event_types=[EventType.ANOMALY_DETECTED],
            condition=lambda ctx: ctx.metadata.get("severity") == "critical",
            decision=Decision.HALT,
            confidence=1.0,
            actions=["freeze_all", "trigger_safe_halt", "notify_governance"],
            requires_approval=False,  # 紧急情况不需要审批
            priority=100,
        ))
        
        # 高风险异常 -> 回滚
        self.add_rule(DecisionRule(
            id="high_risk_rollback",
            name="High Risk Rollback",
            description="高风险异常触发回滚",
            event_types=[EventType.ANOMALY_DETECTED],
            condition=lambda ctx: ctx.metadata.get("severity") == "high",
            decision=Decision.ROLLBACK,
            confidence=0.9,
            actions=["freeze_learning", "restore_checkpoint", "notify_operators"],
            requires_approval=True,
            approvers_required=1,
            priority=90,
        ))
        
        # LU 创建 -> 分类
        self.add_rule(DecisionRule(
            id="lu_auto_classify",
            name="LU Auto Classify",
            description="自动分类新创建的 LU",
            event_types=[EventType.LU_CREATED],
            decision=Decision.CONTINUE,
            actions=["classify_lu", "assess_risk"],
            priority=50,
        ))
        
        # 高风险 LU -> 人工审核
        self.add_rule(DecisionRule(
            id="high_risk_lu_escalate",
            name="High Risk LU Escalate",
            description="高风险 LU 需要人工审核",
            event_types=[EventType.LU_CLASSIFIED],
            condition=lambda ctx: ctx.risk_level in ["high", "critical"],
            decision=Decision.ESCALATE,
            actions=["create_approval_request", "notify_approvers"],
            requires_approval=True,
            approvers_required=2,
            priority=70,
        ))
        
        # 状态转换失败 -> 诊断
        self.add_rule(DecisionRule(
            id="transition_failed_diagnose",
            name="Transition Failed Diagnose",
            description="状态转换失败进入诊断",
            event_types=[EventType.STATE_TRANSITION_FAILED],
            decision=Decision.DIAGNOSE,
            actions=["start_diagnosis", "collect_logs"],
            priority=60,
        ))
        
        # 审批超时 -> 升级
        self.add_rule(DecisionRule(
            id="approval_timeout_escalate",
            name="Approval Timeout Escalate",
            description="审批超时升级处理",
            event_types=[EventType.APPROVAL_TIMEOUT],
            decision=Decision.ESCALATE,
            actions=["notify_governance", "escalate_approval"],
            priority=55,
        ))
        
        # 默认规则 -> 允许
        self.add_rule(DecisionRule(
            id="default_allow",
            name="Default Allow",
            description="默认允许",
            decision=Decision.ALLOW,
            actions=[],
            priority=0,
        ))
    
    def add_rule(self, rule: DecisionRule):
        """添加规则"""
        self._rules.append(rule)
        # 按优先级排序
        self._rules.sort(key=lambda r: r.priority, reverse=True)
    
    def remove_rule(self, rule_id: str):
        """移除规则"""
        self._rules = [r for r in self._rules if r.id != rule_id]
    
    def get_rule(self, rule_id: str) -> Optional[DecisionRule]:
        """获取规则"""
        for rule in self._rules:
            if rule.id == rule_id:
                return rule
        return None
    
    def decide(self, context: DecisionContext) -> DecisionResult:
        """
        根据上下文做出决策
        
        Args:
            context: 决策上下文
            
        Returns:
            决策结果
        """
        matched_rule: Optional[DecisionRule] = None
        
        # 匹配规则
        for rule in self._rules:
            if rule.matches(context):
                matched_rule = rule
                break
        
        if not matched_rule:
            # 使用默认允许
            result = DecisionResult(
                decision=Decision.ALLOW,
                reason="No matching rule, default to allow",
            )
        else:
            # 计算动作参数
            action_params = {}
            if matched_rule.action_params_fn:
                try:
                    action_params = matched_rule.action_params_fn(context)
                except Exception as e:
                    logger.error(f"Action params calculation failed: {e}")
            
            result = DecisionResult(
                decision=matched_rule.decision,
                confidence=matched_rule.confidence,
                reason=matched_rule.description,
                matched_rule=matched_rule.id,
                recommended_actions=matched_rule.actions,
                action_params=action_params,
                requires_approval=matched_rule.requires_approval,
                approvers_required=matched_rule.approvers_required,
            )
        
        # 记录决策
        self._record_decision(result)
        
        logger.info(
            f"Decision made: {result.decision.value} "
            f"(rule={result.matched_rule}, confidence={result.confidence})"
        )
        
        return result
    
    def _record_decision(self, result: DecisionResult):
        """记录决策历史"""
        self._decision_history.append(result)
        if len(self._decision_history) > self._max_history_size:
            self._decision_history = self._decision_history[-self._max_history_size // 2:]
    
    def get_decision_history(self, limit: int = 100) -> List[DecisionResult]:
        """获取决策历史"""
        return sorted(
            self._decision_history,
            key=lambda d: d.decided_at,
            reverse=True
        )[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        decision_counts: Dict[str, int] = {}
        for result in self._decision_history:
            decision_counts[result.decision.value] = \
                decision_counts.get(result.decision.value, 0) + 1
        
        rule_counts: Dict[str, int] = {}
        for result in self._decision_history:
            if result.matched_rule:
                rule_counts[result.matched_rule] = \
                    rule_counts.get(result.matched_rule, 0) + 1
        
        return {
            "total_decisions": len(self._decision_history),
            "decision_distribution": decision_counts,
            "rule_usage": rule_counts,
            "rules_count": len(self._rules),
        }
    
    def list_rules(self) -> List[Dict[str, Any]]:
        """列出所有规则"""
        return [
            {
                "id": r.id,
                "name": r.name,
                "description": r.description,
                "event_types": [e.value for e in r.event_types],
                "decision": r.decision.value,
                "priority": r.priority,
                "enabled": r.enabled,
            }
            for r in self._rules
        ]

