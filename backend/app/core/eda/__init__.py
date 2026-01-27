"""
Event-Decision-Action (EDA) 架构

实现 NLGSM 论文中的三层 EDA 架构：
1. Event Layer: 事件感知层
2. Decision Layer: 决策层
3. Action Layer: 动作执行层
"""

from .event_layer import EventBus, Event, EventType, EventPriority, EventHandler
from .decision_layer import DecisionEngine, DecisionRule, Decision, DecisionContext
from .action_layer import ActionExecutor, Action, ActionResult, ActionRegistry

__all__ = [
    # Event Layer
    "EventBus",
    "Event",
    "EventType",
    "EventPriority",
    "EventHandler",
    # Decision Layer
    "DecisionEngine",
    "DecisionRule",
    "Decision",
    "DecisionContext",
    # Action Layer
    "ActionExecutor",
    "Action",
    "ActionResult",
    "ActionRegistry",
]

