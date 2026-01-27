"""
EDA 动作层

负责执行决策产生的动作
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from uuid import UUID, uuid4
import logging
import asyncio

from .event_layer import Event, EventType, EventBus, get_event_bus
from .decision_layer import DecisionResult


logger = logging.getLogger(__name__)


class ActionStatus(str, Enum):
    """动作状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class Action:
    """动作"""
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    
    # 参数
    params: Dict[str, Any] = field(default_factory=dict)
    
    # 状态
    status: ActionStatus = ActionStatus.PENDING
    
    # 执行信息
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # 结果
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # 重试
    retry_count: int = 0
    max_retries: int = 3
    
    # 关联
    decision_id: Optional[UUID] = None
    source_event_id: Optional[UUID] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "status": self.status.value,
            "params": self.params,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class ActionResult:
    """动作执行结果"""
    action: Action
    success: bool = False
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # 产生的事件
    emitted_events: List[Event] = field(default_factory=list)


class ActionRegistry:
    """
    动作注册表
    
    管理所有可用的动作处理器
    """
    
    def __init__(self):
        self._handlers: Dict[str, Callable] = {}
        self._async_handlers: Dict[str, Callable] = {}
        
        # 注册默认处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认处理器"""
        self.register("freeze_learning", self._action_freeze_learning)
        self.register("unfreeze_learning", self._action_unfreeze_learning)
        self.register("freeze_all", self._action_freeze_all)
        self.register("restore_checkpoint", self._action_restore_checkpoint)
        self.register("create_checkpoint", self._action_create_checkpoint)
        self.register("trigger_safe_halt", self._action_trigger_safe_halt)
        self.register("start_diagnosis", self._action_start_diagnosis)
        self.register("collect_logs", self._action_collect_logs)
        self.register("classify_lu", self._action_classify_lu)
        self.register("assess_risk", self._action_assess_risk)
        self.register("create_approval_request", self._action_create_approval_request)
        self.register("notify_approvers", self._action_notify_approvers)
        self.register("notify_operators", self._action_notify_operators)
        self.register("notify_governance", self._action_notify_governance)
        self.register("escalate_approval", self._action_escalate_approval)
    
    def register(self, name: str, handler: Callable):
        """注册同步处理器"""
        self._handlers[name] = handler
    
    def register_async(self, name: str, handler: Callable):
        """注册异步处理器"""
        self._async_handlers[name] = handler
    
    def get_handler(self, name: str) -> Optional[Callable]:
        """获取处理器"""
        return self._handlers.get(name) or self._async_handlers.get(name)
    
    def is_async(self, name: str) -> bool:
        """检查是否为异步处理器"""
        return name in self._async_handlers
    
    def list_actions(self) -> List[str]:
        """列出所有可用动作"""
        return list(set(list(self._handlers.keys()) + list(self._async_handlers.keys())))
    
    # ==================== 默认动作处理器 ====================
    
    def _action_freeze_learning(self, params: Dict) -> Dict[str, Any]:
        """冻结学习"""
        logger.info("Action: Freezing learning")
        return {"action": "freeze_learning", "status": "completed"}
    
    def _action_unfreeze_learning(self, params: Dict) -> Dict[str, Any]:
        """解冻学习"""
        logger.info("Action: Unfreezing learning")
        return {"action": "unfreeze_learning", "status": "completed"}
    
    def _action_freeze_all(self, params: Dict) -> Dict[str, Any]:
        """冻结所有"""
        logger.info("Action: Freezing all activities")
        return {"action": "freeze_all", "status": "completed"}
    
    def _action_restore_checkpoint(self, params: Dict) -> Dict[str, Any]:
        """恢复检查点"""
        checkpoint_id = params.get("checkpoint_id")
        logger.info(f"Action: Restoring checkpoint {checkpoint_id}")
        return {"action": "restore_checkpoint", "checkpoint_id": checkpoint_id, "status": "completed"}
    
    def _action_create_checkpoint(self, params: Dict) -> Dict[str, Any]:
        """创建检查点"""
        logger.info("Action: Creating checkpoint")
        return {"action": "create_checkpoint", "status": "completed"}
    
    def _action_trigger_safe_halt(self, params: Dict) -> Dict[str, Any]:
        """触发安全停机"""
        logger.warning("Action: Triggering safe halt!")
        return {"action": "trigger_safe_halt", "status": "completed"}
    
    def _action_start_diagnosis(self, params: Dict) -> Dict[str, Any]:
        """开始诊断"""
        logger.info("Action: Starting diagnosis")
        return {"action": "start_diagnosis", "status": "completed"}
    
    def _action_collect_logs(self, params: Dict) -> Dict[str, Any]:
        """收集日志"""
        logger.info("Action: Collecting logs")
        return {"action": "collect_logs", "status": "completed"}
    
    def _action_classify_lu(self, params: Dict) -> Dict[str, Any]:
        """分类 LU"""
        lu_id = params.get("lu_id")
        logger.info(f"Action: Classifying LU {lu_id}")
        return {"action": "classify_lu", "lu_id": lu_id, "status": "completed"}
    
    def _action_assess_risk(self, params: Dict) -> Dict[str, Any]:
        """评估风险"""
        logger.info("Action: Assessing risk")
        return {"action": "assess_risk", "status": "completed"}
    
    def _action_create_approval_request(self, params: Dict) -> Dict[str, Any]:
        """创建审批请求"""
        logger.info("Action: Creating approval request")
        return {"action": "create_approval_request", "status": "completed"}
    
    def _action_notify_approvers(self, params: Dict) -> Dict[str, Any]:
        """通知审批人"""
        logger.info("Action: Notifying approvers")
        return {"action": "notify_approvers", "status": "completed"}
    
    def _action_notify_operators(self, params: Dict) -> Dict[str, Any]:
        """通知运维"""
        logger.info("Action: Notifying operators")
        return {"action": "notify_operators", "status": "completed"}
    
    def _action_notify_governance(self, params: Dict) -> Dict[str, Any]:
        """通知治理委员会"""
        logger.info("Action: Notifying governance committee")
        return {"action": "notify_governance", "status": "completed"}
    
    def _action_escalate_approval(self, params: Dict) -> Dict[str, Any]:
        """升级审批"""
        logger.info("Action: Escalating approval")
        return {"action": "escalate_approval", "status": "completed"}


class ActionExecutor:
    """
    动作执行器
    
    负责执行决策产生的动作
    """
    
    def __init__(
        self,
        registry: Optional[ActionRegistry] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self.registry = registry or ActionRegistry()
        self.event_bus = event_bus or get_event_bus()
        
        self._execution_history: List[Action] = []
        self._max_history_size: int = 1000
    
    def execute(self, decision_result: DecisionResult) -> List[ActionResult]:
        """
        执行决策产生的动作
        
        Args:
            decision_result: 决策结果
            
        Returns:
            动作执行结果列表
        """
        results = []
        
        for action_name in decision_result.recommended_actions:
            # 创建动作
            action = Action(
                name=action_name,
                params=decision_result.action_params,
                decision_id=decision_result.id,
            )
            
            # 执行动作
            result = self._execute_action(action)
            results.append(result)
            
            # 如果动作失败且不允许继续，停止执行
            if not result.success and not self._should_continue_on_failure(action_name):
                logger.warning(f"Action {action_name} failed, stopping execution")
                break
        
        return results
    
    def execute_action(self, action: Action) -> ActionResult:
        """执行单个动作"""
        return self._execute_action(action)
    
    async def execute_async(self, decision_result: DecisionResult) -> List[ActionResult]:
        """异步执行动作"""
        results = []
        
        tasks = []
        for action_name in decision_result.recommended_actions:
            action = Action(
                name=action_name,
                params=decision_result.action_params,
                decision_id=decision_result.id,
            )
            tasks.append(self._execute_action_async(action))
        
        action_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in action_results:
            if isinstance(result, Exception):
                results.append(ActionResult(
                    action=Action(name="error"),
                    success=False,
                    error=str(result),
                ))
            else:
                results.append(result)
        
        return results
    
    def _execute_action(self, action: Action) -> ActionResult:
        """执行动作（内部方法）"""
        action.status = ActionStatus.EXECUTING
        action.started_at = datetime.utcnow()
        
        handler = self.registry.get_handler(action.name)
        
        if not handler:
            action.status = ActionStatus.SKIPPED
            action.error = f"No handler found for action: {action.name}"
            logger.warning(action.error)
            
            return ActionResult(
                action=action,
                success=False,
                error=action.error,
            )
        
        try:
            result_data = handler(action.params)
            
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.utcnow()
            action.duration_ms = int(
                (action.completed_at - action.started_at).total_seconds() * 1000
            )
            action.result = result_data
            
            # 记录历史
            self._record_execution(action)
            
            # 发布完成事件
            self._emit_action_completed_event(action)
            
            return ActionResult(
                action=action,
                success=True,
                output=result_data,
            )
            
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.completed_at = datetime.utcnow()
            action.error = str(e)
            
            logger.error(f"Action {action.name} failed: {e}")
            
            # 记录历史
            self._record_execution(action)
            
            return ActionResult(
                action=action,
                success=False,
                error=str(e),
            )
    
    async def _execute_action_async(self, action: Action) -> ActionResult:
        """异步执行动作"""
        action.status = ActionStatus.EXECUTING
        action.started_at = datetime.utcnow()
        
        handler = self.registry.get_handler(action.name)
        
        if not handler:
            action.status = ActionStatus.SKIPPED
            action.error = f"No handler found for action: {action.name}"
            return ActionResult(action=action, success=False, error=action.error)
        
        try:
            if self.registry.is_async(action.name):
                result_data = await handler(action.params)
            else:
                result_data = handler(action.params)
            
            action.status = ActionStatus.COMPLETED
            action.completed_at = datetime.utcnow()
            action.duration_ms = int(
                (action.completed_at - action.started_at).total_seconds() * 1000
            )
            action.result = result_data
            
            return ActionResult(action=action, success=True, output=result_data)
            
        except Exception as e:
            action.status = ActionStatus.FAILED
            action.completed_at = datetime.utcnow()
            action.error = str(e)
            return ActionResult(action=action, success=False, error=str(e))
    
    def _should_continue_on_failure(self, action_name: str) -> bool:
        """检查失败后是否应该继续"""
        # 通知类动作失败可以继续
        notification_actions = ["notify_approvers", "notify_operators", "notify_governance"]
        return action_name in notification_actions
    
    def _record_execution(self, action: Action):
        """记录执行历史"""
        self._execution_history.append(action)
        if len(self._execution_history) > self._max_history_size:
            self._execution_history = self._execution_history[-self._max_history_size // 2:]
    
    def _emit_action_completed_event(self, action: Action):
        """发布动作完成事件"""
        # 可以选择性地发布事件
        pass
    
    def get_execution_history(self, limit: int = 100) -> List[Action]:
        """获取执行历史"""
        return sorted(
            self._execution_history,
            key=lambda a: a.started_at or datetime.min,
            reverse=True
        )[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts: Dict[str, int] = {}
        action_counts: Dict[str, int] = {}
        total_duration = 0
        completed_count = 0
        
        for action in self._execution_history:
            status_counts[action.status.value] = \
                status_counts.get(action.status.value, 0) + 1
            action_counts[action.name] = \
                action_counts.get(action.name, 0) + 1
            
            if action.duration_ms:
                total_duration += action.duration_ms
                completed_count += 1
        
        return {
            "total_executions": len(self._execution_history),
            "status_distribution": status_counts,
            "action_distribution": action_counts,
            "avg_duration_ms": total_duration / completed_count if completed_count > 0 else 0,
        }

