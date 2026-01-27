"""
非阻塞异步学习模型

解决 Learner 等待 LU 状态通知的核心问题。

核心设计原则：
1. 提交即忘记（Fire-and-Forget）：Learner 提交 LU 后立即可以处理下一个任务
2. 事件驱动调度：状态变更通过异步事件触发后续操作
3. 待处理 LU 追踪：跟踪所有等待审批的 LU
4. 超时和重试机制：处理长时间未响应的 LU
5. 优先级队列：根据紧急程度调度任务

工作流程：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Learner   │────>│  提交 LU    │────>│ 立即返回    │
│  (执行学习)  │     │ (Fire&Forget)│     │ (不等待)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ PendingLU   │
                    │   Tracker   │
                    └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  NLGSM 治理  │     │  人工审批   │     │  超时处理   │
│  (自动分类)  │     │ (可能很长)  │     │ (可配置)    │
└─────────────┘     └─────────────┘     └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    ┌─────────────┐
                    │ 状态变更事件 │
                    │  (异步通知)  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Coordinator │
                    │ (事件处理)  │
                    └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 继续学习    │     │ 新学习      │     │ 调整/停止   │
│ (提交新任务) │     │ (提交新任务) │     │ (更新状态)  │
└─────────────┘     └─────────────┘     └─────────────┘
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from threading import Thread, Lock, Event
from queue import Queue, Empty, PriorityQueue
import uuid
import logging
import time
import traceback

logger = logging.getLogger(__name__)


class PendingLUStatus(Enum):
    """待处理 LU 状态"""
    SUBMITTED = "submitted"           # 已提交，等待自动分类
    AUTO_CLASSIFIED = "auto_classified"  # 已自动分类，等待人工审批
    HUMAN_REVIEW = "human_review"     # 人工审批中
    TIMEOUT = "timeout"               # 超时
    RESOLVED = "resolved"             # 已解决（收到决策）


@dataclass
class PendingLU:
    """
    待处理的 Learning Unit
    
    追踪已提交但尚未收到决策的 LU
    """
    lu_id: str
    task_id: str
    learner_id: str
    
    # 状态
    status: PendingLUStatus = PendingLUStatus.SUBMITTED
    
    # 时间追踪
    submitted_at: datetime = field(default_factory=datetime.now)
    last_status_update: datetime = field(default_factory=datetime.now)
    
    # 超时配置
    auto_classify_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    human_review_timeout: timedelta = field(default_factory=lambda: timedelta(hours=24))
    
    # 重试
    retry_count: int = 0
    max_retries: int = 3
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_timeout(self) -> bool:
        """检查是否超时"""
        now = datetime.now()
        elapsed = now - self.last_status_update
        
        if self.status == PendingLUStatus.SUBMITTED:
            return elapsed > self.auto_classify_timeout
        elif self.status in [PendingLUStatus.AUTO_CLASSIFIED, PendingLUStatus.HUMAN_REVIEW]:
            return elapsed > self.human_review_timeout
        
        return False
    
    def get_wait_time(self) -> timedelta:
        """获取等待时间"""
        return datetime.now() - self.submitted_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lu_id": self.lu_id,
            "task_id": self.task_id,
            "learner_id": self.learner_id,
            "status": self.status.value,
            "submitted_at": self.submitted_at.isoformat(),
            "last_status_update": self.last_status_update.isoformat(),
            "wait_time_seconds": self.get_wait_time().total_seconds(),
            "is_timeout": self.is_timeout(),
            "retry_count": self.retry_count,
        }


class PendingLUTracker:
    """
    待处理 LU 追踪器
    
    核心功能：
    1. 追踪所有待处理的 LU
    2. 检测超时
    3. 提供查询接口
    """
    
    def __init__(
        self,
        auto_classify_timeout: timedelta = timedelta(minutes=5),
        human_review_timeout: timedelta = timedelta(hours=24),
    ):
        self._pending: Dict[str, PendingLU] = {}
        self._lock = Lock()
        
        # 默认超时配置
        self.auto_classify_timeout = auto_classify_timeout
        self.human_review_timeout = human_review_timeout
        
        # 统计
        self._total_submitted = 0
        self._total_resolved = 0
        self._total_timeout = 0
    
    def track(
        self,
        lu_id: str,
        task_id: str,
        learner_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PendingLU:
        """开始追踪 LU"""
        pending = PendingLU(
            lu_id=lu_id,
            task_id=task_id,
            learner_id=learner_id,
            auto_classify_timeout=self.auto_classify_timeout,
            human_review_timeout=self.human_review_timeout,
            metadata=metadata or {},
        )
        
        with self._lock:
            self._pending[lu_id] = pending
            self._total_submitted += 1
        
        logger.info(f"[PendingTracker] Started tracking LU: {lu_id}")
        return pending
    
    def update_status(
        self,
        lu_id: str,
        new_status: PendingLUStatus,
    ) -> Optional[PendingLU]:
        """更新 LU 状态"""
        with self._lock:
            pending = self._pending.get(lu_id)
            if pending:
                pending.status = new_status
                pending.last_status_update = datetime.now()
                
                if new_status == PendingLUStatus.RESOLVED:
                    self._total_resolved += 1
                elif new_status == PendingLUStatus.TIMEOUT:
                    self._total_timeout += 1
        
        return pending
    
    def resolve(self, lu_id: str) -> Optional[PendingLU]:
        """标记 LU 为已解决"""
        return self.update_status(lu_id, PendingLUStatus.RESOLVED)
    
    def remove(self, lu_id: str) -> Optional[PendingLU]:
        """移除追踪"""
        with self._lock:
            return self._pending.pop(lu_id, None)
    
    def get(self, lu_id: str) -> Optional[PendingLU]:
        """获取待处理 LU"""
        with self._lock:
            return self._pending.get(lu_id)
    
    def get_all_pending(self) -> List[PendingLU]:
        """获取所有待处理 LU"""
        with self._lock:
            return list(self._pending.values())
    
    def get_timeout_lus(self) -> List[PendingLU]:
        """获取超时的 LU"""
        with self._lock:
            return [p for p in self._pending.values() if p.is_timeout()]
    
    def get_by_learner(self, learner_id: str) -> List[PendingLU]:
        """获取指定 Learner 的待处理 LU"""
        with self._lock:
            return [p for p in self._pending.values() if p.learner_id == learner_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            by_status = {}
            total_wait_time = timedelta()
            
            for pending in self._pending.values():
                status = pending.status.value
                by_status[status] = by_status.get(status, 0) + 1
                total_wait_time += pending.get_wait_time()
            
            avg_wait_time = (
                total_wait_time / len(self._pending)
                if self._pending else timedelta()
            )
            
            return {
                "total_pending": len(self._pending),
                "total_submitted": self._total_submitted,
                "total_resolved": self._total_resolved,
                "total_timeout": self._total_timeout,
                "by_status": by_status,
                "avg_wait_time_seconds": avg_wait_time.total_seconds(),
                "timeout_count": len([p for p in self._pending.values() if p.is_timeout()]),
            }


class AsyncLearningCoordinator:
    """
    异步学习协调器
    
    核心职责：
    1. 处理状态变更事件
    2. 根据决策自动触发后续操作
    3. 管理待处理 LU
    4. 处理超时
    
    这是解决"Learner 等待状态通知"问题的核心组件。
    """
    
    def __init__(
        self,
        pending_tracker: PendingLUTracker,
        task_submitter: Callable[[Dict[str, Any]], str],  # 提交新任务的回调
        max_chain_depth: int = 5,
        auto_continue: bool = True,
        timeout_check_interval: float = 60.0,  # 超时检查间隔（秒）
    ):
        """
        初始化异步学习协调器
        
        Args:
            pending_tracker: 待处理 LU 追踪器
            task_submitter: 提交新任务的回调函数
            max_chain_depth: 最大链深度
            auto_continue: 是否自动继续学习
            timeout_check_interval: 超时检查间隔
        """
        self.pending_tracker = pending_tracker
        self.task_submitter = task_submitter
        self.max_chain_depth = max_chain_depth
        self.auto_continue = auto_continue
        self.timeout_check_interval = timeout_check_interval
        
        # 事件队列
        self._event_queue: Queue = Queue()
        
        # 控制
        self._running = False
        self._event_processor_thread: Optional[Thread] = None
        self._timeout_checker_thread: Optional[Thread] = None
        
        # 回调
        self._on_timeout_callbacks: List[Callable[[PendingLU], None]] = []
        self._on_decision_callbacks: List[Callable[[str, str, Dict], None]] = []
        
        # 统计
        self._events_processed = 0
        self._tasks_auto_submitted = 0
    
    def start(self) -> None:
        """启动协调器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动事件处理线程
        self._event_processor_thread = Thread(
            target=self._event_processor_loop,
            name="async_coordinator_events",
            daemon=True,
        )
        self._event_processor_thread.start()
        
        # 启动超时检查线程
        self._timeout_checker_thread = Thread(
            target=self._timeout_checker_loop,
            name="async_coordinator_timeout",
            daemon=True,
        )
        self._timeout_checker_thread.start()
        
        logger.info("[AsyncCoordinator] Started")
    
    def stop(self) -> None:
        """停止协调器"""
        self._running = False
        
        # 等待线程结束
        if self._event_processor_thread and self._event_processor_thread.is_alive():
            self._event_processor_thread.join(timeout=5)
        
        if self._timeout_checker_thread and self._timeout_checker_thread.is_alive():
            self._timeout_checker_thread.join(timeout=5)
        
        logger.info("[AsyncCoordinator] Stopped")
    
    def on_lu_submitted(
        self,
        lu_id: str,
        task_id: str,
        learner_id: str,
        lu_data: Dict[str, Any],
    ) -> None:
        """
        LU 提交事件
        
        当 Learner 提交 LU 时调用。
        Learner 调用此方法后立即返回，不等待。
        """
        # 开始追踪
        self.pending_tracker.track(
            lu_id=lu_id,
            task_id=task_id,
            learner_id=learner_id,
            metadata=lu_data,
        )
        
        logger.info(f"[AsyncCoordinator] LU submitted: {lu_id} (Learner: {learner_id})")
    
    def on_state_change(
        self,
        lu_id: str,
        old_status: str,
        new_status: str,
        decision: Optional[str] = None,
        decision_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        状态变更事件
        
        当 NLGSM 治理系统更新 LU 状态时调用。
        这是异步通知的入口点。
        """
        event = {
            "type": "state_change",
            "lu_id": lu_id,
            "old_status": old_status,
            "new_status": new_status,
            "decision": decision,
            "decision_params": decision_params or {},
            "timestamp": datetime.now().isoformat(),
        }
        
        self._event_queue.put(event)
        
        logger.info(f"[AsyncCoordinator] State change event queued: {lu_id} -> {new_status}")
    
    def _event_processor_loop(self) -> None:
        """事件处理循环"""
        while self._running:
            try:
                # 从队列获取事件
                try:
                    event = self._event_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # 处理事件
                self._process_event(event)
                self._events_processed += 1
                
            except Exception as e:
                logger.error(f"[AsyncCoordinator] Event processing error: {e}")
                traceback.print_exc()
    
    def _process_event(self, event: Dict[str, Any]) -> None:
        """处理单个事件"""
        event_type = event.get("type")
        
        if event_type == "state_change":
            self._handle_state_change(event)
        else:
            logger.warning(f"[AsyncCoordinator] Unknown event type: {event_type}")
    
    def _handle_state_change(self, event: Dict[str, Any]) -> None:
        """处理状态变更事件"""
        lu_id = event["lu_id"]
        new_status = event["new_status"]
        decision = event.get("decision")
        decision_params = event.get("decision_params", {})
        
        # 更新追踪状态
        pending = self.pending_tracker.get(lu_id)
        if pending:
            # 映射状态
            status_map = {
                "auto_classified": PendingLUStatus.AUTO_CLASSIFIED,
                "human_review": PendingLUStatus.HUMAN_REVIEW,
                "approved": PendingLUStatus.RESOLVED,
                "rejected": PendingLUStatus.RESOLVED,
                "corrected": PendingLUStatus.RESOLVED,
                "terminated": PendingLUStatus.RESOLVED,
                "internalized": PendingLUStatus.RESOLVED,
            }
            
            if new_status in status_map:
                self.pending_tracker.update_status(lu_id, status_map[new_status])
        
        # 通知回调
        for callback in self._on_decision_callbacks:
            try:
                callback(lu_id, decision, decision_params)
            except Exception as e:
                logger.error(f"Decision callback error: {e}")
        
        # 根据决策自动触发后续操作
        if self.auto_continue and decision:
            self._handle_decision(lu_id, decision, decision_params, pending)
    
    def _handle_decision(
        self,
        lu_id: str,
        decision: str,
        params: Dict[str, Any],
        pending: Optional[PendingLU],
    ) -> None:
        """根据决策触发后续操作"""
        if decision == "continue":
            self._trigger_continue_learning(lu_id, params, pending)
        elif decision == "new_learning":
            self._trigger_new_learning(params)
        elif decision == "adjust":
            self._trigger_adjusted_learning(lu_id, params, pending)
        elif decision == "stop":
            logger.info(f"[AsyncCoordinator] Learning stopped for LU: {lu_id}")
            # 清理追踪
            if pending:
                self.pending_tracker.remove(lu_id)
    
    def _trigger_continue_learning(
        self,
        parent_lu_id: str,
        params: Dict[str, Any],
        pending: Optional[PendingLU],
    ) -> None:
        """触发继续学习"""
        # 检查链深度
        chain_depth = params.get("chain_depth", 0)
        if chain_depth >= self.max_chain_depth:
            logger.info(f"[AsyncCoordinator] Chain depth limit reached: {chain_depth}")
            return
        
        # 构建新任务
        new_goal = params.get("new_goal")
        if not new_goal:
            logger.warning("[AsyncCoordinator] No new goal for continue learning")
            return
        
        task_params = {
            "goal": new_goal,
            "parent_lu_id": parent_lu_id,
            "domain": params.get("domain"),
            "exploration_direction": params.get("exploration_direction"),
            "focus_areas": params.get("focus_areas", []),
            "priority": "normal",
            "source": "auto_continue",
        }
        
        # 提交新任务
        try:
            task_id = self.task_submitter(task_params)
            self._tasks_auto_submitted += 1
            logger.info(f"[AsyncCoordinator] Continue learning task submitted: {task_id}")
        except Exception as e:
            logger.error(f"[AsyncCoordinator] Failed to submit continue task: {e}")
        
        # 清理已解决的追踪
        if pending:
            self.pending_tracker.remove(parent_lu_id)
    
    def _trigger_new_learning(self, params: Dict[str, Any]) -> None:
        """触发新学习"""
        new_goal = params.get("new_goal")
        if not new_goal:
            return
        
        task_params = {
            "goal": new_goal,
            "domain": params.get("domain"),
            "priority": "normal",
            "source": "auto_new",
        }
        
        try:
            task_id = self.task_submitter(task_params)
            self._tasks_auto_submitted += 1
            logger.info(f"[AsyncCoordinator] New learning task submitted: {task_id}")
        except Exception as e:
            logger.error(f"[AsyncCoordinator] Failed to submit new task: {e}")
    
    def _trigger_adjusted_learning(
        self,
        parent_lu_id: str,
        params: Dict[str, Any],
        pending: Optional[PendingLU],
    ) -> None:
        """触发调整后的学习"""
        adjusted_goal = params.get("adjusted_goal")
        if not adjusted_goal:
            return
        
        task_params = {
            "goal": adjusted_goal,
            "parent_lu_id": parent_lu_id,
            "domain": params.get("domain"),
            "exploration_direction": params.get("exploration_direction"),
            "focus_areas": params.get("focus_areas", []),
            "scope": params.get("scope"),
            "priority": "high",  # 调整后的任务优先级更高
            "source": "auto_adjust",
        }
        
        try:
            task_id = self.task_submitter(task_params)
            self._tasks_auto_submitted += 1
            logger.info(f"[AsyncCoordinator] Adjusted learning task submitted: {task_id}")
        except Exception as e:
            logger.error(f"[AsyncCoordinator] Failed to submit adjusted task: {e}")
        
        if pending:
            self.pending_tracker.remove(parent_lu_id)
    
    def _timeout_checker_loop(self) -> None:
        """超时检查循环"""
        while self._running:
            try:
                time.sleep(self.timeout_check_interval)
                
                if not self._running:
                    break
                
                # 检查超时
                timeout_lus = self.pending_tracker.get_timeout_lus()
                
                for pending in timeout_lus:
                    self._handle_timeout(pending)
                
            except Exception as e:
                logger.error(f"[AsyncCoordinator] Timeout check error: {e}")
    
    def _handle_timeout(self, pending: PendingLU) -> None:
        """处理超时"""
        logger.warning(f"[AsyncCoordinator] LU timeout: {pending.lu_id}")
        
        # 更新状态
        self.pending_tracker.update_status(pending.lu_id, PendingLUStatus.TIMEOUT)
        
        # 通知回调
        for callback in self._on_timeout_callbacks:
            try:
                callback(pending)
            except Exception as e:
                logger.error(f"Timeout callback error: {e}")
        
        # 可选：自动重试或升级
        if pending.retry_count < pending.max_retries:
            pending.retry_count += 1
            pending.status = PendingLUStatus.SUBMITTED
            pending.last_status_update = datetime.now()
            logger.info(f"[AsyncCoordinator] Retrying LU: {pending.lu_id} (attempt {pending.retry_count})")
        else:
            # 超过重试次数，移除追踪
            self.pending_tracker.remove(pending.lu_id)
            logger.warning(f"[AsyncCoordinator] LU abandoned after max retries: {pending.lu_id}")
    
    def register_timeout_callback(
        self,
        callback: Callable[[PendingLU], None],
    ) -> None:
        """注册超时回调"""
        self._on_timeout_callbacks.append(callback)
    
    def register_decision_callback(
        self,
        callback: Callable[[str, str, Dict], None],
    ) -> None:
        """注册决策回调"""
        self._on_decision_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "running": self._running,
            "events_processed": self._events_processed,
            "tasks_auto_submitted": self._tasks_auto_submitted,
            "event_queue_size": self._event_queue.qsize(),
            "pending_tracker": self.pending_tracker.get_statistics(),
        }


class NonBlockingLearner:
    """
    非阻塞学习器
    
    核心特性：
    1. 提交 LU 后立即返回，不等待审批
    2. 通过事件驱动接收决策
    3. 支持并发处理多个任务
    
    工作模式：
    - 从任务队列获取任务
    - 执行学习，生成 LU
    - 提交 LU 到审计系统（Fire-and-Forget）
    - 立即处理下一个任务
    - 通过 Coordinator 的事件驱动处理后续操作
    """
    
    def __init__(
        self,
        learner_id: str,
        coordinator: AsyncLearningCoordinator,
        nl_kernel = None,
        llm_adapter = None,
    ):
        from .nl_core import LLMBasedNLKernel, LearningScope, NLLevel
        
        self.learner_id = learner_id
        self.coordinator = coordinator
        
        # NL 内核
        self.nl_kernel = nl_kernel or LLMBasedNLKernel(llm_adapter=llm_adapter)
        if not self.nl_kernel._initialized:
            self.nl_kernel.initialize({})
        
        # 状态
        self._is_learning = False
        self._current_task_id: Optional[str] = None
        
        # 统计
        self._tasks_completed = 0
        self._lus_submitted = 0
        
        # 控制
        self._stop_event = Event()
    
    def execute_task(self, task: Dict[str, Any]) -> Optional[str]:
        """
        执行学习任务
        
        非阻塞模式：提交 LU 后立即返回
        
        Args:
            task: 任务参数
            
        Returns:
            生成的 LU ID（如果成功）
        """
        from .nl_core import LearningScope, NLLevel
        
        task_id = task.get("task_id", f"task_{uuid.uuid4().hex[:8]}")
        goal = task.get("goal", "")
        parent_lu_id = task.get("parent_lu_id")
        
        self._is_learning = True
        self._current_task_id = task_id
        
        try:
            logger.info(f"[Learner {self.learner_id}] Starting task: {task_id}")
            
            # 检查停止信号
            if self._stop_event.is_set():
                return None
            
            # 解冻内核
            self.nl_kernel.unfreeze()
            
            try:
                # 执行学习
                scope = LearningScope(
                    scope_id=f"scope_{uuid.uuid4().hex[:8]}",
                    max_level=NLLevel.MEMORY,
                    allowed_levels=[NLLevel.PARAMETER, NLLevel.MEMORY],
                    created_by=f"learner_{self.learner_id}",
                )
                
                context = {
                    "goal": goal,
                    "domain": task.get("domain", "general"),
                    "parent_lu_id": parent_lu_id,
                    "exploration_direction": task.get("exploration_direction"),
                    "focus_areas": task.get("focus_areas", []),
                }
                
                # 执行学习步骤
                segments = []
                for step in range(3):
                    if self._stop_event.is_set():
                        break
                    
                    step_context = {**context, "step": step}
                    try:
                        segment = self.nl_kernel.execute_learning_step(
                            context=step_context,
                            scope=scope,
                        )
                        segments.append(segment)
                    except Exception as e:
                        logger.warning(f"Learning step {step} failed: {e}")
                
                # 生成 LU
                lu_id = f"lu_{uuid.uuid4().hex[:12]}"
                
                lu_data = {
                    "id": lu_id,
                    "title": f"学习: {goal[:50]}",
                    "learning_goal": goal,
                    "parent_lu_id": parent_lu_id,
                    "chain_depth": task.get("chain_depth", 0) + (1 if parent_lu_id else 0),
                    "domain": task.get("domain", "general"),
                    "segments_count": len(segments),
                    "learner_id": self.learner_id,
                    "task_id": task_id,
                }
                
                # 提交到协调器（Fire-and-Forget）
                self.coordinator.on_lu_submitted(
                    lu_id=lu_id,
                    task_id=task_id,
                    learner_id=self.learner_id,
                    lu_data=lu_data,
                )
                
                self._lus_submitted += 1
                self._tasks_completed += 1
                
                logger.info(f"[Learner {self.learner_id}] LU submitted: {lu_id} (Fire-and-Forget)")
                
                return lu_id
                
            finally:
                self.nl_kernel.freeze()
        
        except Exception as e:
            logger.error(f"[Learner {self.learner_id}] Task failed: {e}")
            traceback.print_exc()
            return None
        
        finally:
            self._is_learning = False
            self._current_task_id = None
    
    def stop(self) -> None:
        """停止学习器"""
        self._stop_event.set()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            "learner_id": self.learner_id,
            "is_learning": self._is_learning,
            "current_task_id": self._current_task_id,
            "tasks_completed": self._tasks_completed,
            "lus_submitted": self._lus_submitted,
        }


class AsyncLearnerPool:
    """
    异步学习器池
    
    管理多个非阻塞学习器，实现高效的并发学习。
    
    核心特性：
    1. 学习器不等待审批，提交后立即处理下一个任务
    2. 通过 Coordinator 处理状态变更和后续操作
    3. 支持动态调整学习器数量
    4. 优先级任务队列
    """
    
    def __init__(
        self,
        num_learners: int = 4,
        pending_tracker: Optional[PendingLUTracker] = None,
        llm_adapter_factory: Optional[Callable[[], Any]] = None,
        max_queue_size: int = 1000,
        auto_continue: bool = True,
        max_chain_depth: int = 5,
    ):
        """
        初始化异步学习器池
        
        Args:
            num_learners: 学习器数量
            pending_tracker: 待处理 LU 追踪器
            llm_adapter_factory: LLM 适配器工厂
            max_queue_size: 最大队列大小
            auto_continue: 是否自动继续学习
            max_chain_depth: 最大链深度
        """
        self.num_learners = num_learners
        self.llm_adapter_factory = llm_adapter_factory
        self.max_queue_size = max_queue_size
        
        # 待处理追踪器
        self.pending_tracker = pending_tracker or PendingLUTracker()
        
        # 协调器
        self.coordinator = AsyncLearningCoordinator(
            pending_tracker=self.pending_tracker,
            task_submitter=self._submit_task_internal,
            max_chain_depth=max_chain_depth,
            auto_continue=auto_continue,
        )
        
        # 任务队列（优先级队列）
        self._task_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        
        # 任务追踪
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._task_counter = 0
        self._tasks_lock = Lock()
        
        # 学习器
        self._learners: Dict[str, NonBlockingLearner] = {}
        self._learner_threads: Dict[str, Thread] = {}
        
        # 控制
        self._running = False
    
    def start(self) -> None:
        """启动学习器池"""
        if self._running:
            return
        
        logger.info(f"[AsyncPool] Starting with {self.num_learners} learners")
        
        self._running = True
        
        # 启动协调器
        self.coordinator.start()
        
        # 创建并启动学习器
        for i in range(self.num_learners):
            learner_id = f"learner_{i}"
            llm_adapter = self.llm_adapter_factory() if self.llm_adapter_factory else None
            
            learner = NonBlockingLearner(
                learner_id=learner_id,
                coordinator=self.coordinator,
                llm_adapter=llm_adapter,
            )
            self._learners[learner_id] = learner
            
            # 启动工作线程
            thread = Thread(
                target=self._worker_loop,
                args=(learner,),
                name=f"learner_{i}",
                daemon=True,
            )
            self._learner_threads[learner_id] = thread
            thread.start()
        
        logger.info("[AsyncPool] Started")
    
    def shutdown(self, wait: bool = True) -> None:
        """关闭学习器池"""
        logger.info("[AsyncPool] Shutting down")
        
        self._running = False
        
        # 停止所有学习器
        for learner in self._learners.values():
            learner.stop()
        
        # 停止协调器
        self.coordinator.stop()
        
        # 等待线程结束
        if wait:
            for thread in self._learner_threads.values():
                if thread.is_alive():
                    thread.join(timeout=5)
        
        logger.info("[AsyncPool] Shutdown complete")
    
    def submit_task(
        self,
        goal: str,
        domain: Optional[str] = None,
        priority: str = "normal",
        parent_lu_id: Optional[str] = None,
        exploration_direction: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        提交学习任务
        
        Args:
            goal: 学习目标
            domain: 领域
            priority: 优先级 (critical/high/normal/low)
            parent_lu_id: 父 LU ID（链式学习）
            exploration_direction: 探索方向
            focus_areas: 重点关注领域
            metadata: 元数据
            
        Returns:
            任务 ID
        """
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        # 优先级映射
        priority_map = {
            "critical": 0,
            "high": 1,
            "normal": 2,
            "low": 3,
        }
        priority_value = priority_map.get(priority, 2)
        
        task = {
            "task_id": task_id,
            "goal": goal,
            "domain": domain,
            "priority": priority,
            "parent_lu_id": parent_lu_id,
            "exploration_direction": exploration_direction,
            "focus_areas": focus_areas or [],
            "metadata": metadata or {},
            "submitted_at": datetime.now().isoformat(),
            "status": "pending",
        }
        
        with self._tasks_lock:
            self._tasks[task_id] = task
            self._task_counter += 1
        
        # 放入优先级队列
        self._task_queue.put((priority_value, self._task_counter, task))
        
        logger.info(f"[AsyncPool] Task submitted: {task_id} - {goal[:50]}")
        
        return task_id
    
    def _submit_task_internal(self, params: Dict[str, Any]) -> str:
        """内部任务提交（供协调器使用）"""
        return self.submit_task(
            goal=params.get("goal", ""),
            domain=params.get("domain"),
            priority=params.get("priority", "normal"),
            parent_lu_id=params.get("parent_lu_id"),
            exploration_direction=params.get("exploration_direction"),
            focus_areas=params.get("focus_areas"),
            metadata={"source": params.get("source", "auto")},
        )
    
    def _worker_loop(self, learner: NonBlockingLearner) -> None:
        """工作循环"""
        while self._running:
            try:
                # 从队列获取任务
                try:
                    _, _, task = self._task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # 更新任务状态
                task_id = task["task_id"]
                with self._tasks_lock:
                    if task_id in self._tasks:
                        self._tasks[task_id]["status"] = "running"
                        self._tasks[task_id]["learner_id"] = learner.learner_id
                
                # 执行任务
                lu_id = learner.execute_task(task)
                
                # 更新任务状态
                with self._tasks_lock:
                    if task_id in self._tasks:
                        self._tasks[task_id]["status"] = "submitted" if lu_id else "failed"
                        self._tasks[task_id]["result_lu_id"] = lu_id
                        self._tasks[task_id]["completed_at"] = datetime.now().isoformat()
                
            except Exception as e:
                logger.error(f"[AsyncPool] Worker error: {e}")
                traceback.print_exc()
    
    def on_governance_decision(
        self,
        lu_id: str,
        old_status: str,
        new_status: str,
        decision: Optional[str] = None,
        decision_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        接收治理决策
        
        这是 NLGSM 治理系统通知学习系统的入口点。
        
        Args:
            lu_id: Learning Unit ID
            old_status: 旧状态
            new_status: 新状态
            decision: 决策 (continue/new_learning/adjust/stop)
            decision_params: 决策参数
        """
        self.coordinator.on_state_change(
            lu_id=lu_id,
            old_status=old_status,
            new_status=new_status,
            decision=decision,
            decision_params=decision_params,
        )
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
        with self._tasks_lock:
            return self._tasks.get(task_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._tasks_lock:
            task_stats = {
                "total": len(self._tasks),
                "pending": len([t for t in self._tasks.values() if t["status"] == "pending"]),
                "running": len([t for t in self._tasks.values() if t["status"] == "running"]),
                "submitted": len([t for t in self._tasks.values() if t["status"] == "submitted"]),
                "failed": len([t for t in self._tasks.values() if t["status"] == "failed"]),
            }
        
        learner_stats = {
            learner_id: learner.get_status()
            for learner_id, learner in self._learners.items()
        }
        
        return {
            "num_learners": self.num_learners,
            "running": self._running,
            "queue_size": self._task_queue.qsize(),
            "tasks": task_stats,
            "learners": learner_stats,
            "coordinator": self.coordinator.get_statistics(),
        }

