"""
多线程并发学习系统

实现生产级的并发学习能力。

核心组件：
1. Learner - 单个学习线程
2. LearnerPool - 学习线程池
3. LearningTask - 学习任务
4. LearningCoordinator - 学习协调器

设计原则：
- 可配置的线程数量
- 任务队列管理
- 优雅的启停控制
- 状态监控和统计
- 与治理系统的集成
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from threading import Thread, Lock, Event
from queue import Queue, Empty, PriorityQueue
from concurrent.futures import ThreadPoolExecutor, Future
import uuid
import logging
import time
import traceback

from .learning_unit_state import (
    LUStateManager,
    SharedLearningUnit,
    LUStatus,
    LUDecision,
    LUStateChange,
    SelfLearningStateHandler,
)
from .nl_core import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    NestedLearningKernel,
    LLMBasedNLKernel,
)

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 3
    NORMAL = 2
    HIGH = 1
    CRITICAL = 0


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class LearnerStatus(Enum):
    """学习器状态"""
    IDLE = "idle"
    LEARNING = "learning"
    WAITING = "waiting"
    STOPPED = "stopped"


@dataclass
class LearningTask:
    """
    学习任务
    
    封装一个学习请求
    """
    task_id: str
    goal: str
    domain: Optional[str] = None
    priority: TaskPriority = TaskPriority.NORMAL
    
    # 链式学习参数
    parent_lu_id: Optional[str] = None
    exploration_direction: Optional[str] = None
    focus_areas: List[str] = field(default_factory=list)
    
    # 学习范围（由治理系统提供）
    scope: Optional[LearningScope] = None
    
    # 状态
    status: TaskStatus = TaskStatus.PENDING
    
    # 结果
    result_lu_id: Optional[str] = None
    error: Optional[str] = None
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 分配的学习器
    assigned_learner_id: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        return self.priority.value < other.priority.value
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "domain": self.domain,
            "priority": self.priority.value,
            "parent_lu_id": self.parent_lu_id,
            "exploration_direction": self.exploration_direction,
            "focus_areas": self.focus_areas,
            "scope": self.scope.to_dict() if self.scope else None,
            "status": self.status.value,
            "result_lu_id": self.result_lu_id,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "assigned_learner_id": self.assigned_learner_id,
            "metadata": self.metadata,
        }


@dataclass
class LearnerStats:
    """学习器统计"""
    learner_id: str
    status: LearnerStatus
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_learning_time: float = 0.0
    current_task_id: Optional[str] = None
    last_active_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "learner_id": self.learner_id,
            "status": self.status.value,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "total_learning_time": self.total_learning_time,
            "current_task_id": self.current_task_id,
            "last_active_at": self.last_active_at.isoformat() if self.last_active_at else None,
        }


class Learner:
    """
    单个学习器
    
    在独立线程中执行学习任务
    """
    
    def __init__(
        self,
        learner_id: str,
        state_manager: LUStateManager,
        nl_kernel: Optional[NestedLearningKernel] = None,
        llm_adapter = None,
    ):
        self.learner_id = learner_id
        self.state_manager = state_manager
        
        # 每个学习器有自己的 NL 内核实例
        self.nl_kernel = nl_kernel or LLMBasedNLKernel(llm_adapter=llm_adapter)
        if not self.nl_kernel._initialized:
            self.nl_kernel.initialize({})
        
        # 状态处理器
        self.state_handler = SelfLearningStateHandler(
            state_manager=state_manager,
            learner_id=learner_id,
        )
        
        # 状态
        self.status = LearnerStatus.IDLE
        self.current_task: Optional[LearningTask] = None
        
        # 统计
        self.stats = LearnerStats(
            learner_id=learner_id,
            status=LearnerStatus.IDLE,
        )
        
        # 控制
        self._stop_event = Event()
        self._pause_event = Event()
        self._pause_event.set()  # 初始不暂停
    
    def execute_task(self, task: LearningTask) -> SharedLearningUnit:
        """
        执行学习任务
        
        Args:
            task: 学习任务
            
        Returns:
            生成的 Learning Unit
        """
        self.current_task = task
        self.status = LearnerStatus.LEARNING
        self.stats.status = LearnerStatus.LEARNING
        self.stats.current_task_id = task.task_id
        self.stats.last_active_at = datetime.now()
        
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.assigned_learner_id = self.learner_id
        
        start_time = time.time()
        
        try:
            logger.info(f"[Learner {self.learner_id}] Starting task: {task.task_id}")
            
            # 检查是否需要暂停
            self._pause_event.wait()
            
            # 检查是否需要停止
            if self._stop_event.is_set():
                raise InterruptedError("Learner stopped")
            
            # 执行学习
            if task.parent_lu_id:
                # 链式学习
                lu = self._execute_chain_learning(task)
            else:
                # 从零开始
                lu = self._execute_new_learning(task)
            
            # 注册 LU 到状态管理器
            self.state_manager.register_lu(lu)
            self.state_handler.set_current_lu(lu.id)
            
            # 更新任务状态
            task.status = TaskStatus.WAITING_APPROVAL
            task.result_lu_id = lu.id
            
            # 等待治理决策
            self.status = LearnerStatus.WAITING
            self.stats.status = LearnerStatus.WAITING
            
            logger.info(f"[Learner {self.learner_id}] Task {task.task_id} waiting for approval")
            
            # 更新统计
            elapsed = time.time() - start_time
            self.stats.tasks_completed += 1
            self.stats.total_learning_time += elapsed
            
            return lu
            
        except Exception as e:
            logger.error(f"[Learner {self.learner_id}] Task {task.task_id} failed: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.stats.tasks_failed += 1
            raise
        
        finally:
            task.completed_at = datetime.now()
            self.current_task = None
            self.status = LearnerStatus.IDLE
            self.stats.status = LearnerStatus.IDLE
            self.stats.current_task_id = None
    
    def _execute_new_learning(self, task: LearningTask) -> SharedLearningUnit:
        """执行新学习"""
        # 解冻内核
        self.nl_kernel.unfreeze()
        
        try:
            # 设置 Scope
            scope = task.scope or self._create_default_scope()
            
            # 执行学习步骤
            segments = []
            context = {
                "goal": task.goal,
                "domain": task.domain or "general",
                "learning_mode": "from_scratch",
            }
            
            for step in range(3):  # 默认 3 步
                if self._stop_event.is_set():
                    break
                
                self._pause_event.wait()
                
                step_context = {
                    **context,
                    "step": step,
                }
                
                try:
                    segment = self.nl_kernel.execute_learning_step(
                        context=step_context,
                        scope=scope,
                    )
                    segments.append(segment)
                except Exception as e:
                    logger.warning(f"Learning step {step} failed: {e}")
            
            # 创建 Learning Unit
            lu_id = f"lu_{uuid.uuid4().hex[:12]}"
            
            lu = SharedLearningUnit(
                id=lu_id,
                title=f"新学习: {task.goal[:50]}",
                learning_goal=task.goal,
                knowledge={
                    "domain": task.domain or "general",
                    "type": "knowledge",
                    "content": {"goal": task.goal},
                    "confidence": 0.5,
                },
                constraints=[],
                chain_depth=0,
                chain_root_id=lu_id,
                metadata={
                    "task_id": task.task_id,
                    "learner_id": self.learner_id,
                    "segments_count": len(segments),
                },
            )
            
            return lu
            
        finally:
            self.nl_kernel.freeze()
    
    def _execute_chain_learning(self, task: LearningTask) -> SharedLearningUnit:
        """执行链式学习"""
        # 获取父 LU
        parent_lu = self.state_manager.get_lu(task.parent_lu_id)
        if not parent_lu:
            raise ValueError(f"Parent LU not found: {task.parent_lu_id}")
        
        if not parent_lu.can_continue_learning():
            raise ValueError(f"Parent LU cannot be continued: {task.parent_lu_id}")
        
        # 解冻内核
        self.nl_kernel.unfreeze()
        
        try:
            # 设置 Scope
            scope = task.scope or self._create_default_scope()
            
            # 构建继续学习上下文
            parent_context = parent_lu.get_context_for_continuation()
            
            context = {
                "goal": task.goal,
                "domain": parent_lu.knowledge.get("domain", "general"),
                "learning_mode": "continuation",
                **parent_context,
                "exploration_direction": task.exploration_direction,
                "focus_areas": task.focus_areas,
            }
            
            # 执行学习步骤
            segments = []
            for step in range(3):
                if self._stop_event.is_set():
                    break
                
                self._pause_event.wait()
                
                step_context = {
                    **context,
                    "step": step,
                }
                
                try:
                    segment = self.nl_kernel.execute_learning_step(
                        context=step_context,
                        scope=scope,
                    )
                    segments.append(segment)
                except Exception as e:
                    logger.warning(f"Learning step {step} failed: {e}")
            
            # 创建 Learning Unit
            lu_id = f"lu_{uuid.uuid4().hex[:12]}"
            
            lu = SharedLearningUnit(
                id=lu_id,
                title=f"继续学习: {task.goal[:50]}",
                learning_goal=task.goal,
                knowledge={
                    "domain": parent_lu.knowledge.get("domain", "general"),
                    "type": "knowledge",
                    "content": {
                        "goal": task.goal,
                        "inherited_from": parent_lu.id,
                    },
                    "confidence": 0.6,
                },
                constraints=parent_lu.constraints.copy(),
                parent_lu_id=parent_lu.id,
                chain_depth=parent_lu.chain_depth + 1,
                chain_root_id=parent_lu.chain_root_id or parent_lu.id,
                metadata={
                    "task_id": task.task_id,
                    "learner_id": self.learner_id,
                    "segments_count": len(segments),
                    "exploration_direction": task.exploration_direction,
                },
            )
            
            return lu
            
        finally:
            self.nl_kernel.freeze()
    
    def _create_default_scope(self) -> LearningScope:
        """创建默认 Scope"""
        return LearningScope(
            scope_id=f"scope_{uuid.uuid4().hex[:8]}",
            max_level=NLLevel.MEMORY,
            allowed_levels=[NLLevel.PARAMETER, NLLevel.MEMORY],
            created_by=f"learner_{self.learner_id}",
        )
    
    def pause(self) -> None:
        """暂停学习器"""
        self._pause_event.clear()
        self.status = LearnerStatus.WAITING
        self.stats.status = LearnerStatus.WAITING
        logger.info(f"[Learner {self.learner_id}] Paused")
    
    def resume(self) -> None:
        """恢复学习器"""
        self._pause_event.set()
        logger.info(f"[Learner {self.learner_id}] Resumed")
    
    def stop(self) -> None:
        """停止学习器"""
        self._stop_event.set()
        self._pause_event.set()  # 确保不会卡在暂停
        self.status = LearnerStatus.STOPPED
        self.stats.status = LearnerStatus.STOPPED
        self.state_handler.cleanup()
        logger.info(f"[Learner {self.learner_id}] Stopped")
    
    def get_stats(self) -> LearnerStats:
        """获取统计信息"""
        return self.stats


class LearnerPool:
    """
    学习器线程池
    
    管理多个学习器，支持并发学习。
    
    使用示例：
        # 创建线程池
        pool = LearnerPool(
            num_learners=4,
            state_manager=state_manager,
        )
        
        # 启动
        pool.start()
        
        # 提交任务
        task = pool.submit_task(
            goal="学习金融风险管理",
            domain="financial",
            priority=TaskPriority.HIGH,
        )
        
        # 等待结果
        result = pool.wait_for_task(task.task_id)
        
        # 停止
        pool.shutdown()
    """
    
    def __init__(
        self,
        num_learners: int = 4,
        state_manager: Optional[LUStateManager] = None,
        llm_adapter_factory: Optional[Callable[[], Any]] = None,
        max_queue_size: int = 100,
    ):
        """
        初始化学习器池
        
        Args:
            num_learners: 学习器数量
            state_manager: 状态管理器
            llm_adapter_factory: LLM 适配器工厂函数
            max_queue_size: 最大队列大小
        """
        self.num_learners = num_learners
        self.state_manager = state_manager or LUStateManager()
        self.llm_adapter_factory = llm_adapter_factory
        
        # 任务队列（优先级队列）
        self.task_queue: PriorityQueue = PriorityQueue(maxsize=max_queue_size)
        
        # 任务追踪
        self.tasks: Dict[str, LearningTask] = {}
        self.task_futures: Dict[str, Future] = {}
        self._tasks_lock = Lock()
        
        # 学习器
        self.learners: Dict[str, Learner] = {}
        self._learners_lock = Lock()
        
        # 线程池
        self._executor: Optional[ThreadPoolExecutor] = None
        
        # 控制
        self._running = False
        self._dispatcher_thread: Optional[Thread] = None
    
    def start(self) -> None:
        """启动线程池"""
        if self._running:
            return
        
        logger.info(f"Starting LearnerPool with {self.num_learners} learners")
        
        # 创建学习器
        for i in range(self.num_learners):
            learner_id = f"learner_{i}"
            llm_adapter = self.llm_adapter_factory() if self.llm_adapter_factory else None
            
            learner = Learner(
                learner_id=learner_id,
                state_manager=self.state_manager,
                llm_adapter=llm_adapter,
            )
            self.learners[learner_id] = learner
        
        # 创建线程池
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_learners,
            thread_name_prefix="learner_",
        )
        
        # 启动调度线程
        self._running = True
        self._dispatcher_thread = Thread(
            target=self._dispatcher_loop,
            name="learner_dispatcher",
            daemon=True,
        )
        self._dispatcher_thread.start()
        
        logger.info("LearnerPool started")
    
    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池"""
        logger.info("Shutting down LearnerPool")
        
        self._running = False
        
        # 停止所有学习器
        for learner in self.learners.values():
            learner.stop()
        
        # 关闭线程池
        if self._executor:
            self._executor.shutdown(wait=wait)
        
        # 等待调度线程
        if self._dispatcher_thread and self._dispatcher_thread.is_alive():
            self._dispatcher_thread.join(timeout=5)
        
        logger.info("LearnerPool shutdown complete")
    
    def submit_task(
        self,
        goal: str,
        domain: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        parent_lu_id: Optional[str] = None,
        exploration_direction: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        scope: Optional[LearningScope] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LearningTask:
        """
        提交学习任务
        
        Args:
            goal: 学习目标
            domain: 领域
            priority: 优先级
            parent_lu_id: 父 LU ID（链式学习）
            exploration_direction: 探索方向
            focus_areas: 重点关注领域
            scope: 学习范围
            metadata: 元数据
            
        Returns:
            学习任务
        """
        task = LearningTask(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            goal=goal,
            domain=domain,
            priority=priority,
            parent_lu_id=parent_lu_id,
            exploration_direction=exploration_direction,
            focus_areas=focus_areas or [],
            scope=scope,
            metadata=metadata or {},
        )
        
        with self._tasks_lock:
            self.tasks[task.task_id] = task
        
        self.task_queue.put(task)
        
        logger.info(f"Task submitted: {task.task_id} - {goal[:50]}")
        
        return task
    
    def _dispatcher_loop(self) -> None:
        """调度循环"""
        while self._running:
            try:
                # 从队列获取任务
                try:
                    task = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # 找到空闲的学习器
                learner = self._get_idle_learner()
                if not learner:
                    # 没有空闲学习器，放回队列
                    self.task_queue.put(task)
                    time.sleep(0.5)
                    continue
                
                # 提交任务到线程池
                future = self._executor.submit(
                    self._execute_task_wrapper,
                    learner,
                    task,
                )
                
                with self._tasks_lock:
                    self.task_futures[task.task_id] = future
                
            except Exception as e:
                logger.error(f"Dispatcher error: {e}")
                traceback.print_exc()
    
    def _execute_task_wrapper(
        self,
        learner: Learner,
        task: LearningTask,
    ) -> SharedLearningUnit:
        """任务执行包装器"""
        try:
            return learner.execute_task(task)
        except Exception as e:
            logger.error(f"Task {task.task_id} execution failed: {e}")
            raise
    
    def _get_idle_learner(self) -> Optional[Learner]:
        """获取空闲的学习器"""
        with self._learners_lock:
            for learner in self.learners.values():
                if learner.status == LearnerStatus.IDLE:
                    return learner
        return None
    
    def get_task(self, task_id: str) -> Optional[LearningTask]:
        """获取任务"""
        with self._tasks_lock:
            return self.tasks.get(task_id)
    
    def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[SharedLearningUnit]:
        """
        等待任务完成
        
        Args:
            task_id: 任务 ID
            timeout: 超时时间
            
        Returns:
            生成的 Learning Unit
        """
        with self._tasks_lock:
            future = self.task_futures.get(task_id)
        
        if not future:
            return None
        
        try:
            return future.result(timeout=timeout)
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._tasks_lock:
            task = self.tasks.get(task_id)
            future = self.task_futures.get(task_id)
        
        if not task:
            return False
        
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            return True
        
        if future and not future.done():
            return future.cancel()
        
        return False
    
    def pause_all(self) -> None:
        """暂停所有学习器"""
        for learner in self.learners.values():
            learner.pause()
    
    def resume_all(self) -> None:
        """恢复所有学习器"""
        for learner in self.learners.values():
            learner.resume()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._tasks_lock:
            task_stats = {
                "total": len(self.tasks),
                "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
                "running": len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
                "completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
                "failed": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
                "waiting_approval": len([t for t in self.tasks.values() if t.status == TaskStatus.WAITING_APPROVAL]),
            }
        
        learner_stats = {
            learner_id: learner.get_stats().to_dict()
            for learner_id, learner in self.learners.items()
        }
        
        return {
            "num_learners": self.num_learners,
            "running": self._running,
            "queue_size": self.task_queue.qsize(),
            "tasks": task_stats,
            "learners": learner_stats,
            "state_manager": self.state_manager.get_statistics(),
        }
    
    def get_learner_stats(self, learner_id: str) -> Optional[LearnerStats]:
        """获取指定学习器的统计"""
        learner = self.learners.get(learner_id)
        if learner:
            return learner.get_stats()
        return None


class LearningCoordinator:
    """
    学习协调器
    
    协调学习器池和治理系统的交互。
    
    功能：
    1. 处理治理决策
    2. 自动触发后续学习
    3. 管理学习链
    """
    
    def __init__(
        self,
        learner_pool: LearnerPool,
        state_manager: LUStateManager,
        max_chain_depth: int = 5,
        auto_continue: bool = True,
    ):
        self.learner_pool = learner_pool
        self.state_manager = state_manager
        self.max_chain_depth = max_chain_depth
        self.auto_continue = auto_continue
        
        # 订阅状态变更
        self.state_manager.subscribe(
            "coordinator",
            self._on_state_change,
        )
    
    def _on_state_change(self, change: LUStateChange) -> None:
        """处理状态变更"""
        if not self.auto_continue:
            return
        
        # 根据决策自动触发后续学习
        if change.decision == LUDecision.CONTINUE:
            self._handle_continue(change)
        elif change.decision == LUDecision.NEW_LEARNING:
            self._handle_new_learning(change)
        elif change.decision == LUDecision.ADJUST:
            self._handle_adjust(change)
    
    def _handle_continue(self, change: LUStateChange) -> None:
        """处理继续学习"""
        lu = self.state_manager.get_lu(change.lu_id)
        if not lu or not lu.can_continue_learning():
            return
        
        if lu.chain_depth >= self.max_chain_depth:
            logger.info(f"Chain depth limit reached for LU {change.lu_id}")
            return
        
        # 从继续参数中获取新目标
        new_goal = change.continue_params.get("new_goal")
        if not new_goal:
            logger.warning(f"No new goal provided for continue learning")
            return
        
        # 提交继续学习任务
        self.learner_pool.submit_task(
            goal=new_goal,
            domain=lu.knowledge.get("domain"),
            parent_lu_id=lu.id,
            exploration_direction=change.continue_params.get("exploration_direction"),
            focus_areas=change.continue_params.get("focus_areas", []),
            priority=TaskPriority.NORMAL,
        )
        
        logger.info(f"Continue learning task submitted for LU {change.lu_id}")
    
    def _handle_new_learning(self, change: LUStateChange) -> None:
        """处理新学习"""
        new_goal = change.continue_params.get("new_goal")
        if not new_goal:
            return
        
        self.learner_pool.submit_task(
            goal=new_goal,
            domain=change.continue_params.get("domain"),
            priority=TaskPriority.NORMAL,
        )
        
        logger.info(f"New learning task submitted")
    
    def _handle_adjust(self, change: LUStateChange) -> None:
        """处理调整"""
        # 调整后继续学习
        lu = self.state_manager.get_lu(change.lu_id)
        if not lu:
            return
        
        # 使用调整后的参数继续
        adjusted_goal = change.adjustment_params.get("adjusted_goal", lu.learning_goal)
        
        self.learner_pool.submit_task(
            goal=adjusted_goal,
            domain=lu.knowledge.get("domain"),
            parent_lu_id=lu.id,
            exploration_direction=change.adjustment_params.get("exploration_direction"),
            focus_areas=change.adjustment_params.get("focus_areas", []),
            scope=LearningScope.from_dict(change.adjustment_params["scope"]) 
                  if "scope" in change.adjustment_params else None,
            priority=TaskPriority.HIGH,
        )
        
        logger.info(f"Adjusted learning task submitted for LU {change.lu_id}")
    
    def cleanup(self) -> None:
        """清理资源"""
        self.state_manager.unsubscribe("coordinator")

