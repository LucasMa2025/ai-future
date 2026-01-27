# Self-Learning System Module
"""
自学习系统模块 v3.1

包含三个实现版本：
1. 基础版本：LearningUnitBuilder（原有实现）
2. NL 增强版：NLLearningUnitBuilder（集成 Nested Learning 框架）
3. 链式学习版：ChainableLearningUnitBuilder（支持知识链式学习）

并发学习支持（v3.0 新增）：
- LearnerPool: 多线程学习器池（同步模式）
- Learner: 单个学习线程
- LearningTask: 学习任务
- LearningCoordinator: 学习协调器

异步学习支持（v3.1 新增）：
- AsyncLearnerPool: 异步学习器池（生产推荐）
- NonBlockingLearner: 非阻塞学习器
- AsyncLearningCoordinator: 异步学习协调器
- PendingLUTracker: 待处理 LU 追踪器

核心设计原则（v3.1）：
- 提交即忘记（Fire-and-Forget）：Learner 提交 LU 后立即处理下一个任务
- 事件驱动调度：状态变更通过异步事件触发后续操作
- 不阻塞等待：人工审批可能需要数小时，学习系统不会被阻塞

状态共享机制（v3.0 重构）：
- LUStateManager: Learning Unit 状态管理器
- SharedLearningUnit: 共享的 Learning Unit（带状态）
- SelfLearningStateHandler: 自学习系统状态处理器
- 异步通知机制：治理系统通过状态变更通知自学习系统

LLM 支持：
- 支持多种 LLM 后端（通过适配器注入）
- DeepSeek, Ollama, vLLM, OpenAI 兼容接口
- 学习的起点是 LLM 的现有知识库

知识循环：
- ProductionKnowledgeReader: 查询生产库已有知识
- ChainableLearningUnitBuilder: 支持链式学习

治理接口：
- GovernanceInterface: 治理系统与自学习的接口
- 支持学习过程暂停、修改、终止
- 支持检查点审查和 Learning Unit 审计反馈

使用示例：
    # 1. 单线程学习（简单场景）
    from self_learning import AutonomousExplorer
    explorer = AutonomousExplorer.with_adapter("ollama", model="llama3.2")
    
    # 2. 异步并发学习（生产推荐）
    from self_learning import AsyncLearnerPool
    
    pool = AsyncLearnerPool(
        num_learners=4,  # 4 个学习线程
        auto_continue=True,  # 自动继续学习
    )
    pool.start()
    
    # 提交任务（立即返回，不等待）
    task_id = pool.submit_task(
        goal="学习金融风险管理",
        domain="financial",
    )
    
    # 治理系统通知决策（异步）
    pool.on_governance_decision(
        lu_id="lu_xxx",
        old_status="pending",
        new_status="approved",
        decision="continue",
        decision_params={"new_goal": "深入学习..."},
    )
    
    # 关闭
    pool.shutdown()
    
    # 3. 同步模式（需要等待结果时）
    from self_learning import LearnerPool, LUStateManager
    
    state_manager = LUStateManager()
    pool = LearnerPool(
        num_learners=4,
        state_manager=state_manager,
    )
    pool.start()
    
    task = pool.submit_task(goal="学习金融风险管理")
    result = pool.wait_for_task(task.task_id)
    pool.shutdown()
"""

from .explorer import AutonomousExplorer
from .knowledge_generator import KnowledgeGenerator
from .checkpoint import CheckpointManager
from .learning_unit_builder import LearningUnitBuilder

# Nested Learning 增强版
from .nl_learning_unit_builder import NLLearningUnitBuilder

# 链式学习版
from .chainable_learning_builder import (
    ChainableLearningUnitBuilder,
    ChainableLearningUnit,
    LearningChain,
    LURelation,
    LURelationType,
    ContinueLearningContext,
)

# 状态共享机制（v3.0）
from .learning_unit_state import (
    LUStateManager,
    SharedLearningUnit,
    LUStatus,
    LUDecision,
    LUStateChange,
    SelfLearningStateHandler,
)

# 并发学习系统（v3.0）- 同步模式
from .concurrent_learner import (
    LearnerPool,
    Learner,
    LearningTask,
    LearningCoordinator,
    TaskPriority,
    TaskStatus,
    LearnerStatus,
    LearnerStats,
)

# 异步学习系统（v3.1）- 生产推荐
from .async_learning_model import (
    AsyncLearnerPool,
    NonBlockingLearner,
    AsyncLearningCoordinator,
    PendingLUTracker,
    PendingLU,
    PendingLUStatus,
)

# 知识读取器（保留 ProductionKnowledgeReader，移除 ApprovedLUReader）
from .knowledge_reader import (
    ProductionKnowledgeReader,
    ProductionKnowledge,
    KnowledgeSearchResult,
    KnowledgeType,
    # 内存实现（用于测试）
    InMemoryProductionKnowledgeReader,
    # 数据库实现
    DatabaseProductionKnowledgeReader,
)

# 治理接口
from .governance_interface import (
    GovernanceInterface,
    GovernanceIntervention,
    InterventionType,
    InterventionPriority,
    CheckpointReview,
    LearningUnitAuditFeedback,
)

# NL 核心模块
from .nl_core import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    MemoryLevel,
    ContinuumMemoryState,
    NestedLearningKernel,
    LLMBasedNLKernel,
    KernelFactory,
    ContinuumMemorySystem,
    ExpressiveOptimizer,
)

__all__ = [
    # 基础组件
    'AutonomousExplorer',
    'KnowledgeGenerator',
    'CheckpointManager',
    
    # 构建器
    'LearningUnitBuilder',           # 基础版
    'NLLearningUnitBuilder',         # NL 增强版
    'ChainableLearningUnitBuilder',  # 链式学习版
    
    # 链式学习类型
    'ChainableLearningUnit',
    'LearningChain',
    'LURelation',
    'LURelationType',
    'ContinueLearningContext',
    
    # 状态共享机制（v3.0）
    'LUStateManager',
    'SharedLearningUnit',
    'LUStatus',
    'LUDecision',
    'LUStateChange',
    'SelfLearningStateHandler',
    
    # 并发学习系统（v3.0）- 同步模式
    'LearnerPool',
    'Learner',
    'LearningTask',
    'LearningCoordinator',
    'TaskPriority',
    'TaskStatus',
    'LearnerStatus',
    'LearnerStats',
    
    # 异步学习系统（v3.1）- 生产推荐
    'AsyncLearnerPool',
    'NonBlockingLearner',
    'AsyncLearningCoordinator',
    'PendingLUTracker',
    'PendingLU',
    'PendingLUStatus',
    
    # 知识读取器
    'ProductionKnowledgeReader',
    'ProductionKnowledge',
    'KnowledgeSearchResult',
    'KnowledgeType',
    'InMemoryProductionKnowledgeReader',
    'DatabaseProductionKnowledgeReader',
    
    # 治理接口
    'GovernanceInterface',
    'GovernanceIntervention',
    'InterventionType',
    'InterventionPriority',
    'CheckpointReview',
    'LearningUnitAuditFeedback',
    
    # NL 类型
    'NLLevel',
    'LearningScope',
    'ContextFlowSegment',
    'MemoryLevel',
    'ContinuumMemoryState',
    
    # NL 内核
    'NestedLearningKernel',
    'LLMBasedNLKernel',
    'KernelFactory',
    
    # NL 系统
    'ContinuumMemorySystem',
    'ExpressiveOptimizer',
]
