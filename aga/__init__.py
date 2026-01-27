# AGA - Auxiliary Governed Attention
"""
AGA (Auxiliary Governed Attention) v2.1 - 热插拔式知识注入系统

v2.1 优化：
- 注入安全: norm clipping 控制注入向量幅度
- Early Exit: 门值过低时跳过 AGA 计算
- Auto Deprecate: 自动废弃长期未命中的槽位
- 批量注入: inject_knowledge_batch 减少开销
- 增强诊断: router_scores, gate_mean, early_exit_ratio

v2.0 优化：
- Slot Routing: O(N) → O(k) 复杂度优化
- Delta Subspace: value 通过 bottleneck projection 受控干预
- 熵信号解耦: 支持多种不确定性信号源，兼容 FlashAttention
- 元数据外置: DB 管理 lifecycle/LU，AGA 只是执行层

核心特性：
- 零训练注入：知识直接写入 buffer，无需梯度计算
- 热插拔设计：运行时动态添加/移除知识
- 治理控制：生命周期状态、熵门控、可追溯性
- 即时隔离：问题知识可立即移除影响

使用示例:
    from aga import AGA, AGAConfig, LifecycleState
    
    # v2.1: 使用配置创建 AGA 实例
    config = AGAConfig(
        hidden_dim=4096, 
        num_slots=100,
        top_k_routing=8,  # 路由优化
        use_value_projection=True,  # delta subspace
        enable_norm_clipping=True,  # 注入安全
        enable_early_exit=True,  # 性能优化
    )
    aga = AGA(config=config)
    
    # 注入知识
    aga.inject_knowledge(
        slot_idx=0,
        key_vector=key_vec,
        value_vector=val_vec,
        lu_id="LU_001",
    )
    
    # 挂载到模型
    manager = AGAManager()
    manager.attach_to_model(model, layer_indices=[-2, -1])

API 服务:
    from aga.api import create_aga_router
    
    # 创建 FastAPI 路由
    router = create_aga_router(aga_manager)
    app.include_router(router, prefix="/aga")
"""

from .core import (
    AuxiliaryGovernedAttention,
    AGAAugmentedTransformerLayer,
    AGAManager,
    LifecycleState,
    KnowledgeSlotInfo,
    AGADiagnostics,
    AGAStatistics,
    SlotInfo,
    # v2.0 新增
    AGAConfig,
    UncertaintySource,
    UncertaintyEstimator,
    SlotRouter,
)

from .persistence import (
    AGAPersistence,
    SQLitePersistence,
    AGAPersistenceManager,
    KnowledgeRecord,
)

# API (optional import, requires fastapi)
try:
    from .api import create_aga_api, AGAService, AGAClient
    create_aga_router = create_aga_api  # Alias for backward compatibility
except ImportError:
    create_aga_api = None
    create_aga_router = None
    AGAService = None
    AGAClient = None

__version__ = "2.1.0"
__author__ = "Lucas Ma"

__all__ = [
    # Core
    "AuxiliaryGovernedAttention",
    "AGAAugmentedTransformerLayer", 
    "AGAManager",
    "LifecycleState",
    "KnowledgeSlotInfo",
    "AGADiagnostics",
    "AGAStatistics",
    "SlotInfo",
    # v2.0 Core
    "AGAConfig",
    "UncertaintySource",
    "UncertaintyEstimator",
    "SlotRouter",
    # Persistence
    "AGAPersistence",
    "SQLitePersistence",
    "AGAPersistenceManager",
    "KnowledgeRecord",
    # API
    "create_aga_api",
    "create_aga_router",  # alias
    "AGAService",
    "AGAClient",
    # Alias
    "AGA",
]

# Alias for convenience
AGA = AuxiliaryGovernedAttention

