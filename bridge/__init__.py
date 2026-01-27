# Bridge Module
"""
Learning Unit 桥接模块

将审计通过的 Learning Unit 写入生产系统，实现知识内化。

推荐方案（默认）：
┌─────────────────────────────────────────────────────────────┐
│  AGA Bridge - 热插拔式知识系统（远程 API 版本）             │
│  ├─ 零训练注入：知识直接写入 buffer                         │
│  ├─ 即时生效：毫秒级内化                                    │
│  ├─ 即时隔离：问题知识可立即移除                            │
│  ├─ 完全可追溯：每个贡献可追溯到具体 LU                     │
│  ├─ 无灾难性遗忘：不修改原始模型参数                        │
│  └─ 远程 API：支持分布式部署，GPU/CPU 分离                  │
└─────────────────────────────────────────────────────────────┘

架构说明：
┌─────────────────────────────────────────────────────────────┐
│  Backend Server (CPU)                                       │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ AGABridge                                              │ │
│  │  - write_learning_unit()                               │ │
│  │  - confirm/deprecate/quarantine_learning_unit()        │ │
│  │  - 通过 HTTP 调用远程 AGA API                         │ │
│  └─────────────────────────┬─────────────────────────────┘ │
└────────────────────────────┼────────────────────────────────┘
                             │ HTTP REST API
                             v
┌─────────────────────────────────────────────────────────────┐
│  AGA Server (GPU) - python -m aga.api --port 8081           │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ FastAPI (aga/api.py)                                   │ │
│  │  - POST /inject, /inject/batch                         │ │
│  │  - POST /lifecycle/update, /quarantine/*               │ │
│  │  - GET /slot/*, /statistics                            │ │
│  └─────────────────────────┬─────────────────────────────┘ │
│                            v                                │
│  ┌───────────────────────────────────────────────────────┐ │
│  │ AGA Module (aga/core.py v2.1)                          │ │
│  │  - AuxiliaryGovernedAttention                          │ │
│  │  - 挂载到 Transformer 模型                             │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

传统方案（保留）：
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: DeepInternalizationService (深度内化服务)         │
│  Layer 2: Knowledge Adapter (深层内化)                      │
│  Layer 1: Decision Head (表层内化)                          │
│  Layer 0: Permission + Bridge (权限和写入控制)              │
└─────────────────────────────────────────────────────────────┘

使用方式：
```python
from bridge import create_bridge

# 默认使用 AGA（推荐）
bridge = create_bridge(
    aga_api_url="http://gpu-server:8081",
    encoder=KnowledgeEncoder(model, tokenizer),  # 可选
)

# 或指定传统方案
bridge = create_bridge(model, tokenizer, bridge_type="traditional")
```
"""

# ==================== AGA Bridge（推荐，远程 API 模式） ====================
from .aga_bridge import (
    AGABridge,
    AGABridgeConfig,
    AGAWriteRecord,
    AGAAPIClient,
    KnowledgeEncoder,
    BridgeWriteResult,
    LifecycleState,
    LearningUnitProtocol,
    AuditApprovalProtocol,
    ConstraintProtocol,
)

# AGA 核心模块（用于本地 AGA 服务启动）
try:
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from aga.core import (
        AuxiliaryGovernedAttention,
        AGAAugmentedTransformerLayer,
        AGAManager,
        KnowledgeSlotInfo,
        AGADiagnostics,
        AGAConfig,
    )
    from aga.api import create_aga_api, AGAService, AGAClient
    AGA_VERSION = "v2.1"
    HAS_AGA_CORE = True
except ImportError:
    AuxiliaryGovernedAttention = None
    AGAAugmentedTransformerLayer = None
    AGAManager = None
    KnowledgeSlotInfo = None
    AGADiagnostics = None
    AGAConfig = None
    create_aga_api = None
    AGAService = None
    AGAClient = None
    AGA_VERSION = "remote-only"
    HAS_AGA_CORE = False

from .bridge_factory import (
    BridgeFactory,
    BridgeType,
    BridgeConfig,
    create_bridge,
)

# ==================== 传统方案（保留） ====================
from .permission import PermissionValidator
from .internalization import InternalizationEngine
from .production_bridge import ProductionBridge

# 增强版 - Decision Head 改进
from .enhanced_internalization import (
    EnhancedInternalizationEngine,
    EnhancedDecisionHead,
    InternalizationConfig,
    EWCRegularizer,
)

# 深层内化 - Knowledge Adapter
from .knowledge_adapter import (
    KnowledgeAdapter,
    KnowledgeAdapterLayer,
    KnowledgeAdapterManager,
    KnowledgeSlot,
    create_knowledge_encoder,
)

# 深度内化服务 - 完整方案
from .deep_internalization_service import (
    DeepInternalizationService,
    InternalizationResult,
)

__all__ = [
    # ========== AGA Bridge（推荐，远程 API 模式） ==========
    # Bridge（主要接口）
    'AGABridge',
    'AGABridgeConfig',
    'AGAWriteRecord',
    'AGAAPIClient',  # HTTP 客户端
    'KnowledgeEncoder',
    'BridgeWriteResult',
    'LifecycleState',
    
    # 协议接口（用于类型检查）
    'LearningUnitProtocol',
    'AuditApprovalProtocol',
    'ConstraintProtocol',
    
    # AGA 核心（用于本地 AGA 服务启动）
    'AuxiliaryGovernedAttention',
    'AGAAugmentedTransformerLayer',
    'AGAManager',
    'KnowledgeSlotInfo',
    'AGADiagnostics',
    'AGAConfig',
    'create_aga_api',  # FastAPI 应用工厂
    'AGAService',      # AGA 服务单例
    'AGAClient',       # HTTP 客户端（来自 aga.api）
    'AGA_VERSION',
    'HAS_AGA_CORE',
    
    # 工厂
    'BridgeFactory',
    'BridgeType',
    'BridgeConfig',
    'create_bridge',  # 推荐入口
    
    # ========== 传统方案（保留） ==========
    # 基础组件
    'PermissionValidator',
    'InternalizationEngine',
    'ProductionBridge',
    
    # Decision Head 增强
    'EnhancedInternalizationEngine',
    'EnhancedDecisionHead',
    'InternalizationConfig',
    'EWCRegularizer',
    
    # Knowledge Adapter（深层内化）
    'KnowledgeAdapter',
    'KnowledgeAdapterLayer',
    'KnowledgeAdapterManager',
    'KnowledgeSlot',
    'create_knowledge_encoder',
    
    # 深度内化服务
    'DeepInternalizationService',
    'InternalizationResult',
]

