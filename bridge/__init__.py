# Bridge Module
"""
Learning Unit 桥接模块 v4.0

⚠️ 重要说明 (v4.0)
==================
1. 本模块已废弃，保留仅为兼容性
2. AIFuture (治理系统) 与 AGA 是独立部署的服务
3. 知识转移通过 HTTP API 完成，详见:
   - 治理系统: AIFuture/backend/app/services/knowledge_transfer_service.py
   - Portal API: AGA/docs/Portal_API_Reference.md

架构变更:
=========
旧架构 (v3.x):
    AIFuture ─── (import) ───> bridge/ ─── (local call) ───> AGA 内部模块
    
新架构 (v4.0):
    ┌────────────────────────────┐         ┌────────────────────────────┐
    │  AIFuture (治理系统)        │         │  AGA Portal (知识管理)     │
    │  - 可能部署在服务器 A       │  HTTP   │  - 可能部署在服务器 B       │
    │  - knowledge_transfer_     │◄───────►│  - 提供 REST API           │
    │    service.py 负责转移      │   API   │  - 无状态，易于扩展         │
    └────────────────────────────┘         └────────────────────────────┘

废弃模块说明:
============
以下模块已废弃，仅保留文件供参考，不再维护:

- bridge_factory.py      : 工厂模式已不适用于分布式架构
- aga_bridge.py          : AGA Bridge 概念已废弃，改用 HTTP API
- aga_core.py            : 本地 AGA 核心集成已移除
- portal_client.py       : 已迁移到 aga.client，本地副本废弃
- production_bridge.py   : 传统 Bridge 模式不支持分布式
- knowledge_adapter.py   : 内化方案仅 AGA 有效，其他方案已废弃
- deep_internalization_service.py : 深度内化服务已废弃
- enhanced_internalization.py     : 增强内化已废弃
- internalization.py     : 内化引擎已废弃

保留模块:
=========
- permission.py          : 权限验证（仍可用于本地权限检查）

推荐使用:
=========
治理系统应使用以下方式与 AGA 通信:

1. 配置 AGA Portal 地址和认证信息:
   - 修改 AIFuture/backend/app/config.py
   - 设置 AGA_PORTAL_URL, AGA_PORTAL_API_KEY 等

2. 使用 KnowledgeTransferService:
   ```python
   from app.services.knowledge_transfer_service import KnowledgeTransferService
   
   service = KnowledgeTransferService(db_session)
   result = service.transfer_to_aga(lu_id)
   ```

3. 或直接使用 httpx 调用 Portal API:
   ```python
   import httpx
   
   client = httpx.Client(base_url="http://aga-portal:8081")
   response = client.post("/knowledge/inject", json={...})
   ```

4. AGA 独立安装到生产服务器时，可复制 aga/client/portal_client.py
   作为独立的客户端库使用。
"""

import warnings

# ==================== 发出废弃警告 ====================
warnings.warn(
    "AIFuture.bridge 模块已废弃 (v4.0)。"
    "知识转移功能已迁移到 app.services.knowledge_transfer_service。"
    "请参阅模块文档了解新架构。",
    DeprecationWarning,
    stacklevel=2,
)


# ==================== 保留模块 ====================
# 权限验证器仍可用
from .permission import PermissionValidator

# ==================== 废弃模块的兼容性导入 ====================
# 这些导入会发出警告，仅为向后兼容保留

def _deprecated_import(name: str):
    """辅助函数：导入废弃模块时发出警告"""
    warnings.warn(
        f"'{name}' 已废弃。请使用 app.services.knowledge_transfer_service 或直接调用 AGA Portal API。",
        DeprecationWarning,
        stacklevel=3,
    )

# AGA Bridge 相关（废弃，仅保留类型定义）
try:
    from .aga_bridge import (
        AGABridge,
        AGABridgeConfig,
        AGAWriteRecord,
        KnowledgeEncoder,
        BridgeWriteResult,
        LifecycleState,
        LearningUnitProtocol,
        AuditApprovalProtocol,
        ConstraintProtocol,
    )
except ImportError:
    AGABridge = None
    AGABridgeConfig = None
    AGAWriteRecord = None
    KnowledgeEncoder = None
    BridgeWriteResult = None
    LifecycleState = None
    LearningUnitProtocol = None
    AuditApprovalProtocol = None
    ConstraintProtocol = None

# 传统桥接组件（废弃）
try:
    from .production_bridge import ProductionBridge
    from .internalization import InternalizationEngine
except ImportError:
    ProductionBridge = None
    InternalizationEngine = None

# 工厂（废弃）
try:
    from .bridge_factory import BridgeFactory, BridgeType, BridgeConfig, create_bridge
except ImportError:
    BridgeFactory = None
    BridgeType = None
    BridgeConfig = None
    create_bridge = None


__all__ = [
    # 保留的功能
    'PermissionValidator',
    
    # 废弃但仍导出的类型（仅用于兼容性）
    'AGABridge',
    'AGABridgeConfig',
    'AGAWriteRecord',
    'KnowledgeEncoder',
    'BridgeWriteResult',
    'LifecycleState',
    'LearningUnitProtocol',
    'AuditApprovalProtocol',
    'ConstraintProtocol',
    
    # 工厂（废弃）
    'BridgeFactory',
    'BridgeType',
    'BridgeConfig',
    'create_bridge',
    
    # 传统组件（废弃）
    'ProductionBridge',
    'InternalizationEngine',
]

# 模块元数据
__version__ = "4.0.0"
__status__ = "deprecated"
__successor__ = "app.services.knowledge_transfer_service"
