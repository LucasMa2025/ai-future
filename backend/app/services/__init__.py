"""
服务层模块

NLGSM 治理系统服务层，包含：
- 基础服务：用户、角色、权限、认证
- 通知服务：WebSocket、Email
- 状态机服务：NLGSM 状态管理
- 审计服务：业务审计日志
- Learning Unit 服务：LU 生命周期管理
- 审批服务：LU 审批工作流
- 治理服务：学习干预和检查点管理
- 知识转移服务：AGA 知识内化
"""

# 基础服务
from .user_service import UserService
from .auth_service import AuthService
from .role_service import RoleService
from .permission_service import PermissionService

# 通知服务
from .notification_service import NotificationService
from .email_service import EmailService
from .websocket_service import WebSocketManager, websocket_manager

# 状态机和审计
from .state_machine_service import StateMachineService, Event, TransitionResult
from .audit_service import AuditService

# Learning Unit 相关服务
from .learning_unit_service import LearningUnitService
from .approval_service import ApprovalService
from .governance_service import (
    GovernanceService,
    InterventionType,
    InterventionPriority,
)
from .knowledge_transfer_service import (
    KnowledgeTransferService,
    AGALifecycleState,
    BridgeConstraint,
    BridgeLearningUnit,
    BridgeAuditApproval,
    AGABridgeProtocol,
)

__all__ = [
    # 基础服务
    "UserService",
    "AuthService",
    "RoleService",
    "PermissionService",
    
    # 通知服务
    "NotificationService",
    "EmailService",
    "WebSocketManager",
    "websocket_manager",
    
    # 状态机和审计
    "StateMachineService",
    "Event",
    "TransitionResult",
    "AuditService",
    
    # Learning Unit 相关服务
    "LearningUnitService",
    "ApprovalService",
    "GovernanceService",
    "InterventionType",
    "InterventionPriority",
    "KnowledgeTransferService",
    "AGALifecycleState",
    "BridgeConstraint",
    "BridgeLearningUnit",
    "BridgeAuditApproval",
    "AGABridgeProtocol",
]

