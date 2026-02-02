"""
API V1 路由注册
"""
from fastapi import APIRouter

from .endpoints import (
    auth, 
    users, 
    roles, 
    state_machine, 
    approvals, 
    audit, 
    learning_control,
    system_functions,
    permissions,
    operation_logs,
    backups,
    system_configs,
)

api_router = APIRouter()

# ==================== 认证模块 ====================
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["认证"]
)

# ==================== 系统管理模块 ====================
api_router.include_router(
    users.router,
    prefix="/users",
    tags=["用户管理"]
)

api_router.include_router(
    roles.router,
    prefix="/roles",
    tags=["角色管理"]
)

api_router.include_router(
    system_functions.router,
    prefix="/system-functions",
    tags=["系统功能"]
)

api_router.include_router(
    permissions.router,
    prefix="/permissions",
    tags=["权限管理"]
)

api_router.include_router(
    operation_logs.router,
    prefix="/operation-logs",
    tags=["操作日志"]
)

api_router.include_router(
    backups.router,
    prefix="/backups",
    tags=["数据备份"]
)

api_router.include_router(
    system_configs.router,
    prefix="/system-configs",
    tags=["系统配置"]
)

# ==================== NLGSM 治理模块 ====================
api_router.include_router(
    state_machine.router,
    prefix="/state-machine",
    tags=["状态机"]
)

api_router.include_router(
    approvals.router,
    prefix="/approvals",
    tags=["审批中心"]
)

api_router.include_router(
    audit.router,
    prefix="/audit",
    tags=["审计日志"]
)

api_router.include_router(
    learning_control.router,
    prefix="/learning",
    tags=["学习控制"]
)

