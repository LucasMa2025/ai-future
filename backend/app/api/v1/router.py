"""
API V1 路由注册
"""
from fastapi import APIRouter

from .endpoints import auth, users, roles, state_machine, approvals, audit

api_router = APIRouter()

# 注册各模块路由
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["认证"]
)

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

