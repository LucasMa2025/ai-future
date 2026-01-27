"""
数据库初始化脚本
"""
from sqlalchemy.orm import Session
from ..models import Base, Role, Permission, SystemFunction, SystemState
from ..core.enums import NLGSMState, AuditLevel
from .session import engine
import logging

logger = logging.getLogger(__name__)


def init_db(db: Session) -> None:
    """初始化数据库"""
    # 创建所有表
    Base.metadata.create_all(bind=engine)
    
    # 初始化角色
    _init_roles(db)
    
    # 初始化系统功能
    _init_system_functions(db)
    
    # 初始化权限
    _init_permissions(db)
    
    # 初始化系统状态
    _init_system_state(db)
    
    db.commit()
    logger.info("Database initialized successfully")


def _init_roles(db: Session) -> None:
    """初始化预置角色"""
    roles_data = [
        {
            "name": "admin",
            "display_name": "系统管理员",
            "description": "拥有系统全部权限",
            "risk_level_limit": "critical",
            "is_system": True,
            "sort_order": 1,
        },
        {
            "name": "governance_committee",
            "display_name": "治理委员会",
            "description": "负责关键决策和高风险审批",
            "risk_level_limit": "critical",
            "is_system": True,
            "sort_order": 2,
        },
        {
            "name": "senior_engineer",
            "display_name": "高级工程师",
            "description": "负责高风险审批和系统回滚",
            "risk_level_limit": "high",
            "is_system": True,
            "sort_order": 3,
        },
        {
            "name": "ml_engineer",
            "display_name": "ML工程师",
            "description": "负责中低风险审批",
            "risk_level_limit": "medium",
            "is_system": True,
            "sort_order": 4,
        },
        {
            "name": "operator",
            "display_name": "运维人员",
            "description": "负责系统运维和状态控制",
            "risk_level_limit": "low",
            "is_system": True,
            "sort_order": 5,
        },
        {
            "name": "auditor",
            "display_name": "审计员",
            "description": "只读访问审计日志",
            "risk_level_limit": None,
            "is_system": True,
            "sort_order": 6,
        },
        {
            "name": "viewer",
            "display_name": "观察者",
            "description": "只读访问仪表盘",
            "risk_level_limit": None,
            "is_system": True,
            "sort_order": 7,
        },
    ]
    
    for role_data in roles_data:
        existing = db.query(Role).filter(Role.name == role_data["name"]).first()
        if not existing:
            role = Role(**role_data)
            db.add(role)
            logger.info(f"Created role: {role_data['name']}")


def _init_system_functions(db: Session) -> None:
    """初始化系统功能表"""
    functions_data = [
        # 模块级别
        {"code": "dashboard", "name": "仪表盘", "level": 1, "module": "dashboard", "sort_order": 1, "is_audited": False, "audit_level": "none"},
        {"code": "state_machine", "name": "状态管理", "level": 1, "module": "state_machine", "sort_order": 2, "is_audited": True, "audit_level": "important"},
        {"code": "artifacts", "name": "工件管理", "level": 1, "module": "artifacts", "sort_order": 3, "is_audited": True, "audit_level": "important"},
        {"code": "learning_units", "name": "学习单元", "level": 1, "module": "learning_units", "sort_order": 4, "is_audited": True, "audit_level": "normal"},
        {"code": "approvals", "name": "审批中心", "level": 1, "module": "approvals", "sort_order": 5, "is_audited": True, "audit_level": "critical"},
        {"code": "audit_logs", "name": "审计日志", "level": 1, "module": "audit_logs", "sort_order": 6, "is_audited": True, "audit_level": "info"},
        {"code": "anomaly", "name": "异常监控", "level": 1, "module": "anomaly", "sort_order": 7, "is_audited": True, "audit_level": "important"},
        {"code": "users", "name": "用户管理", "level": 1, "module": "users", "sort_order": 8, "is_audited": True, "audit_level": "important"},
        {"code": "roles", "name": "角色管理", "level": 1, "module": "roles", "sort_order": 9, "is_audited": True, "audit_level": "important"},
        {"code": "settings", "name": "系统配置", "level": 1, "module": "settings", "sort_order": 10, "is_audited": True, "audit_level": "critical"},
        
        # API 端点级别 - 状态机
        {"code": "state_machine.get_current", "name": "获取当前状态", "level": 3, "module": "state_machine", "method": "GET", "api_path": "/api/v1/state-machine/current", "sort_order": 1, "is_audited": False},
        {"code": "state_machine.trigger", "name": "触发状态转换", "level": 3, "module": "state_machine", "method": "POST", "api_path": "/api/v1/state-machine/trigger", "sort_order": 2, "is_audited": True, "audit_level": "critical"},
        {"code": "state_machine.history", "name": "状态历史", "level": 3, "module": "state_machine", "method": "GET", "api_path": "/api/v1/state-machine/history", "sort_order": 3, "is_audited": False},
        
        # API 端点级别 - 审批
        {"code": "approvals.list", "name": "审批列表", "level": 3, "module": "approvals", "method": "GET", "api_path": "/api/v1/approvals", "sort_order": 1, "is_audited": False},
        {"code": "approvals.approve", "name": "批准", "level": 3, "module": "approvals", "method": "POST", "api_path": "/api/v1/approvals/{id}/approve", "sort_order": 2, "is_audited": True, "audit_level": "critical"},
        {"code": "approvals.reject", "name": "拒绝", "level": 3, "module": "approvals", "method": "POST", "api_path": "/api/v1/approvals/{id}/reject", "sort_order": 3, "is_audited": True, "audit_level": "critical"},
        
        # API 端点级别 - 用户
        {"code": "users.list", "name": "用户列表", "level": 3, "module": "users", "method": "GET", "api_path": "/api/v1/users", "sort_order": 1, "is_audited": False},
        {"code": "users.create", "name": "创建用户", "level": 3, "module": "users", "method": "POST", "api_path": "/api/v1/users", "sort_order": 2, "is_audited": True, "audit_level": "important"},
        {"code": "users.update", "name": "更新用户", "level": 3, "module": "users", "method": "PUT", "api_path": "/api/v1/users/{id}", "sort_order": 3, "is_audited": True, "audit_level": "important"},
        {"code": "users.delete", "name": "删除用户", "level": 3, "module": "users", "method": "DELETE", "api_path": "/api/v1/users/{id}", "sort_order": 4, "is_audited": True, "audit_level": "critical"},
    ]
    
    for func_data in functions_data:
        existing = db.query(SystemFunction).filter(SystemFunction.code == func_data["code"]).first()
        if not existing:
            func = SystemFunction(**func_data)
            db.add(func)


def _init_permissions(db: Session) -> None:
    """初始化权限"""
    permissions_data = [
        # 仪表盘
        {"code": "dashboard.view", "name": "查看仪表盘", "action": "read"},
        
        # 状态机
        {"code": "state_machine.view", "name": "查看状态", "action": "read"},
        {"code": "state_machine.trigger", "name": "触发状态转换", "action": "update"},
        
        # 工件
        {"code": "artifacts.view", "name": "查看工件", "action": "read"},
        {"code": "artifacts.rollback", "name": "回滚工件", "action": "update"},
        
        # 审批
        {"code": "approvals.view", "name": "查看审批", "action": "read"},
        {"code": "approvals.approve_low", "name": "审批低风险", "action": "approve"},
        {"code": "approvals.approve_medium", "name": "审批中风险", "action": "approve"},
        {"code": "approvals.approve_high", "name": "审批高风险", "action": "approve"},
        {"code": "approvals.approve_critical", "name": "审批关键风险", "action": "approve"},
        
        # 审计
        {"code": "audit.view", "name": "查看审计日志", "action": "read"},
        {"code": "audit.export", "name": "导出审计日志", "action": "export"},
        
        # 用户管理
        {"code": "users.view", "name": "查看用户", "action": "read"},
        {"code": "users.create", "name": "创建用户", "action": "create"},
        {"code": "users.update", "name": "更新用户", "action": "update"},
        {"code": "users.delete", "name": "删除用户", "action": "delete"},
        
        # 角色管理
        {"code": "roles.view", "name": "查看角色", "action": "read"},
        {"code": "roles.manage", "name": "管理角色", "action": "update"},
        
        # 系统配置
        {"code": "settings.view", "name": "查看配置", "action": "read"},
        {"code": "settings.update", "name": "更新配置", "action": "update"},
    ]
    
    for perm_data in permissions_data:
        existing = db.query(Permission).filter(Permission.code == perm_data["code"]).first()
        if not existing:
            perm = Permission(**perm_data)
            db.add(perm)


def _init_system_state(db: Session) -> None:
    """初始化系统状态"""
    existing = db.query(SystemState).filter(SystemState.is_current == True).first()
    if not existing:
        state = SystemState(
            state=NLGSMState.FROZEN.value,
            is_current=True,
            trigger_event="system_init",
            trigger_source="init_db",
            metadata={"initialized": True}
        )
        db.add(state)
        logger.info("Initialized system state to FROZEN")

