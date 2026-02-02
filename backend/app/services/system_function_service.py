"""
系统功能服务

管理系统功能表（模块、功能、子功能三级模式）
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..models.permission import SystemFunction, Permission


class SystemFunctionService:
    """
    系统功能服务
    
    功能层级结构:
    - Level 1: 模块 (如: 系统管理、NLGSM治理)
    - Level 2: 功能组 (如: 用户管理、角色管理)
    - Level 3: 具体功能/API端点 (如: 用户列表、创建用户)
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== 查询操作 ====================
    
    def get_all_functions(
        self, 
        include_disabled: bool = False
    ) -> List[SystemFunction]:
        """获取所有系统功能"""
        query = self.db.query(SystemFunction)
        if not include_disabled:
            query = query.filter(SystemFunction.is_enabled == True)
        return query.order_by(SystemFunction.level, SystemFunction.sort_order).all()
    
    def get_function_by_id(self, function_id: int) -> Optional[SystemFunction]:
        """根据ID获取功能"""
        return self.db.query(SystemFunction).filter(
            SystemFunction.id == function_id
        ).first()
    
    def get_function_by_code(self, code: str) -> Optional[SystemFunction]:
        """根据代码获取功能"""
        return self.db.query(SystemFunction).filter(
            SystemFunction.code == code
        ).first()
    
    def get_functions_by_level(self, level: int) -> List[SystemFunction]:
        """获取指定层级的功能"""
        return self.db.query(SystemFunction).filter(
            and_(
                SystemFunction.level == level,
                SystemFunction.is_enabled == True
            )
        ).order_by(SystemFunction.sort_order).all()
    
    def get_children(self, parent_id: int) -> List[SystemFunction]:
        """获取子功能"""
        return self.db.query(SystemFunction).filter(
            and_(
                SystemFunction.parent_id == parent_id,
                SystemFunction.is_enabled == True
            )
        ).order_by(SystemFunction.sort_order).all()
    
    def get_function_tree(self) -> List[Dict[str, Any]]:
        """
        获取功能树形结构
        
        返回格式:
        [
            {
                "id": 1,
                "code": "system",
                "name": "系统管理",
                "level": 1,
                "children": [
                    {
                        "id": 2,
                        "code": "system.user",
                        "name": "用户管理",
                        "level": 2,
                        "children": [...]
                    }
                ]
            }
        ]
        """
        # 获取所有启用的功能
        functions = self.get_all_functions()
        
        # 构建 ID -> 功能 映射
        func_map = {f.id: self._function_to_dict(f) for f in functions}
        
        # 构建树
        roots = []
        for func in functions:
            func_dict = func_map[func.id]
            if func.parent_id is None:
                roots.append(func_dict)
            else:
                parent = func_map.get(func.parent_id)
                if parent:
                    if "children" not in parent:
                        parent["children"] = []
                    parent["children"].append(func_dict)
        
        return roots
    
    def _function_to_dict(self, func: SystemFunction) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": func.id,
            "code": func.code,
            "name": func.name,
            "description": func.description,
            "level": func.level,
            "parent_id": func.parent_id,
            "module": func.module,
            "method": func.method,
            "api_path": func.api_path,
            "icon": func.icon,
            "is_visible": func.is_visible,
            "is_audited": func.is_audited,
            "audit_level": func.audit_level,
            "sort_order": func.sort_order,
            "is_enabled": func.is_enabled,
            "children": []
        }
    
    # ==================== 创建操作 ====================
    
    def create_function(
        self,
        code: str,
        name: str,
        level: int = 1,
        parent_id: Optional[int] = None,
        **kwargs
    ) -> SystemFunction:
        """
        创建系统功能
        
        Args:
            code: 功能代码（唯一）
            name: 功能名称
            level: 层级 (1=模块, 2=功能组, 3=API端点)
            parent_id: 父级ID
            **kwargs: 其他可选字段
        """
        # 检查代码唯一性
        existing = self.get_function_by_code(code)
        if existing:
            raise ValueError(f"功能代码 '{code}' 已存在")
        
        # 验证父级
        if parent_id:
            parent = self.get_function_by_id(parent_id)
            if not parent:
                raise ValueError(f"父级功能不存在: {parent_id}")
            if parent.level >= level:
                raise ValueError(f"父级层级({parent.level})必须小于当前层级({level})")
        
        function = SystemFunction(
            code=code,
            name=name,
            level=level,
            parent_id=parent_id,
            description=kwargs.get("description"),
            module=kwargs.get("module"),
            method=kwargs.get("method"),
            api_path=kwargs.get("api_path"),
            icon=kwargs.get("icon"),
            is_visible=kwargs.get("is_visible", True),
            is_audited=kwargs.get("is_audited", True),
            audit_level=kwargs.get("audit_level", "normal"),
            sort_order=kwargs.get("sort_order", 0),
            extra_config=kwargs.get("extra_config", {}),
            is_enabled=kwargs.get("is_enabled", True),
        )
        
        self.db.add(function)
        self.db.commit()
        self.db.refresh(function)
        
        return function
    
    def create_module(self, code: str, name: str, **kwargs) -> SystemFunction:
        """创建模块（一级功能）"""
        return self.create_function(code, name, level=1, **kwargs)
    
    def create_feature_group(
        self, 
        code: str, 
        name: str, 
        parent_id: int,
        **kwargs
    ) -> SystemFunction:
        """创建功能组（二级功能）"""
        return self.create_function(code, name, level=2, parent_id=parent_id, **kwargs)
    
    def create_api_endpoint(
        self,
        code: str,
        name: str,
        parent_id: int,
        method: str,
        api_path: str,
        **kwargs
    ) -> SystemFunction:
        """创建API端点（三级功能）"""
        return self.create_function(
            code, name, level=3, parent_id=parent_id,
            method=method, api_path=api_path, **kwargs
        )
    
    # ==================== 更新操作 ====================
    
    def update_function(
        self,
        function_id: int,
        **kwargs
    ) -> Optional[SystemFunction]:
        """更新功能"""
        function = self.get_function_by_id(function_id)
        if not function:
            return None
        
        # 如果更新code，检查唯一性
        if "code" in kwargs and kwargs["code"] != function.code:
            existing = self.get_function_by_code(kwargs["code"])
            if existing:
                raise ValueError(f"功能代码 '{kwargs['code']}' 已存在")
        
        for key, value in kwargs.items():
            if hasattr(function, key):
                setattr(function, key, value)
        
        self.db.commit()
        self.db.refresh(function)
        
        return function
    
    def toggle_function(self, function_id: int, enabled: bool) -> Optional[SystemFunction]:
        """启用/禁用功能"""
        return self.update_function(function_id, is_enabled=enabled)
    
    # ==================== 删除操作 ====================
    
    def delete_function(self, function_id: int) -> bool:
        """删除功能（同时删除子功能和关联权限）"""
        function = self.get_function_by_id(function_id)
        if not function:
            return False
        
        # 递归删除子功能
        children = self.get_children(function_id)
        for child in children:
            self.delete_function(child.id)
        
        self.db.delete(function)
        self.db.commit()
        
        return True
    
    # ==================== 批量操作 ====================
    
    def batch_create_functions(
        self,
        functions_data: List[Dict[str, Any]]
    ) -> List[SystemFunction]:
        """
        批量创建功能
        
        Args:
            functions_data: 功能数据列表，支持嵌套children
        """
        created = []
        
        def create_recursive(data_list: List[Dict], parent_id: Optional[int] = None):
            for data in data_list:
                children = data.pop("children", [])
                
                func = self.create_function(
                    code=data["code"],
                    name=data["name"],
                    level=data.get("level", 1 if parent_id is None else 2),
                    parent_id=parent_id,
                    **data
                )
                created.append(func)
                
                if children:
                    create_recursive(children, func.id)
        
        create_recursive(functions_data)
        return created
    
    def init_default_functions(self) -> List[SystemFunction]:
        """初始化默认系统功能"""
        default_functions = [
            {
                "code": "system",
                "name": "系统管理",
                "icon": "ri:settings-3-line",
                "module": "system",
                "children": [
                    {
                        "code": "system.user",
                        "name": "用户管理",
                        "module": "system",
                        "children": [
                            {"code": "system.user.list", "name": "用户列表", "method": "GET", "api_path": "/api/v1/users"},
                            {"code": "system.user.create", "name": "创建用户", "method": "POST", "api_path": "/api/v1/users"},
                            {"code": "system.user.update", "name": "更新用户", "method": "PUT", "api_path": "/api/v1/users/{id}"},
                            {"code": "system.user.delete", "name": "删除用户", "method": "DELETE", "api_path": "/api/v1/users/{id}"},
                            {"code": "system.user.reset_password", "name": "重置密码", "method": "POST", "api_path": "/api/v1/users/{id}/reset-password"},
                        ]
                    },
                    {
                        "code": "system.role",
                        "name": "角色管理",
                        "module": "system",
                        "children": [
                            {"code": "system.role.list", "name": "角色列表", "method": "GET", "api_path": "/api/v1/roles"},
                            {"code": "system.role.create", "name": "创建角色", "method": "POST", "api_path": "/api/v1/roles"},
                            {"code": "system.role.update", "name": "更新角色", "method": "PUT", "api_path": "/api/v1/roles/{id}"},
                            {"code": "system.role.delete", "name": "删除角色", "method": "DELETE", "api_path": "/api/v1/roles/{id}"},
                        ]
                    },
                    {
                        "code": "system.permission",
                        "name": "权限管理",
                        "module": "system",
                        "children": [
                            {"code": "system.permission.view", "name": "查看权限", "method": "GET", "api_path": "/api/v1/permissions"},
                            {"code": "system.permission.assign", "name": "分配权限", "method": "POST", "api_path": "/api/v1/permissions/assign"},
                        ]
                    },
                    {
                        "code": "system.log",
                        "name": "操作日志",
                        "module": "system",
                        "children": [
                            {"code": "system.log.list", "name": "日志列表", "method": "GET", "api_path": "/api/v1/operation-logs"},
                            {"code": "system.log.export", "name": "导出日志", "method": "GET", "api_path": "/api/v1/operation-logs/export"},
                        ]
                    },
                    {
                        "code": "system.backup",
                        "name": "数据备份",
                        "module": "system",
                        "children": [
                            {"code": "system.backup.list", "name": "备份列表", "method": "GET", "api_path": "/api/v1/backups"},
                            {"code": "system.backup.create", "name": "创建备份", "method": "POST", "api_path": "/api/v1/backups"},
                            {"code": "system.backup.restore", "name": "恢复备份", "method": "POST", "api_path": "/api/v1/backups/{id}/restore"},
                            {"code": "system.backup.delete", "name": "删除备份", "method": "DELETE", "api_path": "/api/v1/backups/{id}"},
                        ]
                    },
                    {
                        "code": "system.config",
                        "name": "系统设置",
                        "module": "system",
                        "children": [
                            {"code": "system.config.view", "name": "查看配置", "method": "GET", "api_path": "/api/v1/system-configs"},
                            {"code": "system.config.update", "name": "更新配置", "method": "PUT", "api_path": "/api/v1/system-configs"},
                        ]
                    },
                ]
            },
            {
                "code": "nlgsm",
                "name": "NLGSM治理",
                "icon": "ri:shield-check-line",
                "module": "nlgsm",
                "children": [
                    {
                        "code": "nlgsm.dashboard",
                        "name": "治理概览",
                        "module": "nlgsm",
                        "children": [
                            {"code": "nlgsm.dashboard.view", "name": "查看概览", "method": "GET", "api_path": "/api/v1/state-machine/dashboard/stats"},
                        ]
                    },
                    {
                        "code": "nlgsm.state_machine",
                        "name": "状态机",
                        "module": "nlgsm",
                        "children": [
                            {"code": "nlgsm.state_machine.view", "name": "查看状态", "method": "GET", "api_path": "/api/v1/state-machine/current"},
                            {"code": "nlgsm.state_machine.trigger", "name": "触发事件", "method": "POST", "api_path": "/api/v1/state-machine/event"},
                            {"code": "nlgsm.state_machine.force", "name": "强制设置状态", "method": "POST", "api_path": "/api/v1/state-machine/force", "audit_level": "critical"},
                        ]
                    },
                    {
                        "code": "nlgsm.learning_unit",
                        "name": "Learning Units",
                        "module": "nlgsm",
                        "children": [
                            {"code": "nlgsm.learning_unit.list", "name": "LU列表", "method": "GET", "api_path": "/api/v1/learning-units"},
                            {"code": "nlgsm.learning_unit.view", "name": "LU详情", "method": "GET", "api_path": "/api/v1/learning-units/{id}"},
                            {"code": "nlgsm.learning_unit.approve", "name": "审批LU", "method": "POST", "api_path": "/api/v1/learning-units/{id}/approve"},
                            {"code": "nlgsm.learning_unit.reject", "name": "拒绝LU", "method": "POST", "api_path": "/api/v1/learning-units/{id}/reject"},
                        ]
                    },
                    {
                        "code": "nlgsm.learning",
                        "name": "学习控制",
                        "module": "nlgsm",
                        "children": [
                            {"code": "nlgsm.learning.view", "name": "查看学习状态", "method": "GET", "api_path": "/api/v1/learning/status"},
                            {"code": "nlgsm.learning.control", "name": "控制学习", "method": "POST", "api_path": "/api/v1/learning/*"},
                        ]
                    },
                ]
            },
        ]
        
        return self.batch_create_functions(default_functions)
