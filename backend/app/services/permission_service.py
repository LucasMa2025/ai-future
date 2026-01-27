"""
权限服务

实现权限和系统功能相关的业务逻辑
"""
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from ..models.permission import Permission, SystemFunction
from ..core.exceptions import NotFoundError, AlreadyExistsError


class PermissionService:
    """
    权限服务
    
    处理权限和系统功能的管理
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== 权限管理 ====================
    
    def get_permission_by_id(self, permission_id: int) -> Optional[Permission]:
        """根据ID获取权限"""
        return self.db.query(Permission).filter(Permission.id == permission_id).first()
    
    def get_permission_by_code(self, code: str) -> Optional[Permission]:
        """根据代码获取权限"""
        return self.db.query(Permission).filter(Permission.code == code).first()
    
    def list_permissions(
        self,
        function_id: Optional[int] = None,
        action: Optional[str] = None,
        is_enabled: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100
    ) -> tuple[List[Permission], int]:
        """获取权限列表"""
        query = self.db.query(Permission)
        
        if function_id is not None:
            query = query.filter(Permission.function_id == function_id)
        
        if action:
            query = query.filter(Permission.action == action)
        
        if is_enabled is not None:
            query = query.filter(Permission.is_enabled == is_enabled)
        
        total = query.count()
        permissions = query.order_by(Permission.id).offset(skip).limit(limit).all()
        
        return permissions, total
    
    def create_permission(
        self,
        code: str,
        name: str,
        description: Optional[str] = None,
        function_id: Optional[int] = None,
        action: str = "access",
        resource_type: Optional[str] = None,
        resource_scope: str = "all"
    ) -> Permission:
        """创建权限"""
        if self.get_permission_by_code(code):
            raise AlreadyExistsError("Permission", code)
        
        permission = Permission(
            code=code,
            name=name,
            description=description,
            function_id=function_id,
            action=action,
            resource_type=resource_type,
            resource_scope=resource_scope,
        )
        
        self.db.add(permission)
        self.db.commit()
        self.db.refresh(permission)
        
        return permission
    
    def update_permission(
        self,
        permission_id: int,
        data: Dict[str, Any]
    ) -> Permission:
        """更新权限"""
        permission = self.get_permission_by_id(permission_id)
        if not permission:
            raise NotFoundError("Permission", permission_id)
        
        # 检查代码唯一性
        if "code" in data and data["code"] != permission.code:
            existing = self.get_permission_by_code(data["code"])
            if existing:
                raise AlreadyExistsError("Permission", data["code"])
        
        allowed_fields = {
            "code", "name", "description", "action",
            "resource_type", "resource_scope", "is_enabled"
        }
        
        for field, value in data.items():
            if field in allowed_fields:
                setattr(permission, field, value)
        
        self.db.commit()
        self.db.refresh(permission)
        
        return permission
    
    def delete_permission(self, permission_id: int) -> bool:
        """删除权限"""
        permission = self.get_permission_by_id(permission_id)
        if not permission:
            raise NotFoundError("Permission", permission_id)
        
        self.db.delete(permission)
        self.db.commit()
        
        return True
    
    # ==================== 系统功能管理 ====================
    
    def get_function_by_id(self, function_id: int) -> Optional[SystemFunction]:
        """根据ID获取系统功能"""
        return self.db.query(SystemFunction).filter(SystemFunction.id == function_id).first()
    
    def get_function_by_code(self, code: str) -> Optional[SystemFunction]:
        """根据代码获取系统功能"""
        return self.db.query(SystemFunction).filter(SystemFunction.code == code).first()
    
    def list_functions(
        self,
        level: Optional[int] = None,
        module: Optional[str] = None,
        parent_id: Optional[int] = None,
        is_visible: Optional[bool] = None,
        is_enabled: Optional[bool] = None
    ) -> List[SystemFunction]:
        """获取系统功能列表"""
        query = self.db.query(SystemFunction)
        
        if level is not None:
            query = query.filter(SystemFunction.level == level)
        
        if module:
            query = query.filter(SystemFunction.module == module)
        
        if parent_id is not None:
            query = query.filter(SystemFunction.parent_id == parent_id)
        elif level == 1:
            # 顶级模块没有父级
            query = query.filter(SystemFunction.parent_id.is_(None))
        
        if is_visible is not None:
            query = query.filter(SystemFunction.is_visible == is_visible)
        
        if is_enabled is not None:
            query = query.filter(SystemFunction.is_enabled == is_enabled)
        
        return query.order_by(SystemFunction.sort_order, SystemFunction.id).all()
    
    def get_function_tree(self) -> List[Dict[str, Any]]:
        """获取功能树形结构"""
        # 获取所有功能
        functions = self.db.query(SystemFunction).order_by(
            SystemFunction.sort_order, SystemFunction.id
        ).all()
        
        # 构建树
        func_dict = {f.id: self._function_to_dict(f) for f in functions}
        tree = []
        
        for func in functions:
            node = func_dict[func.id]
            if func.parent_id is None:
                tree.append(node)
            else:
                parent = func_dict.get(func.parent_id)
                if parent:
                    if "children" not in parent:
                        parent["children"] = []
                    parent["children"].append(node)
        
        return tree
    
    def create_function(
        self,
        code: str,
        name: str,
        level: int = 1,
        **kwargs
    ) -> SystemFunction:
        """创建系统功能"""
        if self.get_function_by_code(code):
            raise AlreadyExistsError("SystemFunction", code)
        
        function = SystemFunction(
            code=code,
            name=name,
            level=level,
            **kwargs
        )
        
        self.db.add(function)
        self.db.commit()
        self.db.refresh(function)
        
        return function
    
    def update_function(
        self,
        function_id: int,
        data: Dict[str, Any]
    ) -> SystemFunction:
        """更新系统功能"""
        function = self.get_function_by_id(function_id)
        if not function:
            raise NotFoundError("SystemFunction", function_id)
        
        # 检查代码唯一性
        if "code" in data and data["code"] != function.code:
            existing = self.get_function_by_code(data["code"])
            if existing:
                raise AlreadyExistsError("SystemFunction", data["code"])
        
        allowed_fields = {
            "code", "name", "description", "parent_id", "level",
            "sort_order", "module", "method", "api_path",
            "is_visible", "icon", "is_audited", "audit_level",
            "extra_config", "is_enabled"
        }
        
        for field, value in data.items():
            if field in allowed_fields:
                setattr(function, field, value)
        
        self.db.commit()
        self.db.refresh(function)
        
        return function
    
    def delete_function(self, function_id: int) -> bool:
        """删除系统功能"""
        function = self.get_function_by_id(function_id)
        if not function:
            raise NotFoundError("SystemFunction", function_id)
        
        # 检查是否有子功能
        children = self.db.query(SystemFunction).filter(
            SystemFunction.parent_id == function_id
        ).count()
        
        if children > 0:
            from ..core.exceptions import BusinessError
            raise BusinessError(
                f"Function has {children} children",
                code="HAS_CHILDREN"
            )
        
        self.db.delete(function)
        self.db.commit()
        
        return True
    
    def get_audit_config(
        self,
        method: str,
        path: str
    ) -> Optional[Dict[str, Any]]:
        """获取API的审计配置"""
        # 尝试精确匹配
        function = self.db.query(SystemFunction).filter(
            SystemFunction.method == method,
            SystemFunction.api_path == path
        ).first()
        
        if function:
            return {
                "function_code": function.code,
                "is_audited": function.is_audited,
                "audit_level": function.audit_level,
            }
        
        return None
    
    def _function_to_dict(self, func: SystemFunction) -> Dict[str, Any]:
        """转换功能为字典"""
        return {
            "id": func.id,
            "code": func.code,
            "name": func.name,
            "description": func.description,
            "level": func.level,
            "sort_order": func.sort_order,
            "module": func.module,
            "method": func.method,
            "api_path": func.api_path,
            "is_visible": func.is_visible,
            "icon": func.icon,
            "is_audited": func.is_audited,
            "audit_level": func.audit_level,
            "is_enabled": func.is_enabled,
        }

