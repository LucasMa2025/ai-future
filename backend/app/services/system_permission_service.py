"""
系统权限服务

管理角色和用户的权限分配
支持两种授权模式：角色授权和用户直接授权
"""
from typing import Optional, List, Dict, Any, Set
from uuid import UUID
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from datetime import datetime

from ..models.user import User, Role, UserRole
from ..models.permission import (
    Permission, 
    RolePermission, 
    UserPermission,
    SystemFunction,
)


class SystemPermissionService:
    """
    系统权限服务
    
    权限层次:
    1. 超级管理员 (is_superuser=True) - 拥有所有权限
    2. 用户直接权限 (user_permissions) - 可以是 allow 或 deny
    3. 角色权限 (role_permissions) - 通过角色继承
    
    权限计算优先级:
    - deny > allow
    - 用户直接权限 > 角色权限
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== 权限查询 ====================
    
    def get_all_permissions(self, include_disabled: bool = False) -> List[Permission]:
        """获取所有权限"""
        query = self.db.query(Permission)
        if not include_disabled:
            query = query.filter(Permission.is_enabled == True)
        return query.all()
    
    def get_permission_by_id(self, permission_id: int) -> Optional[Permission]:
        """根据ID获取权限"""
        return self.db.query(Permission).filter(
            Permission.id == permission_id
        ).first()
    
    def get_permission_by_code(self, code: str) -> Optional[Permission]:
        """根据代码获取权限"""
        return self.db.query(Permission).filter(
            Permission.code == code
        ).first()
    
    def get_permissions_by_function(self, function_id: int) -> List[Permission]:
        """获取功能关联的权限"""
        return self.db.query(Permission).filter(
            and_(
                Permission.function_id == function_id,
                Permission.is_enabled == True
            )
        ).all()
    
    # ==================== 权限创建 ====================
    
    def create_permission(
        self,
        code: str,
        name: str,
        function_id: Optional[int] = None,
        action: str = "access",
        **kwargs
    ) -> Permission:
        """创建权限"""
        existing = self.get_permission_by_code(code)
        if existing:
            raise ValueError(f"权限代码 '{code}' 已存在")
        
        permission = Permission(
            code=code,
            name=name,
            function_id=function_id,
            action=action,
            description=kwargs.get("description"),
            resource_type=kwargs.get("resource_type"),
            resource_scope=kwargs.get("resource_scope", "all"),
            is_enabled=kwargs.get("is_enabled", True),
        )
        
        self.db.add(permission)
        self.db.commit()
        self.db.refresh(permission)
        
        return permission
    
    def create_permissions_for_function(
        self,
        function: SystemFunction
    ) -> List[Permission]:
        """为功能自动创建CRUD权限"""
        actions = ["access", "create", "read", "update", "delete"]
        permissions = []
        
        for action in actions:
            code = f"{function.code}.{action}"
            name = f"{function.name} - {self._get_action_name(action)}"
            
            try:
                perm = self.create_permission(
                    code=code,
                    name=name,
                    function_id=function.id,
                    action=action,
                )
                permissions.append(perm)
            except ValueError:
                # 权限已存在，跳过
                pass
        
        return permissions
    
    def _get_action_name(self, action: str) -> str:
        """获取操作的中文名"""
        names = {
            "access": "访问",
            "create": "创建",
            "read": "查看",
            "update": "修改",
            "delete": "删除",
            "approve": "审批",
            "export": "导出",
        }
        return names.get(action, action)
    
    # ==================== 角色权限管理 ====================
    
    def get_role_permissions(self, role_id: int) -> List[Permission]:
        """获取角色的所有权限"""
        role = self.db.query(Role).filter(Role.id == role_id).first()
        if not role:
            return []
        return role.permissions
    
    def assign_permission_to_role(
        self,
        role_id: int,
        permission_id: int,
        constraints: Optional[Dict] = None
    ) -> RolePermission:
        """为角色分配权限"""
        # 检查是否已存在
        existing = self.db.query(RolePermission).filter(
            and_(
                RolePermission.role_id == role_id,
                RolePermission.permission_id == permission_id
            )
        ).first()
        
        if existing:
            # 更新约束
            existing.constraints = constraints or {}
            self.db.commit()
            return existing
        
        rp = RolePermission(
            role_id=role_id,
            permission_id=permission_id,
            constraints=constraints or {},
        )
        self.db.add(rp)
        self.db.commit()
        
        return rp
    
    def revoke_permission_from_role(
        self,
        role_id: int,
        permission_id: int
    ) -> bool:
        """撤销角色的权限"""
        result = self.db.query(RolePermission).filter(
            and_(
                RolePermission.role_id == role_id,
                RolePermission.permission_id == permission_id
            )
        ).delete()
        self.db.commit()
        return result > 0
    
    def set_role_permissions(
        self,
        role_id: int,
        permission_ids: List[int]
    ) -> List[RolePermission]:
        """设置角色的权限（替换现有权限）"""
        # 删除现有权限
        self.db.query(RolePermission).filter(
            RolePermission.role_id == role_id
        ).delete()
        
        # 添加新权限
        results = []
        for pid in permission_ids:
            rp = RolePermission(role_id=role_id, permission_id=pid)
            self.db.add(rp)
            results.append(rp)
        
        self.db.commit()
        return results
    
    # ==================== 用户直接权限管理 ====================
    
    def get_user_direct_permissions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的直接权限"""
        user_perms = self.db.query(UserPermission).filter(
            UserPermission.user_id == user_id
        ).all()
        
        result = []
        for up in user_perms:
            perm = self.get_permission_by_id(up.permission_id)
            if perm:
                result.append({
                    "id": up.id,
                    "permission_id": up.permission_id,
                    "permission_code": perm.code,
                    "permission_name": perm.name,
                    "grant_type": up.grant_type,
                    "expires_at": up.expires_at.isoformat() if up.expires_at else None,
                    "constraints": up.constraints,
                })
        
        return result
    
    def assign_permission_to_user(
        self,
        user_id: str,
        permission_id: int,
        grant_type: str = "allow",
        granted_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        constraints: Optional[Dict] = None
    ) -> UserPermission:
        """为用户分配直接权限"""
        # 检查是否已存在
        existing = self.db.query(UserPermission).filter(
            and_(
                UserPermission.user_id == user_id,
                UserPermission.permission_id == permission_id
            )
        ).first()
        
        if existing:
            # 更新
            existing.grant_type = grant_type
            existing.granted_by = granted_by
            existing.expires_at = expires_at
            existing.constraints = constraints or {}
            self.db.commit()
            return existing
        
        up = UserPermission(
            user_id=user_id,
            permission_id=permission_id,
            grant_type=grant_type,
            granted_by=granted_by,
            expires_at=expires_at,
            constraints=constraints or {},
        )
        self.db.add(up)
        self.db.commit()
        
        return up
    
    def revoke_permission_from_user(
        self,
        user_id: str,
        permission_id: int
    ) -> bool:
        """撤销用户的直接权限"""
        result = self.db.query(UserPermission).filter(
            and_(
                UserPermission.user_id == user_id,
                UserPermission.permission_id == permission_id
            )
        ).delete()
        self.db.commit()
        return result > 0
    
    def set_user_permissions(
        self,
        user_id: str,
        permission_data: List[Dict[str, Any]],
        granted_by: Optional[str] = None
    ) -> List[UserPermission]:
        """
        设置用户的直接权限（替换现有权限）
        
        permission_data格式:
        [
            {"permission_id": 1, "grant_type": "allow"},
            {"permission_id": 2, "grant_type": "deny"},
        ]
        """
        # 删除现有权限
        self.db.query(UserPermission).filter(
            UserPermission.user_id == user_id
        ).delete()
        
        # 添加新权限
        results = []
        for data in permission_data:
            up = UserPermission(
                user_id=user_id,
                permission_id=data["permission_id"],
                grant_type=data.get("grant_type", "allow"),
                granted_by=granted_by,
                expires_at=data.get("expires_at"),
                constraints=data.get("constraints", {}),
            )
            self.db.add(up)
            results.append(up)
        
        self.db.commit()
        return results
    
    # ==================== 权限检查 ====================
    
    def get_user_effective_permissions(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户的有效权限（合并角色和直接权限）
        
        返回格式:
        {
            "is_superuser": False,
            "permissions": {
                "system.user.list": {"allowed": True, "source": "role:admin"},
                "system.user.delete": {"allowed": False, "source": "user:deny"},
            }
        }
        """
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"is_superuser": False, "permissions": {}}
        
        if user.is_superuser:
            return {"is_superuser": True, "permissions": {"*": {"allowed": True, "source": "superuser"}}}
        
        permissions = {}
        
        # 1. 收集角色权限
        for role in user.roles:
            for perm in role.permissions:
                if perm.code not in permissions:
                    permissions[perm.code] = {
                        "allowed": True,
                        "source": f"role:{role.name}",
                    }
        
        # 2. 应用用户直接权限（覆盖角色权限）
        now = datetime.utcnow()
        user_perms = self.db.query(UserPermission).filter(
            and_(
                UserPermission.user_id == user_id,
                or_(
                    UserPermission.expires_at.is_(None),
                    UserPermission.expires_at > now
                )
            )
        ).all()
        
        for up in user_perms:
            perm = self.get_permission_by_id(up.permission_id)
            if perm:
                permissions[perm.code] = {
                    "allowed": up.grant_type == "allow",
                    "source": f"user:{up.grant_type}",
                }
        
        return {
            "is_superuser": False,
            "permissions": permissions
        }
    
    def check_permission(
        self,
        user_id: str,
        permission_code: str
    ) -> bool:
        """检查用户是否拥有指定权限"""
        effective = self.get_user_effective_permissions(user_id)
        
        if effective["is_superuser"]:
            return True
        
        perms = effective["permissions"]
        
        # 检查通配符
        if "*" in perms and perms["*"]["allowed"]:
            return True
        
        # 检查具体权限
        if permission_code in perms:
            return perms[permission_code]["allowed"]
        
        # 检查父级权限（如 system.user 允许访问 system.user.list）
        parts = permission_code.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_code = ".".join(parts[:i])
            if parent_code in perms and perms[parent_code]["allowed"]:
                return True
        
        return False
    
    def get_permission_tree_with_status(
        self,
        user_id: Optional[str] = None,
        role_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        获取权限树，带有授权状态
        
        用于权限分配页面展示
        """
        from .system_function_service import SystemFunctionService
        
        func_service = SystemFunctionService(self.db)
        func_tree = func_service.get_function_tree()
        
        # 获取目标的当前权限
        current_perms: Set[int] = set()
        
        if role_id:
            role_perms = self.get_role_permissions(role_id)
            current_perms = {p.id for p in role_perms}
        elif user_id:
            user_perms = self.db.query(UserPermission).filter(
                and_(
                    UserPermission.user_id == user_id,
                    UserPermission.grant_type == "allow"
                )
            ).all()
            current_perms = {up.permission_id for up in user_perms}
        
        # 标记权限状态
        def mark_status(nodes: List[Dict]) -> List[Dict]:
            for node in nodes:
                # 获取功能关联的权限
                perms = self.get_permissions_by_function(node["id"])
                node["permissions"] = [
                    {
                        "id": p.id,
                        "code": p.code,
                        "name": p.name,
                        "action": p.action,
                        "checked": p.id in current_perms,
                    }
                    for p in perms
                ]
                
                if node.get("children"):
                    mark_status(node["children"])
            
            return nodes
        
        return mark_status(func_tree)
