"""
角色服务

实现角色相关的业务逻辑
"""
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session

from ..models.user import Role
from ..models.permission import Permission, RolePermission
from ..core.exceptions import NotFoundError, AlreadyExistsError, BusinessError


class RoleService:
    """
    角色服务
    
    处理角色 CRUD 和权限分配
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, role_id: int) -> Optional[Role]:
        """根据ID获取角色"""
        return self.db.query(Role).filter(Role.id == role_id).first()
    
    def get_by_name(self, name: str) -> Optional[Role]:
        """根据名称获取角色"""
        return self.db.query(Role).filter(Role.name == name).first()
    
    def list_roles(
        self,
        include_system: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> tuple[List[Role], int]:
        """
        获取角色列表
        
        Returns:
            (角色列表, 总数)
        """
        query = self.db.query(Role)
        
        if not include_system:
            query = query.filter(Role.is_system == False)
        
        total = query.count()
        roles = query.order_by(Role.sort_order, Role.id).offset(skip).limit(limit).all()
        
        return roles, total
    
    def create_role(
        self,
        name: str,
        display_name: str,
        description: Optional[str] = None,
        risk_level_limit: Optional[str] = None,
        permission_ids: Optional[List[int]] = None,
        sort_order: int = 0
    ) -> Role:
        """创建角色"""
        # 检查名称是否存在
        if self.get_by_name(name):
            raise AlreadyExistsError("Role", name)
        
        # 创建角色
        role = Role(
            name=name,
            display_name=display_name,
            description=description,
            risk_level_limit=risk_level_limit,
            is_system=False,
            sort_order=sort_order,
        )
        
        self.db.add(role)
        self.db.flush()
        
        # 分配权限
        if permission_ids:
            permissions = self.db.query(Permission).filter(
                Permission.id.in_(permission_ids)
            ).all()
            role.permissions = permissions
        
        self.db.commit()
        self.db.refresh(role)
        
        return role
    
    def update_role(
        self,
        role_id: int,
        data: Dict[str, Any]
    ) -> Role:
        """更新角色"""
        role = self.get_by_id(role_id)
        if not role:
            raise NotFoundError("Role", role_id)
        
        # 系统角色只允许修改部分字段
        if role.is_system:
            allowed_fields = {"display_name", "description"}
        else:
            allowed_fields = {
                "name", "display_name", "description", 
                "risk_level_limit", "sort_order"
            }
        
        # 检查名称唯一性
        if "name" in data and data["name"] != role.name:
            existing = self.get_by_name(data["name"])
            if existing:
                raise AlreadyExistsError("Role", data["name"])
        
        # 更新字段
        for field, value in data.items():
            if field in allowed_fields and value is not None:
                setattr(role, field, value)
        
        # 更新权限
        if "permission_ids" in data and not role.is_system:
            permissions = self.db.query(Permission).filter(
                Permission.id.in_(data["permission_ids"])
            ).all()
            role.permissions = permissions
        
        self.db.commit()
        self.db.refresh(role)
        
        return role
    
    def delete_role(self, role_id: int) -> bool:
        """删除角色"""
        role = self.get_by_id(role_id)
        if not role:
            raise NotFoundError("Role", role_id)
        
        # 不允许删除系统角色
        if role.is_system:
            raise BusinessError("Cannot delete system role", code="CANNOT_DELETE_SYSTEM_ROLE")
        
        # 检查是否有用户使用该角色
        if role.users:
            raise BusinessError(
                f"Role is assigned to {len(role.users)} users",
                code="ROLE_IN_USE"
            )
        
        self.db.delete(role)
        self.db.commit()
        
        return True
    
    def assign_permissions(
        self,
        role_id: int,
        permission_ids: List[int]
    ) -> Role:
        """分配权限"""
        role = self.get_by_id(role_id)
        if not role:
            raise NotFoundError("Role", role_id)
        
        # 系统角色不允许修改权限
        if role.is_system:
            raise BusinessError(
                "Cannot modify system role permissions",
                code="CANNOT_MODIFY_SYSTEM_ROLE"
            )
        
        permissions = self.db.query(Permission).filter(
            Permission.id.in_(permission_ids)
        ).all()
        
        role.permissions = permissions
        self.db.commit()
        self.db.refresh(role)
        
        return role
    
    def get_role_permissions(self, role_id: int) -> List[Permission]:
        """获取角色权限"""
        role = self.get_by_id(role_id)
        if not role:
            raise NotFoundError("Role", role_id)
        
        return list(role.permissions)
    
    def get_role_users_count(self, role_id: int) -> int:
        """获取角色下的用户数量"""
        role = self.get_by_id(role_id)
        if not role:
            raise NotFoundError("Role", role_id)
        
        return len(role.users)

