"""
用户服务

实现用户相关的业务逻辑
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import or_

from ..models.user import User, Role, UserRole
from ..core.security import get_password_hash, verify_password
from ..core.exceptions import NotFoundError, AlreadyExistsError, BusinessError


class UserService:
    """
    用户服务
    
    处理用户 CRUD 和相关业务逻辑
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_id(self, user_id: UUID) -> Optional[User]:
        """根据ID获取用户"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return self.db.query(User).filter(User.email == email).first()
    
    def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        is_active: Optional[bool] = None,
        role_name: Optional[str] = None,
        search: Optional[str] = None,
    ) -> tuple[List[User], int]:
        """
        获取用户列表
        
        Returns:
            (用户列表, 总数)
        """
        query = self.db.query(User)
        
        # 过滤条件
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        if role_name:
            query = query.join(User.roles).filter(Role.name == role_name)
        
        if search:
            search_pattern = f"%{search}%"
            query = query.filter(
                or_(
                    User.username.ilike(search_pattern),
                    User.email.ilike(search_pattern),
                    User.full_name.ilike(search_pattern),
                )
            )
        
        # 总数
        total = query.count()
        
        # 分页
        users = query.order_by(User.created_at.desc()).offset(skip).limit(limit).all()
        
        return users, total
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        phone: Optional[str] = None,
        is_active: bool = True,
        is_superuser: bool = False,
        role_ids: Optional[List[int]] = None,
    ) -> User:
        """创建用户"""
        # 检查用户名是否存在
        if self.get_by_username(username):
            raise AlreadyExistsError("User", username)
        
        # 检查邮箱是否存在
        if self.get_by_email(email):
            raise AlreadyExistsError("User", email)
        
        # 创建用户
        user = User(
            username=username,
            email=email,
            hashed_password=get_password_hash(password),
            full_name=full_name,
            phone=phone,
            is_active=is_active,
            is_superuser=is_superuser,
        )
        
        self.db.add(user)
        self.db.flush()  # 获取用户ID
        
        # 分配角色
        if role_ids:
            roles = self.db.query(Role).filter(Role.id.in_(role_ids)).all()
            user.roles = roles
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def update_user(
        self,
        user_id: UUID,
        data: Dict[str, Any],
        updated_by: Optional[UUID] = None
    ) -> User:
        """更新用户"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        # 检查用户名唯一性
        if "username" in data and data["username"] != user.username:
            existing = self.get_by_username(data["username"])
            if existing:
                raise AlreadyExistsError("User", data["username"])
        
        # 检查邮箱唯一性
        if "email" in data and data["email"] != user.email:
            existing = self.get_by_email(data["email"])
            if existing:
                raise AlreadyExistsError("User", data["email"])
        
        # 更新字段
        allowed_fields = {
            "username", "email", "full_name", "phone", 
            "is_active", "avatar", "notification_preferences"
        }
        
        for field, value in data.items():
            if field in allowed_fields and value is not None:
                setattr(user, field, value)
        
        # 更新角色
        if "role_ids" in data:
            roles = self.db.query(Role).filter(Role.id.in_(data["role_ids"])).all()
            user.roles = roles
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def change_password(
        self,
        user_id: UUID,
        old_password: str,
        new_password: str
    ) -> bool:
        """修改密码"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        # 验证旧密码
        if not verify_password(old_password, user.hashed_password):
            raise BusinessError("Old password is incorrect", code="INVALID_PASSWORD")
        
        # 更新密码
        user.hashed_password = get_password_hash(new_password)
        self.db.commit()
        
        return True
    
    def reset_password(
        self,
        user_id: UUID,
        new_password: str,
        reset_by: Optional[UUID] = None
    ) -> bool:
        """重置密码（管理员操作）"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        user.hashed_password = get_password_hash(new_password)
        self.db.commit()
        
        return True
    
    def delete_user(self, user_id: UUID) -> bool:
        """删除用户"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        # 不允许删除超级管理员
        if user.is_superuser:
            raise BusinessError("Cannot delete superuser", code="CANNOT_DELETE_SUPERUSER")
        
        self.db.delete(user)
        self.db.commit()
        
        return True
    
    def deactivate_user(self, user_id: UUID) -> User:
        """停用用户"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        user.is_active = False
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def activate_user(self, user_id: UUID) -> User:
        """激活用户"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        user.is_active = True
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def assign_roles(
        self,
        user_id: UUID,
        role_ids: List[int],
        assigned_by: Optional[UUID] = None
    ) -> User:
        """分配角色"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        roles = self.db.query(Role).filter(Role.id.in_(role_ids)).all()
        if len(roles) != len(role_ids):
            raise BusinessError("Some roles not found", code="ROLES_NOT_FOUND")
        
        user.roles = roles
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def get_user_permissions(self, user_id: UUID) -> List[str]:
        """获取用户所有权限"""
        user = self.get_by_id(user_id)
        if not user:
            raise NotFoundError("User", user_id)
        
        if user.is_superuser:
            return ["*"]
        
        permissions = set()
        for role in user.roles:
            for perm in role.permissions:
                permissions.add(perm.code)
        
        return list(permissions)
    
    def can_user_approve_risk_level(
        self,
        user_id: UUID,
        risk_level: str
    ) -> bool:
        """检查用户是否可以审批指定风险等级"""
        user = self.get_by_id(user_id)
        if not user:
            return False
        
        if user.is_superuser:
            return True
        
        risk_order = ["low", "medium", "high", "critical"]
        
        for role in user.roles:
            if role.risk_level_limit:
                try:
                    user_level_idx = risk_order.index(role.risk_level_limit)
                    required_idx = risk_order.index(risk_level)
                    if required_idx <= user_level_idx:
                        return True
                except ValueError:
                    continue
        
        return False

