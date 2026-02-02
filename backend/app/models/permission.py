"""
权限与系统功能模型
"""
from sqlalchemy import Column, String, Boolean, ForeignKey, Integer, Text, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base, TimestampMixin


class SystemFunction(Base, TimestampMixin):
    """
    系统功能表
    
    用于定义系统所有功能模块和 API 端点
    支持层级结构：模块 -> 方法 -> API端点
    """
    __tablename__ = "system_functions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 基本信息
    code = Column(String(100), unique=True, nullable=False, index=True, comment="功能代码")
    name = Column(String(100), nullable=False, comment="功能名称")
    description = Column(Text, comment="功能描述")
    
    # 层级结构
    parent_id = Column(Integer, ForeignKey("system_functions.id"), comment="父级ID")
    level = Column(Integer, default=1, comment="层级: 1=模块, 2=功能组, 3=API端点")
    sort_order = Column(Integer, default=0, comment="排序序号")
    
    # API 相关
    module = Column(String(50), comment="所属模块")
    method = Column(String(20), comment="HTTP方法: GET/POST/PUT/DELETE")
    api_path = Column(String(200), comment="API路径")
    
    # 显示控制
    is_visible = Column(Boolean, default=True, comment="是否在菜单显示")
    icon = Column(String(50), comment="图标")
    
    # 审计配置
    is_audited = Column(Boolean, default=True, comment="是否审计")
    audit_level = Column(String(20), default="normal", comment="审计级别: none/info/normal/important/critical")
    
    # 扩展配置
    extra_config = Column(JSONB, default={}, comment="扩展配置")
    
    # 状态
    is_enabled = Column(Boolean, default=True, comment="是否启用")
    
    # 关系
    parent = relationship("SystemFunction", remote_side=[id], backref="children")
    permissions = relationship("Permission", back_populates="function")
    
    def __repr__(self):
        return f"<SystemFunction {self.code}>"


class Permission(Base, TimestampMixin):
    """
    权限表
    
    定义对系统功能的具体权限
    """
    __tablename__ = "permissions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 权限标识
    code = Column(String(100), unique=True, nullable=False, index=True, comment="权限代码")
    name = Column(String(100), nullable=False, comment="权限名称")
    description = Column(Text, comment="权限描述")
    
    # 关联的功能
    function_id = Column(Integer, ForeignKey("system_functions.id"), comment="关联功能ID")
    
    # 权限类型
    action = Column(String(50), default="access", comment="操作类型: access/create/read/update/delete/approve/export")
    
    # 资源范围
    resource_type = Column(String(50), comment="资源类型")
    resource_scope = Column(String(50), default="all", comment="资源范围: all/own/department")
    
    # 状态
    is_enabled = Column(Boolean, default=True, comment="是否启用")
    
    # 关系
    function = relationship("SystemFunction", back_populates="permissions")
    roles = relationship("Role", secondary="role_permissions", back_populates="permissions")
    
    def __repr__(self):
        return f"<Permission {self.code}>"


class RolePermission(Base):
    """角色权限关联表"""
    __tablename__ = "role_permissions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    role_id = Column(
        Integer, 
        ForeignKey("roles.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    permission_id = Column(
        Integer, 
        ForeignKey("permissions.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # 扩展约束
    constraints = Column(JSONB, default={}, comment="额外约束条件")
    
    __table_args__ = (
        {"comment": "角色权限关联表"},
    )


class UserPermission(Base, TimestampMixin):
    """
    用户权限关联表
    
    支持直接给用户分配权限（独立于角色）
    """
    __tablename__ = "user_permissions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(
        String(36),  # UUID as string for flexibility
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    permission_id = Column(
        Integer,
        ForeignKey("permissions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 授权方式
    grant_type = Column(String(20), default="allow", comment="授权类型: allow=允许, deny=拒绝")
    
    # 授权人
    granted_by = Column(String(36), comment="授权人ID")
    
    # 扩展约束
    constraints = Column(JSONB, default={}, comment="额外约束条件")
    
    # 过期时间（可选）
    expires_at = Column(DateTime(timezone=True), comment="过期时间")
    
    __table_args__ = (
        {"comment": "用户权限关联表"},
    )


class DataBackup(Base, TimestampMixin):
    """
    数据备份记录表
    """
    __tablename__ = "data_backups"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 备份信息
    backup_name = Column(String(200), nullable=False, comment="备份名称")
    backup_type = Column(String(50), nullable=False, comment="备份类型: full=完整备份, incremental=增量备份, tables=指定表")
    
    # 备份范围
    tables = Column(JSONB, default=[], comment="备份的表列表")
    
    # 文件信息
    file_path = Column(String(500), comment="备份文件路径")
    file_size = Column(Integer, default=0, comment="文件大小(字节)")
    compressed = Column(Boolean, default=True, comment="是否压缩")
    
    # 状态
    status = Column(String(20), default="pending", comment="状态: pending/running/completed/failed")
    progress = Column(Integer, default=0, comment="进度(0-100)")
    
    # 执行信息
    started_at = Column(DateTime(timezone=True), comment="开始时间")
    completed_at = Column(DateTime(timezone=True), comment="完成时间")
    duration_seconds = Column(Integer, comment="耗时(秒)")
    
    # 操作人
    created_by = Column(String(36), comment="创建人ID")
    
    # 结果
    error_message = Column(Text, comment="错误信息")
    record_counts = Column(JSONB, default={}, comment="各表记录数")
    
    # 备注
    description = Column(Text, comment="备份说明")
    
    __table_args__ = (
        {"comment": "数据备份记录表"},
    )


class SystemConfig(Base, TimestampMixin):
    """
    系统配置表
    """
    __tablename__ = "system_configs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 配置项
    config_key = Column(String(100), unique=True, nullable=False, index=True, comment="配置键")
    config_value = Column(Text, comment="配置值")
    value_type = Column(String(20), default="string", comment="值类型: string/number/boolean/json")
    
    # 分组
    config_group = Column(String(50), index=True, comment="配置分组")
    
    # 元信息
    display_name = Column(String(100), comment="显示名称")
    description = Column(Text, comment="描述")
    
    # 约束
    is_readonly = Column(Boolean, default=False, comment="是否只读")
    is_secret = Column(Boolean, default=False, comment="是否敏感信息")
    default_value = Column(Text, comment="默认值")
    validation_rules = Column(JSONB, default={}, comment="验证规则")
    
    # 修改人
    updated_by = Column(String(36), comment="修改人ID")
    
    __table_args__ = (
        {"comment": "系统配置表"},
    )

