"""
系统配置服务
"""
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
import json

from ..models.permission import SystemConfig


class SystemConfigService:
    """系统配置服务"""
    
    # 默认配置
    DEFAULT_CONFIGS = [
        # 系统基础配置
        {
            "config_key": "system.name",
            "config_value": "NLGSM 治理系统",
            "value_type": "string",
            "config_group": "system",
            "display_name": "系统名称",
            "is_readonly": False,
        },
        {
            "config_key": "system.version",
            "config_value": "4.0.0",
            "value_type": "string",
            "config_group": "system",
            "display_name": "系统版本",
            "is_readonly": True,
        },
        {
            "config_key": "system.maintenance_mode",
            "config_value": "false",
            "value_type": "boolean",
            "config_group": "system",
            "display_name": "维护模式",
            "description": "开启后普通用户无法访问系统",
        },
        
        # 安全配置
        {
            "config_key": "security.session_timeout",
            "config_value": "3600",
            "value_type": "number",
            "config_group": "security",
            "display_name": "会话超时时间(秒)",
            "default_value": "3600",
        },
        {
            "config_key": "security.max_login_attempts",
            "config_value": "5",
            "value_type": "number",
            "config_group": "security",
            "display_name": "最大登录尝试次数",
            "default_value": "5",
        },
        {
            "config_key": "security.lockout_duration",
            "config_value": "300",
            "value_type": "number",
            "config_group": "security",
            "display_name": "账户锁定时长(秒)",
            "default_value": "300",
        },
        {
            "config_key": "security.password_min_length",
            "config_value": "8",
            "value_type": "number",
            "config_group": "security",
            "display_name": "密码最小长度",
            "default_value": "8",
        },
        
        # 审计配置
        {
            "config_key": "audit.enabled",
            "config_value": "true",
            "value_type": "boolean",
            "config_group": "audit",
            "display_name": "启用审计日志",
        },
        {
            "config_key": "audit.retention_days",
            "config_value": "90",
            "value_type": "number",
            "config_group": "audit",
            "display_name": "审计日志保留天数",
        },
        {
            "config_key": "audit.log_request_body",
            "config_value": "true",
            "value_type": "boolean",
            "config_group": "audit",
            "display_name": "记录请求体",
        },
        
        # 备份配置
        {
            "config_key": "backup.auto_backup",
            "config_value": "true",
            "value_type": "boolean",
            "config_group": "backup",
            "display_name": "自动备份",
        },
        {
            "config_key": "backup.schedule",
            "config_value": "0 2 * * *",
            "value_type": "string",
            "config_group": "backup",
            "display_name": "备份计划(Cron表达式)",
            "description": "默认每天凌晨2点执行",
        },
        {
            "config_key": "backup.retention_count",
            "config_value": "7",
            "value_type": "number",
            "config_group": "backup",
            "display_name": "保留备份数量",
        },
        
        # 学习控制配置
        {
            "config_key": "learning.auto_checkpoint",
            "config_value": "true",
            "value_type": "boolean",
            "config_group": "learning",
            "display_name": "自动创建检查点",
        },
        {
            "config_key": "learning.checkpoint_interval",
            "config_value": "3600",
            "value_type": "number",
            "config_group": "learning",
            "display_name": "检查点间隔(秒)",
        },
        {
            "config_key": "learning.max_depth",
            "config_value": "10",
            "value_type": "number",
            "config_group": "learning",
            "display_name": "最大学习深度",
        },
    ]
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== 查询操作 ====================
    
    def get_all_configs(
        self,
        group: Optional[str] = None,
        include_secret: bool = False,
    ) -> List[Dict[str, Any]]:
        """获取所有配置"""
        query = self.db.query(SystemConfig)
        
        if group:
            query = query.filter(SystemConfig.config_group == group)
        
        configs = query.order_by(SystemConfig.config_group, SystemConfig.config_key).all()
        
        return [
            self._config_to_dict(c, include_secret) 
            for c in configs
        ]
    
    def get_config(self, key: str) -> Optional[SystemConfig]:
        """获取单个配置"""
        return self.db.query(SystemConfig).filter(
            SystemConfig.config_key == key
        ).first()
    
    def get_config_value(
        self, 
        key: str, 
        default: Any = None,
        cast_type: bool = True
    ) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            cast_type: 是否转换类型
        """
        config = self.get_config(key)
        if not config:
            return default
        
        if not cast_type:
            return config.config_value
        
        return self._cast_value(config.config_value, config.value_type)
    
    def get_configs_by_group(self, group: str) -> Dict[str, Any]:
        """获取指定分组的所有配置"""
        configs = self.db.query(SystemConfig).filter(
            SystemConfig.config_group == group
        ).all()
        
        return {
            c.config_key: self._cast_value(c.config_value, c.value_type)
            for c in configs
        }
    
    def _config_to_dict(
        self, 
        config: SystemConfig,
        include_secret: bool = False
    ) -> Dict[str, Any]:
        """转换为字典"""
        value = config.config_value
        
        # 隐藏敏感值
        if config.is_secret and not include_secret:
            value = "******"
        
        return {
            "id": config.id,
            "config_key": config.config_key,
            "config_value": value,
            "value_type": config.value_type,
            "config_group": config.config_group,
            "display_name": config.display_name,
            "description": config.description,
            "is_readonly": config.is_readonly,
            "is_secret": config.is_secret,
            "default_value": config.default_value,
            "updated_at": config.updated_at.isoformat() if config.updated_at else None,
            "updated_by": config.updated_by,
        }
    
    def _cast_value(self, value: str, value_type: str) -> Any:
        """转换配置值类型"""
        if value is None:
            return None
        
        try:
            if value_type == "number":
                return int(value) if "." not in value else float(value)
            elif value_type == "boolean":
                return value.lower() in ("true", "1", "yes")
            elif value_type == "json":
                return json.loads(value)
            else:
                return value
        except:
            return value
    
    # ==================== 更新操作 ====================
    
    def set_config(
        self,
        key: str,
        value: Any,
        updated_by: Optional[str] = None,
    ) -> SystemConfig:
        """设置配置值"""
        config = self.get_config(key)
        
        if not config:
            raise ValueError(f"配置项 '{key}' 不存在")
        
        if config.is_readonly:
            raise ValueError(f"配置项 '{key}' 是只读的")
        
        # 转换为字符串
        if isinstance(value, bool):
            str_value = "true" if value else "false"
        elif isinstance(value, (dict, list)):
            str_value = json.dumps(value, ensure_ascii=False)
        else:
            str_value = str(value)
        
        # 验证值
        if config.validation_rules:
            self._validate_value(str_value, config.validation_rules)
        
        config.config_value = str_value
        config.updated_by = updated_by
        
        self.db.commit()
        self.db.refresh(config)
        
        return config
    
    def batch_set_configs(
        self,
        configs: Dict[str, Any],
        updated_by: Optional[str] = None,
    ) -> List[SystemConfig]:
        """批量设置配置"""
        results = []
        
        for key, value in configs.items():
            try:
                config = self.set_config(key, value, updated_by)
                results.append(config)
            except ValueError:
                # 跳过不存在或只读的配置
                pass
        
        return results
    
    def _validate_value(self, value: str, rules: Dict) -> None:
        """验证配置值"""
        if "min" in rules:
            try:
                if float(value) < rules["min"]:
                    raise ValueError(f"值不能小于 {rules['min']}")
            except ValueError as e:
                if "小于" in str(e):
                    raise
        
        if "max" in rules:
            try:
                if float(value) > rules["max"]:
                    raise ValueError(f"值不能大于 {rules['max']}")
            except ValueError as e:
                if "大于" in str(e):
                    raise
        
        if "pattern" in rules:
            import re
            if not re.match(rules["pattern"], value):
                raise ValueError(f"值格式不正确")
        
        if "enum" in rules:
            if value not in rules["enum"]:
                raise ValueError(f"值必须是以下之一: {rules['enum']}")
    
    # ==================== 创建操作 ====================
    
    def create_config(
        self,
        config_key: str,
        config_value: str,
        value_type: str = "string",
        config_group: Optional[str] = None,
        **kwargs
    ) -> SystemConfig:
        """创建配置项"""
        existing = self.get_config(config_key)
        if existing:
            raise ValueError(f"配置项 '{config_key}' 已存在")
        
        config = SystemConfig(
            config_key=config_key,
            config_value=config_value,
            value_type=value_type,
            config_group=config_group,
            display_name=kwargs.get("display_name"),
            description=kwargs.get("description"),
            is_readonly=kwargs.get("is_readonly", False),
            is_secret=kwargs.get("is_secret", False),
            default_value=kwargs.get("default_value"),
            validation_rules=kwargs.get("validation_rules", {}),
        )
        
        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)
        
        return config
    
    # ==================== 初始化 ====================
    
    def init_default_configs(self) -> int:
        """初始化默认配置"""
        created_count = 0
        
        for config_data in self.DEFAULT_CONFIGS:
            key = config_data["config_key"]
            existing = self.get_config(key)
            
            if not existing:
                config = SystemConfig(**config_data)
                self.db.add(config)
                created_count += 1
        
        self.db.commit()
        return created_count
    
    # ==================== 分组信息 ====================
    
    def get_config_groups(self) -> List[Dict[str, Any]]:
        """获取配置分组信息"""
        groups = {
            "system": {"name": "系统基础", "icon": "ri:settings-line"},
            "security": {"name": "安全设置", "icon": "ri:shield-line"},
            "audit": {"name": "审计配置", "icon": "ri:file-list-line"},
            "backup": {"name": "备份配置", "icon": "ri:database-2-line"},
            "learning": {"name": "学习控制", "icon": "ri:brain-line"},
        }
        
        result = []
        for key, info in groups.items():
            count = self.db.query(SystemConfig).filter(
                SystemConfig.config_group == key
            ).count()
            
            result.append({
                "key": key,
                "name": info["name"],
                "icon": info["icon"],
                "config_count": count,
            })
        
        return result
