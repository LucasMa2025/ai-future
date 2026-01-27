"""
业务审计中间件

实现:
1. 自动记录重要业务操作
2. 根据 SystemFunction 配置决定是否审计
3. 链式哈希保证审计日志不可篡改
"""
import hashlib
import json
import uuid
from datetime import datetime
from typing import Callable, Optional, Dict, Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from sqlalchemy.orm import Session

from ..config import settings
from ..db.session import SessionLocal
from ..models.audit import BusinessAuditLog
from ..models.permission import SystemFunction


class AuditMiddleware(BaseHTTPMiddleware):
    """
    业务审计中间件
    
    在请求处理后，根据配置决定是否记录审计日志
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._function_cache: Dict[str, Dict] = {}
        self._last_hash: str = "GENESIS"
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """处理请求"""
        if not settings.AUDIT_ENABLED:
            return await call_next(request)
        
        # 跳过不需要审计的路径
        if self._should_skip(request.url.path):
            return await call_next(request)
        
        # 记录请求开始时间
        start_time = datetime.utcnow()
        
        # 获取请求体（如果有）
        request_body = None
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.body()
                if body:
                    request_body = json.loads(body.decode())
                    # 脱敏处理
                    request_body = self._sanitize_data(request_body)
            except:
                pass
        
        # 执行请求
        response = await call_next(request)
        
        # 获取审计配置
        audit_config = self._get_audit_config(request.method, request.url.path)
        
        if audit_config and audit_config.get("is_audited"):
            # 记录审计日志
            await self._create_audit_log(
                request=request,
                response=response,
                start_time=start_time,
                request_body=request_body,
                audit_config=audit_config,
            )
        
        return response
    
    def _should_skip(self, path: str) -> bool:
        """判断是否跳过审计"""
        skip_paths = [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
            "/ws",
        ]
        return any(path.startswith(p) for p in skip_paths)
    
    def _get_audit_config(self, method: str, path: str) -> Optional[Dict]:
        """获取审计配置"""
        # 生成缓存键
        cache_key = f"{method}:{path}"
        
        if cache_key in self._function_cache:
            return self._function_cache[cache_key]
        
        # 从数据库查询
        db = SessionLocal()
        try:
            # 尝试精确匹配
            func = db.query(SystemFunction).filter(
                SystemFunction.method == method,
                SystemFunction.api_path == path,
            ).first()
            
            # 如果没有精确匹配，尝试模糊匹配（处理路径参数）
            if not func:
                # 将路径中的 ID 替换为 {id}
                normalized_path = self._normalize_path(path)
                func = db.query(SystemFunction).filter(
                    SystemFunction.method == method,
                    SystemFunction.api_path == normalized_path,
                ).first()
            
            if func:
                config = {
                    "function_code": func.code,
                    "function_name": func.name,
                    "is_audited": func.is_audited,
                    "audit_level": func.audit_level,
                    "module": func.module,
                }
                self._function_cache[cache_key] = config
                return config
            
            return None
            
        finally:
            db.close()
    
    def _normalize_path(self, path: str) -> str:
        """标准化路径（将 UUID/ID 替换为参数占位符）"""
        import re
        
        # 替换 UUID
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}',
            path
        )
        
        # 替换纯数字 ID
        path = re.sub(r'/\d+(?=/|$)', '/{id}', path)
        
        return path
    
    def _sanitize_data(self, data: Dict) -> Dict:
        """脱敏处理敏感数据"""
        sensitive_keys = {"password", "token", "secret", "api_key", "credit_card"}
        
        def sanitize(obj):
            if isinstance(obj, dict):
                return {
                    k: "***REDACTED***" if k.lower() in sensitive_keys else sanitize(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            return obj
        
        return sanitize(data)
    
    async def _create_audit_log(
        self,
        request: Request,
        response: Response,
        start_time: datetime,
        request_body: Optional[Dict],
        audit_config: Dict,
    ) -> None:
        """创建审计日志"""
        db = SessionLocal()
        try:
            # 获取用户信息
            user_id = None
            user_name = None
            if hasattr(request.state, "user") and request.state.user:
                user_id = request.state.user.id
                user_name = request.state.user.username
            
            # 生成条目 ID
            entry_id = str(uuid.uuid4())
            
            # 确定操作动作
            action = self._determine_action(request.method, request.url.path)
            
            # 确定目标
            target_type, target_id = self._extract_target(request.url.path)
            
            # 计算哈希
            entry_hash = self._compute_hash(
                entry_id=entry_id,
                previous_hash=self._last_hash,
                event_type=audit_config.get("module", "unknown"),
                action=action,
                actor_id=str(user_id) if user_id else None,
                timestamp=start_time.isoformat(),
            )
            
            # 创建日志条目
            audit_log = BusinessAuditLog(
                entry_id=entry_id,
                event_type=audit_config.get("module", "unknown"),
                event_category=self._determine_category(audit_config.get("module")),
                actor_id=user_id,
                actor_name=user_name,
                actor_ip=self._get_client_ip(request),
                action=action,
                target_type=target_type,
                target_id=target_id,
                details={
                    "function_code": audit_config.get("function_code"),
                    "function_name": audit_config.get("function_name"),
                    "path": str(request.url.path),
                    "query_params": dict(request.query_params),
                },
                request_data=request_body,
                result="success" if response.status_code < 400 else "failure",
                audit_level=audit_config.get("audit_level", "normal"),
                previous_hash=self._last_hash,
                entry_hash=entry_hash,
            )
            
            db.add(audit_log)
            db.commit()
            
            # 更新最后哈希
            self._last_hash = entry_hash
            
        except Exception as e:
            db.rollback()
            # 审计失败不应影响请求
            import logging
            logging.getLogger(__name__).error(f"Audit log failed: {e}")
        finally:
            db.close()
    
    def _determine_action(self, method: str, path: str) -> str:
        """确定操作动作"""
        method_action_map = {
            "GET": "view",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        return method_action_map.get(method, "unknown")
    
    def _determine_category(self, module: Optional[str]) -> str:
        """确定事件分类"""
        category_map = {
            "state_machine": "state",
            "approvals": "approval",
            "artifacts": "artifact",
            "users": "user",
            "roles": "user",
            "settings": "system",
            "anomaly": "anomaly",
        }
        return category_map.get(module, "system")
    
    def _extract_target(self, path: str) -> tuple[Optional[str], Optional[str]]:
        """从路径中提取目标信息"""
        import re
        
        parts = path.strip("/").split("/")
        
        # 查找资源类型
        resource_types = ["users", "roles", "artifacts", "approvals", "learning-units"]
        
        target_type = None
        target_id = None
        
        for i, part in enumerate(parts):
            if part in resource_types:
                target_type = part.rstrip("s")  # users -> user
                # 检查下一部分是否是 ID
                if i + 1 < len(parts):
                    potential_id = parts[i + 1]
                    # 检查是否是 UUID 或数字
                    if re.match(r'^[0-9a-f-]{36}$', potential_id) or potential_id.isdigit():
                        target_id = potential_id
                break
        
        return target_type, target_id
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端 IP"""
        # 检查代理头
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _compute_hash(self, **data) -> str:
        """计算哈希"""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

