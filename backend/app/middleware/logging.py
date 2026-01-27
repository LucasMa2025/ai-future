"""
用户操作日志中间件

实现:
1. 记录所有用户请求
2. 记录响应时间
3. 脱敏处理敏感数据
"""
import json
import time
import uuid
from datetime import datetime
from typing import Callable, Optional, Dict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..db.session import SessionLocal
from ..models.audit import OperationLog


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    用户操作日志中间件
    
    记录所有请求的详细信息，用于问题追溯和行为分析
    """
    
    def __init__(self, app: ASGIApp, exclude_paths: list = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/ws",
        ]
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """处理请求"""
        # 跳过不需要记录的路径
        if self._should_skip(request.url.path):
            return await call_next(request)
        
        # 生成请求 ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录开始时间
        start_time = time.time()
        
        # 获取请求体
        request_body = await self._get_request_body(request)
        
        # 执行请求
        response = await call_next(request)
        
        # 计算响应时间
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # 异步记录日志（不阻塞响应）
        await self._log_request(
            request=request,
            response=response,
            request_id=request_id,
            request_body=request_body,
            response_time_ms=response_time_ms,
        )
        
        # 添加请求 ID 到响应头
        response.headers["X-Request-ID"] = request_id
        
        return response
    
    def _should_skip(self, path: str) -> bool:
        """判断是否跳过日志"""
        return any(path.startswith(p) for p in self.exclude_paths)
    
    async def _get_request_body(self, request: Request) -> Optional[Dict]:
        """获取并脱敏请求体"""
        if request.method not in ("POST", "PUT", "PATCH"):
            return None
        
        try:
            body = await request.body()
            if not body:
                return None
            
            data = json.loads(body.decode())
            return self._sanitize_body(data)
        except:
            return None
    
    def _sanitize_body(self, data: Dict) -> Dict:
        """脱敏请求体"""
        sensitive_keys = {
            "password", "old_password", "new_password", "confirm_password",
            "token", "access_token", "refresh_token",
            "secret", "api_key", "private_key",
            "credit_card", "card_number", "cvv",
        }
        
        def sanitize(obj):
            if isinstance(obj, dict):
                return {
                    k: "***" if k.lower() in sensitive_keys else sanitize(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            return obj
        
        return sanitize(data)
    
    async def _log_request(
        self,
        request: Request,
        response: Response,
        request_id: str,
        request_body: Optional[Dict],
        response_time_ms: int,
    ) -> None:
        """记录请求日志"""
        db = SessionLocal()
        try:
            # 获取用户信息
            user_id = None
            username = None
            if hasattr(request.state, "user") and request.state.user:
                user_id = request.state.user.id
                username = request.state.user.username
            
            # 获取功能代码
            function_code = self._get_function_code(request.method, request.url.path)
            
            # 创建日志
            log = OperationLog(
                user_id=user_id,
                username=username,
                request_id=request_id,
                method=request.method,
                path=str(request.url.path),
                query_params=dict(request.query_params) or None,
                request_body=request_body,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("User-Agent", "")[:500],
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                function_code=function_code,
                is_success=1 if response.status_code < 400 else 0,
            )
            
            db.add(log)
            db.commit()
            
        except Exception as e:
            db.rollback()
            import logging
            logging.getLogger(__name__).warning(f"Failed to log request: {e}")
        finally:
            db.close()
    
    def _get_function_code(self, method: str, path: str) -> Optional[str]:
        """根据请求生成功能代码"""
        # 从路径提取模块
        parts = path.strip("/").split("/")
        
        if len(parts) < 3:
            return None
        
        # /api/v1/users -> users
        module = parts[2] if len(parts) > 2 else None
        
        if not module:
            return None
        
        # 确定操作类型
        action_map = {
            "GET": "list" if len(parts) == 3 else "get",
            "POST": "create",
            "PUT": "update",
            "PATCH": "update",
            "DELETE": "delete",
        }
        action = action_map.get(method, "unknown")
        
        return f"{module}.{action}"
    
    def _get_client_ip(self, request: Request) -> str:
        """获取客户端 IP"""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

