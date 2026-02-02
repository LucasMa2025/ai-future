"""
操作日志服务
"""
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc
import csv
import io

from ..models.audit import OperationLog


class OperationLogService:
    """操作日志服务"""
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== 查询操作 ====================
    
    def get_logs(
        self,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        method: Optional[str] = None,
        path: Optional[str] = None,
        function_code: Optional[str] = None,
        is_success: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        查询操作日志
        """
        query = self.db.query(OperationLog)
        
        # 筛选条件
        if user_id:
            query = query.filter(OperationLog.user_id == user_id)
        if username:
            query = query.filter(OperationLog.username.ilike(f"%{username}%"))
        if method:
            query = query.filter(OperationLog.method == method)
        if path:
            query = query.filter(OperationLog.path.ilike(f"%{path}%"))
        if function_code:
            query = query.filter(OperationLog.function_code == function_code)
        if is_success is not None:
            query = query.filter(OperationLog.is_success == (1 if is_success else 0))
        if start_date:
            query = query.filter(OperationLog.created_at >= start_date)
        if end_date:
            query = query.filter(OperationLog.created_at <= end_date)
        if ip_address:
            query = query.filter(OperationLog.ip_address.ilike(f"%{ip_address}%"))
        
        # 总数
        total = query.count()
        
        # 分页
        offset = (page - 1) * page_size
        logs = query.order_by(desc(OperationLog.created_at)).offset(offset).limit(page_size).all()
        
        return {
            "items": [self._log_to_dict(log) for log in logs],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    
    def get_log_by_id(self, log_id: int) -> Optional[OperationLog]:
        """获取日志详情"""
        return self.db.query(OperationLog).filter(
            OperationLog.id == log_id
        ).first()
    
    def _log_to_dict(self, log: OperationLog) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": log.id,
            "user_id": str(log.user_id) if log.user_id else None,
            "username": log.username,
            "request_id": log.request_id,
            "method": log.method,
            "path": log.path,
            "query_params": log.query_params,
            "ip_address": log.ip_address,
            "user_agent": log.user_agent,
            "status_code": log.status_code,
            "response_time_ms": log.response_time_ms,
            "function_code": log.function_code,
            "is_success": log.is_success == 1,
            "error_message": log.error_message,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
    
    # ==================== 创建操作 ====================
    
    def create_log(
        self,
        method: str,
        path: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        request_id: Optional[str] = None,
        query_params: Optional[Dict] = None,
        request_body: Optional[Dict] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        function_code: Optional[str] = None,
        is_success: bool = True,
        error_message: Optional[str] = None,
    ) -> OperationLog:
        """创建操作日志"""
        # 脱敏处理敏感字段
        if request_body:
            request_body = self._sanitize_request_body(request_body)
        
        log = OperationLog(
            user_id=user_id,
            username=username,
            request_id=request_id,
            method=method,
            path=path,
            query_params=query_params,
            request_body=request_body,
            ip_address=ip_address,
            user_agent=user_agent[:500] if user_agent else None,
            status_code=status_code,
            response_time_ms=response_time_ms,
            function_code=function_code,
            is_success=1 if is_success else 0,
            error_message=error_message,
        )
        
        self.db.add(log)
        self.db.commit()
        
        return log
    
    def _sanitize_request_body(self, body: Dict) -> Dict:
        """脱敏请求体中的敏感信息"""
        sensitive_keys = {"password", "token", "secret", "api_key", "authorization"}
        
        def sanitize(obj):
            if isinstance(obj, dict):
                return {
                    k: "***" if k.lower() in sensitive_keys else sanitize(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            return obj
        
        return sanitize(body)
    
    # ==================== 统计操作 ====================
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """获取操作日志统计"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=7)
        if not end_date:
            end_date = datetime.utcnow()
        
        query = self.db.query(OperationLog).filter(
            and_(
                OperationLog.created_at >= start_date,
                OperationLog.created_at <= end_date
            )
        )
        
        # 总请求数
        total_requests = query.count()
        
        # 成功/失败数
        success_count = query.filter(OperationLog.is_success == 1).count()
        failure_count = total_requests - success_count
        
        # 按方法统计
        by_method = self.db.query(
            OperationLog.method,
            func.count(OperationLog.id).label("count")
        ).filter(
            and_(
                OperationLog.created_at >= start_date,
                OperationLog.created_at <= end_date
            )
        ).group_by(OperationLog.method).all()
        
        # 按用户统计 top 10
        by_user = self.db.query(
            OperationLog.username,
            func.count(OperationLog.id).label("count")
        ).filter(
            and_(
                OperationLog.created_at >= start_date,
                OperationLog.created_at <= end_date,
                OperationLog.username.isnot(None)
            )
        ).group_by(OperationLog.username).order_by(
            desc(func.count(OperationLog.id))
        ).limit(10).all()
        
        # 按天统计
        by_day = self.db.query(
            func.date(OperationLog.created_at).label("date"),
            func.count(OperationLog.id).label("count")
        ).filter(
            and_(
                OperationLog.created_at >= start_date,
                OperationLog.created_at <= end_date
            )
        ).group_by(func.date(OperationLog.created_at)).order_by(
            func.date(OperationLog.created_at)
        ).all()
        
        # 平均响应时间
        avg_response_time = self.db.query(
            func.avg(OperationLog.response_time_ms)
        ).filter(
            and_(
                OperationLog.created_at >= start_date,
                OperationLog.created_at <= end_date,
                OperationLog.response_time_ms.isnot(None)
            )
        ).scalar()
        
        return {
            "total_requests": total_requests,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": round(success_count / total_requests * 100, 2) if total_requests > 0 else 0,
            "avg_response_time_ms": round(avg_response_time, 2) if avg_response_time else 0,
            "by_method": {m: c for m, c in by_method},
            "by_user": [{"username": u, "count": c} for u, c in by_user],
            "by_day": [{"date": str(d), "count": c} for d, c in by_day],
        }
    
    # ==================== 导出操作 ====================
    
    def export_to_csv(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        **filters
    ) -> str:
        """导出日志为CSV"""
        result = self.get_logs(
            page=1,
            page_size=10000,
            start_date=start_date,
            end_date=end_date,
            **filters
        )
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # 写入表头
        headers = [
            "ID", "用户名", "请求方法", "请求路径", "IP地址",
            "状态码", "响应时间(ms)", "功能代码", "是否成功", "错误信息", "创建时间"
        ]
        writer.writerow(headers)
        
        # 写入数据
        for log in result["items"]:
            writer.writerow([
                log["id"],
                log["username"] or "",
                log["method"] or "",
                log["path"] or "",
                log["ip_address"] or "",
                log["status_code"] or "",
                log["response_time_ms"] or "",
                log["function_code"] or "",
                "成功" if log["is_success"] else "失败",
                log["error_message"] or "",
                log["created_at"] or "",
            ])
        
        return output.getvalue()
    
    # ==================== 清理操作 ====================
    
    def cleanup_old_logs(self, days: int = 90) -> int:
        """清理指定天数前的日志"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = self.db.query(OperationLog).filter(
            OperationLog.created_at < cutoff_date
        ).delete()
        
        self.db.commit()
        
        return result
