"""
审计服务

实现:
1. 业务审计日志查询
2. 操作日志查询
3. 审计日志完整性验证
"""
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from ..models.audit import BusinessAuditLog, OperationLog


class AuditService:
    """
    审计服务
    
    处理审计日志的查询和验证
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    # ==================== 业务审计日志 ====================
    
    def get_audit_logs(
        self,
        event_type: Optional[str] = None,
        event_category: Optional[str] = None,
        actor_id: Optional[UUID] = None,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        audit_level: Optional[str] = None,
        result: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[BusinessAuditLog], int]:
        """
        查询业务审计日志
        
        Returns:
            (日志列表, 总数)
        """
        query = self.db.query(BusinessAuditLog)
        
        # 应用过滤条件
        if event_type:
            query = query.filter(BusinessAuditLog.event_type == event_type)
        
        if event_category:
            query = query.filter(BusinessAuditLog.event_category == event_category)
        
        if actor_id:
            query = query.filter(BusinessAuditLog.actor_id == actor_id)
        
        if target_type:
            query = query.filter(BusinessAuditLog.target_type == target_type)
        
        if target_id:
            query = query.filter(BusinessAuditLog.target_id == target_id)
        
        if audit_level:
            query = query.filter(BusinessAuditLog.audit_level == audit_level)
        
        if result:
            query = query.filter(BusinessAuditLog.result == result)
        
        if start_date:
            query = query.filter(BusinessAuditLog.created_at >= start_date)
        
        if end_date:
            query = query.filter(BusinessAuditLog.created_at <= end_date)
        
        total = query.count()
        logs = query.order_by(
            BusinessAuditLog.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return logs, total
    
    def get_audit_log_by_id(self, entry_id: str) -> Optional[BusinessAuditLog]:
        """根据ID获取审计日志"""
        return self.db.query(BusinessAuditLog).filter(
            BusinessAuditLog.entry_id == entry_id
        ).first()
    
    def verify_audit_chain(
        self,
        start_id: Optional[int] = None,
        end_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        验证审计日志链的完整性
        
        Returns:
            验证结果
        """
        query = self.db.query(BusinessAuditLog).order_by(BusinessAuditLog.id)
        
        if start_id:
            query = query.filter(BusinessAuditLog.id >= start_id)
        if end_id:
            query = query.filter(BusinessAuditLog.id <= end_id)
        
        logs = query.all()
        
        if not logs:
            return {
                "valid": True,
                "checked_count": 0,
                "errors": []
            }
        
        errors = []
        previous_hash = "GENESIS"
        
        for log in logs:
            # 验证前一个哈希
            if log.previous_hash != previous_hash:
                errors.append({
                    "entry_id": log.entry_id,
                    "error": "Previous hash mismatch",
                    "expected": previous_hash,
                    "actual": log.previous_hash
                })
            
            # 验证当前哈希
            computed_hash = self._compute_entry_hash(log)
            if log.entry_hash != computed_hash:
                errors.append({
                    "entry_id": log.entry_id,
                    "error": "Entry hash mismatch",
                    "expected": computed_hash,
                    "actual": log.entry_hash
                })
            
            previous_hash = log.entry_hash
        
        return {
            "valid": len(errors) == 0,
            "checked_count": len(logs),
            "errors": errors
        }
    
    def get_audit_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取审计统计信息"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        base_query = self.db.query(BusinessAuditLog).filter(
            and_(
                BusinessAuditLog.created_at >= start_date,
                BusinessAuditLog.created_at <= end_date
            )
        )
        
        # 总数
        total = base_query.count()
        
        # 按事件类型统计
        by_event_type = dict(
            base_query.with_entities(
                BusinessAuditLog.event_type,
                func.count(BusinessAuditLog.id)
            ).group_by(BusinessAuditLog.event_type).all()
        )
        
        # 按审计级别统计
        by_audit_level = dict(
            base_query.with_entities(
                BusinessAuditLog.audit_level,
                func.count(BusinessAuditLog.id)
            ).group_by(BusinessAuditLog.audit_level).all()
        )
        
        # 按结果统计
        by_result = dict(
            base_query.with_entities(
                BusinessAuditLog.result,
                func.count(BusinessAuditLog.id)
            ).group_by(BusinessAuditLog.result).all()
        )
        
        return {
            "total": total,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "by_event_type": by_event_type,
            "by_audit_level": by_audit_level,
            "by_result": by_result
        }
    
    def export_audit_logs(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> str:
        """导出审计日志"""
        logs, _ = self.get_audit_logs(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        if format == "json":
            return json.dumps(
                [self._log_to_dict(log) for log in logs],
                indent=2,
                default=str
            )
        
        # CSV 格式
        lines = ["entry_id,event_type,actor_name,action,target_type,target_id,result,audit_level,created_at"]
        for log in logs:
            lines.append(
                f'"{log.entry_id}","{log.event_type}","{log.actor_name or ""}","{log.action}",'
                f'"{log.target_type or ""}","{log.target_id or ""}","{log.result}","{log.audit_level}",'
                f'"{log.created_at.isoformat()}"'
            )
        
        return "\n".join(lines)
    
    # ==================== 操作日志 ====================
    
    def get_operation_logs(
        self,
        user_id: Optional[UUID] = None,
        method: Optional[str] = None,
        path_pattern: Optional[str] = None,
        function_code: Optional[str] = None,
        is_success: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[OperationLog], int]:
        """
        查询操作日志
        
        Returns:
            (日志列表, 总数)
        """
        query = self.db.query(OperationLog)
        
        if user_id:
            query = query.filter(OperationLog.user_id == user_id)
        
        if method:
            query = query.filter(OperationLog.method == method)
        
        if path_pattern:
            query = query.filter(OperationLog.path.ilike(f"%{path_pattern}%"))
        
        if function_code:
            query = query.filter(OperationLog.function_code == function_code)
        
        if is_success is not None:
            query = query.filter(OperationLog.is_success == (1 if is_success else 0))
        
        if start_date:
            query = query.filter(OperationLog.created_at >= start_date)
        
        if end_date:
            query = query.filter(OperationLog.created_at <= end_date)
        
        total = query.count()
        logs = query.order_by(
            OperationLog.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        return logs, total
    
    def get_user_activity(
        self,
        user_id: UUID,
        days: int = 7
    ) -> Dict[str, Any]:
        """获取用户活动统计"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        query = self.db.query(OperationLog).filter(
            OperationLog.user_id == user_id,
            OperationLog.created_at >= start_date
        )
        
        # 请求总数
        total_requests = query.count()
        
        # 成功率
        success_count = query.filter(OperationLog.is_success == 1).count()
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        
        # 平均响应时间
        avg_response_time = self.db.query(
            func.avg(OperationLog.response_time_ms)
        ).filter(
            OperationLog.user_id == user_id,
            OperationLog.created_at >= start_date
        ).scalar() or 0
        
        # 最常访问的功能
        top_functions = self.db.query(
            OperationLog.function_code,
            func.count(OperationLog.id).label("count")
        ).filter(
            OperationLog.user_id == user_id,
            OperationLog.created_at >= start_date,
            OperationLog.function_code.isnot(None)
        ).group_by(
            OperationLog.function_code
        ).order_by(
            func.count(OperationLog.id).desc()
        ).limit(10).all()
        
        return {
            "user_id": str(user_id),
            "period_days": days,
            "total_requests": total_requests,
            "success_rate": round(success_rate, 2),
            "avg_response_time_ms": round(float(avg_response_time), 2),
            "top_functions": [
                {"function": f[0], "count": f[1]}
                for f in top_functions if f[0]
            ]
        }
    
    # ==================== 私有方法 ====================
    
    def _compute_entry_hash(self, log: BusinessAuditLog) -> str:
        """计算审计日志条目哈希"""
        content = json.dumps({
            "entry_id": log.entry_id,
            "previous_hash": log.previous_hash,
            "event_type": log.event_type,
            "action": log.action,
            "actor_id": str(log.actor_id) if log.actor_id else None,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }, sort_keys=True, default=str)
        
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _log_to_dict(self, log: BusinessAuditLog) -> Dict[str, Any]:
        """转换审计日志为字典"""
        return {
            "entry_id": log.entry_id,
            "event_type": log.event_type,
            "event_category": log.event_category,
            "actor_id": str(log.actor_id) if log.actor_id else None,
            "actor_name": log.actor_name,
            "actor_ip": log.actor_ip,
            "action": log.action,
            "target_type": log.target_type,
            "target_id": log.target_id,
            "details": log.details,
            "result": log.result,
            "audit_level": log.audit_level,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }

