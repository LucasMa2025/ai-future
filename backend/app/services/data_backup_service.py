"""
数据备份服务

管理 NLGSM 系统数据库的备份和恢复
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text, desc
import os
import subprocess
import gzip
import json
import threading

from ..models.permission import DataBackup
from ..config import settings


class DataBackupService:
    """
    数据备份服务
    
    支持:
    - 完整数据库备份
    - 指定表备份
    - 备份压缩
    - 备份恢复
    """
    
    # 可备份的核心表
    BACKUP_TABLES = [
        "users",
        "roles",
        "user_roles",
        "permissions",
        "role_permissions",
        "user_permissions",
        "system_functions",
        "system_states",
        "state_transitions",
        "learning_units",
        "lu_constraints",
        "exploration_steps",
        "learning_sessions",
        "checkpoints",
        "business_audit_logs",
        "operation_logs",
        "notifications",
        "system_configs",
        "data_backups",
    ]
    
    def __init__(self, db: Session):
        self.db = db
        self.backup_dir = getattr(settings, "BACKUP_DIR", "./backups")
        
        # 确保备份目录存在
        os.makedirs(self.backup_dir, exist_ok=True)
    
    # ==================== 查询操作 ====================
    
    def get_backups(
        self,
        page: int = 1,
        page_size: int = 20,
        status: Optional[str] = None,
        backup_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取备份列表"""
        query = self.db.query(DataBackup)
        
        if status:
            query = query.filter(DataBackup.status == status)
        if backup_type:
            query = query.filter(DataBackup.backup_type == backup_type)
        
        total = query.count()
        
        offset = (page - 1) * page_size
        backups = query.order_by(desc(DataBackup.created_at)).offset(offset).limit(page_size).all()
        
        return {
            "items": [self._backup_to_dict(b) for b in backups],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size
        }
    
    def get_backup_by_id(self, backup_id: int) -> Optional[DataBackup]:
        """获取备份详情"""
        return self.db.query(DataBackup).filter(
            DataBackup.id == backup_id
        ).first()
    
    def _backup_to_dict(self, backup: DataBackup) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": backup.id,
            "backup_name": backup.backup_name,
            "backup_type": backup.backup_type,
            "tables": backup.tables,
            "file_path": backup.file_path,
            "file_size": backup.file_size,
            "file_size_display": self._format_file_size(backup.file_size),
            "compressed": backup.compressed,
            "status": backup.status,
            "progress": backup.progress,
            "started_at": backup.started_at.isoformat() if backup.started_at else None,
            "completed_at": backup.completed_at.isoformat() if backup.completed_at else None,
            "duration_seconds": backup.duration_seconds,
            "created_by": backup.created_by,
            "error_message": backup.error_message,
            "record_counts": backup.record_counts,
            "description": backup.description,
            "created_at": backup.created_at.isoformat() if backup.created_at else None,
        }
    
    def _format_file_size(self, size: int) -> str:
        """格式化文件大小"""
        if not size:
            return "0 B"
        
        units = ["B", "KB", "MB", "GB"]
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.2f} {units[unit_index]}"
    
    # ==================== 创建备份 ====================
    
    def create_backup(
        self,
        backup_name: str,
        backup_type: str = "full",
        tables: Optional[List[str]] = None,
        compress: bool = True,
        description: Optional[str] = None,
        created_by: Optional[str] = None,
        async_mode: bool = True,
    ) -> DataBackup:
        """
        创建数据备份
        
        Args:
            backup_name: 备份名称
            backup_type: 备份类型 (full/tables)
            tables: 要备份的表列表（backup_type=tables时使用）
            compress: 是否压缩
            description: 备份说明
            created_by: 创建人ID
            async_mode: 是否异步执行
        """
        # 确定要备份的表
        if backup_type == "full":
            backup_tables = self.BACKUP_TABLES
        else:
            backup_tables = tables or []
            # 验证表名
            for table in backup_tables:
                if table not in self.BACKUP_TABLES:
                    raise ValueError(f"不支持备份表: {table}")
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"nlgsm_backup_{timestamp}"
        if compress:
            file_name += ".sql.gz"
        else:
            file_name += ".sql"
        
        file_path = os.path.join(self.backup_dir, file_name)
        
        # 创建备份记录
        backup = DataBackup(
            backup_name=backup_name,
            backup_type=backup_type,
            tables=backup_tables,
            file_path=file_path,
            compressed=compress,
            status="pending",
            progress=0,
            created_by=created_by,
            description=description,
        )
        
        self.db.add(backup)
        self.db.commit()
        self.db.refresh(backup)
        
        # 执行备份
        if async_mode:
            thread = threading.Thread(
                target=self._execute_backup,
                args=(backup.id, backup_tables, file_path, compress)
            )
            thread.start()
        else:
            self._execute_backup(backup.id, backup_tables, file_path, compress)
        
        return backup
    
    def _execute_backup(
        self,
        backup_id: int,
        tables: List[str],
        file_path: str,
        compress: bool
    ):
        """执行备份（同步）"""
        from ..db.session import SessionLocal
        
        db = SessionLocal()
        try:
            backup = db.query(DataBackup).filter(DataBackup.id == backup_id).first()
            if not backup:
                return
            
            backup.status = "running"
            backup.started_at = datetime.utcnow()
            db.commit()
            
            record_counts = {}
            total_tables = len(tables)
            
            # 收集数据
            all_data = {}
            for i, table in enumerate(tables):
                try:
                    # 获取表数据
                    result = db.execute(text(f"SELECT * FROM {table}"))
                    rows = result.fetchall()
                    columns = result.keys()
                    
                    all_data[table] = {
                        "columns": list(columns),
                        "rows": [dict(zip(columns, row)) for row in rows]
                    }
                    record_counts[table] = len(rows)
                    
                    # 更新进度
                    backup.progress = int((i + 1) / total_tables * 90)
                    db.commit()
                    
                except Exception as e:
                    all_data[table] = {"error": str(e)}
                    record_counts[table] = 0
            
            # 转换为 JSON
            backup_content = json.dumps(all_data, default=str, ensure_ascii=False, indent=2)
            
            # 写入文件
            if compress:
                with gzip.open(file_path, "wt", encoding="utf-8") as f:
                    f.write(backup_content)
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(backup_content)
            
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            
            # 更新备份记录
            backup.status = "completed"
            backup.progress = 100
            backup.completed_at = datetime.utcnow()
            backup.duration_seconds = int(
                (backup.completed_at - backup.started_at).total_seconds()
            )
            backup.file_size = file_size
            backup.record_counts = record_counts
            
            db.commit()
            
        except Exception as e:
            backup = db.query(DataBackup).filter(DataBackup.id == backup_id).first()
            if backup:
                backup.status = "failed"
                backup.error_message = str(e)
                backup.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()
    
    # ==================== 恢复备份 ====================
    
    def restore_backup(
        self,
        backup_id: int,
        tables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        恢复备份
        
        Args:
            backup_id: 备份ID
            tables: 要恢复的表列表（不指定则恢复全部）
        
        Returns:
            恢复结果
        """
        backup = self.get_backup_by_id(backup_id)
        if not backup:
            raise ValueError("备份不存在")
        
        if backup.status != "completed":
            raise ValueError("只能恢复已完成的备份")
        
        if not os.path.exists(backup.file_path):
            raise ValueError("备份文件不存在")
        
        # 读取备份文件
        if backup.compressed:
            with gzip.open(backup.file_path, "rt", encoding="utf-8") as f:
                backup_data = json.load(f)
        else:
            with open(backup.file_path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)
        
        # 确定要恢复的表
        restore_tables = tables or list(backup_data.keys())
        
        results = {}
        
        for table in restore_tables:
            if table not in backup_data:
                results[table] = {"success": False, "error": "备份中不包含此表"}
                continue
            
            table_data = backup_data[table]
            
            if "error" in table_data:
                results[table] = {"success": False, "error": table_data["error"]}
                continue
            
            try:
                # 清空表
                self.db.execute(text(f"DELETE FROM {table}"))
                
                # 插入数据
                rows = table_data.get("rows", [])
                columns = table_data.get("columns", [])
                
                if rows and columns:
                    for row in rows:
                        cols = ", ".join(columns)
                        placeholders = ", ".join([f":{col}" for col in columns])
                        sql = f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"
                        self.db.execute(text(sql), row)
                
                self.db.commit()
                results[table] = {"success": True, "records": len(rows)}
                
            except Exception as e:
                self.db.rollback()
                results[table] = {"success": False, "error": str(e)}
        
        return {
            "backup_id": backup_id,
            "results": results,
            "restored_at": datetime.utcnow().isoformat()
        }
    
    # ==================== 删除备份 ====================
    
    def delete_backup(self, backup_id: int) -> bool:
        """删除备份"""
        backup = self.get_backup_by_id(backup_id)
        if not backup:
            return False
        
        # 删除文件
        if backup.file_path and os.path.exists(backup.file_path):
            try:
                os.remove(backup.file_path)
            except:
                pass
        
        # 删除记录
        self.db.delete(backup)
        self.db.commit()
        
        return True
    
    # ==================== 统计信息 ====================
    
    def get_backup_stats(self) -> Dict[str, Any]:
        """获取备份统计信息"""
        total_backups = self.db.query(DataBackup).count()
        
        completed_backups = self.db.query(DataBackup).filter(
            DataBackup.status == "completed"
        ).count()
        
        failed_backups = self.db.query(DataBackup).filter(
            DataBackup.status == "failed"
        ).count()
        
        # 总存储大小
        from sqlalchemy import func
        total_size = self.db.query(func.sum(DataBackup.file_size)).filter(
            DataBackup.status == "completed"
        ).scalar() or 0
        
        # 最近备份
        latest_backup = self.db.query(DataBackup).filter(
            DataBackup.status == "completed"
        ).order_by(desc(DataBackup.completed_at)).first()
        
        return {
            "total_backups": total_backups,
            "completed_backups": completed_backups,
            "failed_backups": failed_backups,
            "total_storage_size": total_size,
            "total_storage_display": self._format_file_size(total_size),
            "latest_backup": self._backup_to_dict(latest_backup) if latest_backup else None,
            "available_tables": self.BACKUP_TABLES,
        }
    
    def get_table_info(self) -> List[Dict[str, Any]]:
        """获取可备份表的信息"""
        result = []
        
        for table in self.BACKUP_TABLES:
            try:
                count_result = self.db.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                )
                count = count_result.scalar()
                
                result.append({
                    "table_name": table,
                    "record_count": count,
                })
            except:
                result.append({
                    "table_name": table,
                    "record_count": 0,
                    "error": "无法获取表信息"
                })
        
        return result
