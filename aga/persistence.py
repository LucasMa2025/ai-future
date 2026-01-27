"""
AGA 数据持久化模块 v2.0

支持将 AGA 状态持久化到数据库，避免停机失效。

v2.0 优化：
- 元数据外置架构：DB 管理 lifecycle/LU，AGA 只是执行层
- 增量同步：只同步变更的槽位
- 批量操作优化：减少数据库往返
- 活跃槽位预加载：启动时只加载活跃槽位

Demo 使用 SQLite，生产环境可替换为 PostgreSQL/Redis。
"""
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging
from contextlib import contextmanager
from dataclasses import dataclass

from .core import (
    AuxiliaryGovernedAttention, 
    AGAManager, 
    LifecycleState, 
    KnowledgeSlotInfo,
    AGAConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeRecord:
    """知识记录（数据库层面）"""
    slot_idx: int
    lu_id: str
    condition: str
    decision: str
    key_vector: List[float]
    value_vector: List[float]
    lifecycle_state: str
    hit_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'slot_idx': self.slot_idx,
            'lu_id': self.lu_id,
            'condition': self.condition,
            'decision': self.decision,
            'key_vector': self.key_vector,
            'value_vector': self.value_vector,
            'lifecycle_state': self.lifecycle_state,
            'hit_count': self.hit_count,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
        }


class AGAPersistence(ABC):
    """
    AGA 持久化抽象基类
    
    v2.0: 增强为"治理层"角色，AGA 模块只是执行层
    
    生产环境可实现 PostgreSQL、Redis 等版本
    """
    
    @abstractmethod
    def save_aga_config(self, aga_id: str, config: Dict[str, Any]) -> bool:
        """保存 AGA 配置"""
        pass
    
    @abstractmethod
    def load_aga_config(self, aga_id: str) -> Optional[Dict[str, Any]]:
        """加载 AGA 配置"""
        pass
    
    @abstractmethod
    def save_knowledge(
        self,
        aga_id: str,
        slot_idx: int,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        lifecycle_state: str,
    ) -> bool:
        """保存单条知识"""
        pass
    
    @abstractmethod
    def save_knowledge_batch(
        self,
        aga_id: str,
        records: List[KnowledgeRecord],
    ) -> int:
        """批量保存知识（返回成功数量）"""
        pass
    
    @abstractmethod
    def load_active_knowledge(self, aga_id: str) -> List[Dict[str, Any]]:
        """仅加载活跃知识（非 QUARANTINED）"""
        pass
    
    @abstractmethod
    def load_all_knowledge(self, aga_id: str) -> List[Dict[str, Any]]:
        """加载所有知识"""
        pass
    
    @abstractmethod
    def update_lifecycle(self, aga_id: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期状态"""
        pass
    
    @abstractmethod
    def update_lifecycle_batch(
        self, 
        aga_id: str, 
        updates: List[tuple]  # [(lu_id, new_state), ...]
    ) -> int:
        """批量更新生命周期"""
        pass
    
    @abstractmethod
    def delete_knowledge(self, aga_id: str, lu_id: str) -> bool:
        """删除知识"""
        pass
    
    @abstractmethod
    def get_knowledge_by_lu_id(self, aga_id: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """按 LU ID 获取知识"""
        pass
    
    @abstractmethod
    def get_knowledge_count(self, aga_id: str, state: Optional[str] = None) -> int:
        """获取知识数量"""
        pass
    
    @abstractmethod
    def increment_hit_count(self, aga_id: str, lu_ids: List[str]) -> bool:
        """批量增加命中计数"""
        pass


class SQLitePersistence(AGAPersistence):
    """
    SQLite 持久化实现 v2.0
    
    优化：
    - 连接池管理
    - 批量操作
    - 索引优化
    - 事务支持
    """
    
    def __init__(self, db_path: str = "aga_data.db"):
        """
        初始化 SQLite 持久化
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表"""
        with self._get_conn() as conn:
            cursor = conn.cursor()
            
            # AGA 配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS aga_configs (
                    aga_id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # 知识表（优化索引）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS knowledge (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    aga_id TEXT NOT NULL,
                    slot_idx INTEGER NOT NULL,
                    lu_id TEXT NOT NULL,
                    condition TEXT,
                    decision TEXT,
                    key_vector TEXT NOT NULL,
                    value_vector TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(aga_id, lu_id)
                )
            ''')
            
            # 审计日志表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    aga_id TEXT NOT NULL,
                    lu_id TEXT,
                    action TEXT NOT NULL,
                    old_state TEXT,
                    new_state TEXT,
                    details TEXT,
                    timestamp TEXT NOT NULL
                )
            ''')
            
            # v2.0: 同步状态表（用于增量同步）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_state (
                    aga_id TEXT PRIMARY KEY,
                    last_sync_at TEXT NOT NULL,
                    dirty_slots TEXT,
                    version INTEGER DEFAULT 0
                )
            ''')
            
            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_aga_id 
                ON knowledge(aga_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_lu_id 
                ON knowledge(lu_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_knowledge_state 
                ON knowledge(aga_id, lifecycle_state)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_aga_id 
                ON audit_log(aga_id)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_log(timestamp)
            ''')
            
            conn.commit()
        
        logger.info(f"SQLite database initialized: {self.db_path}")
    
    @contextmanager
    def _get_conn(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def save_aga_config(self, aga_id: str, config: Dict[str, Any]) -> bool:
        """保存 AGA 配置"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                config_json = json.dumps(config)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO aga_configs (aga_id, config, created_at, updated_at)
                    VALUES (?, ?, COALESCE((SELECT created_at FROM aga_configs WHERE aga_id = ?), ?), ?)
                ''', (aga_id, config_json, aga_id, now, now))
                
                conn.commit()
            
            self._log_audit(aga_id, None, "save_config", None, None, "Config saved")
            return True
        except Exception as e:
            logger.error(f"Failed to save AGA config: {e}")
            return False
    
    def load_aga_config(self, aga_id: str) -> Optional[Dict[str, Any]]:
        """加载 AGA 配置"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM aga_configs WHERE aga_id = ?', (aga_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    'aga_id': row['aga_id'],
                    'config': json.loads(row['config']),
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at'],
                }
        except Exception as e:
            logger.error(f"Failed to load AGA config: {e}")
            return None
    
    def save_knowledge(
        self,
        aga_id: str,
        slot_idx: int,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        lifecycle_state: str,
    ) -> bool:
        """保存单条知识"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                key_json = json.dumps(key_vector)
                value_json = json.dumps(value_vector)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO knowledge 
                    (aga_id, slot_idx, lu_id, condition, decision, key_vector, value_vector, 
                     lifecycle_state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 
                            COALESCE((SELECT created_at FROM knowledge WHERE aga_id = ? AND lu_id = ?), ?), ?)
                ''', (aga_id, slot_idx, lu_id, condition, decision, key_json, value_json,
                      lifecycle_state, aga_id, lu_id, now, now))
                
                conn.commit()
            
            self._log_audit(aga_id, lu_id, "save_knowledge", None, lifecycle_state, 
                           f"slot={slot_idx}")
            return True
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
            return False
    
    def save_knowledge_batch(
        self,
        aga_id: str,
        records: List[KnowledgeRecord],
    ) -> int:
        """批量保存知识"""
        if not records:
            return 0
        
        success_count = 0
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                for record in records:
                    try:
                        key_json = json.dumps(record.key_vector)
                        value_json = json.dumps(record.value_vector)
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO knowledge 
                            (aga_id, slot_idx, lu_id, condition, decision, key_vector, value_vector, 
                             lifecycle_state, hit_count, created_at, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                                    COALESCE((SELECT created_at FROM knowledge WHERE aga_id = ? AND lu_id = ?), ?), ?)
                        ''', (aga_id, record.slot_idx, record.lu_id, record.condition, 
                              record.decision, key_json, value_json, record.lifecycle_state,
                              record.hit_count, aga_id, record.lu_id, now, now))
                        success_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to save record {record.lu_id}: {e}")
                
                conn.commit()
            
            self._log_audit(aga_id, None, "save_knowledge_batch", None, None, 
                           f"count={success_count}")
            return success_count
        except Exception as e:
            logger.error(f"Failed to save knowledge batch: {e}")
            return success_count
    
    def load_active_knowledge(self, aga_id: str) -> List[Dict[str, Any]]:
        """仅加载活跃知识（非 QUARANTINED）"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM knowledge 
                    WHERE aga_id = ? AND lifecycle_state != ?
                    ORDER BY slot_idx
                ''', (aga_id, LifecycleState.QUARANTINED.value))
                
                rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to load active knowledge: {e}")
            return []
    
    def load_all_knowledge(self, aga_id: str) -> List[Dict[str, Any]]:
        """加载所有知识"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM knowledge WHERE aga_id = ? 
                    ORDER BY slot_idx
                ''', (aga_id,))
                
                rows = cursor.fetchall()
            
            return [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to load knowledge: {e}")
            return []
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """转换行到字典"""
        return {
            'slot_idx': row['slot_idx'],
            'lu_id': row['lu_id'],
            'condition': row['condition'],
            'decision': row['decision'],
            'key_vector': json.loads(row['key_vector']),
            'value_vector': json.loads(row['value_vector']),
            'lifecycle_state': row['lifecycle_state'],
            'hit_count': row['hit_count'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
        }
    
    def update_lifecycle(self, aga_id: str, lu_id: str, new_state: str) -> bool:
        """更新生命周期状态"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                # 获取旧状态
                cursor.execute('''
                    SELECT lifecycle_state FROM knowledge 
                    WHERE aga_id = ? AND lu_id = ?
                ''', (aga_id, lu_id))
                row = cursor.fetchone()
                old_state = row['lifecycle_state'] if row else None
                
                # 更新
                now = datetime.now().isoformat()
                cursor.execute('''
                    UPDATE knowledge 
                    SET lifecycle_state = ?, updated_at = ?
                    WHERE aga_id = ? AND lu_id = ?
                ''', (new_state, now, aga_id, lu_id))
                
                conn.commit()
            
            self._log_audit(aga_id, lu_id, "update_lifecycle", old_state, new_state, None)
            return True
        except Exception as e:
            logger.error(f"Failed to update lifecycle: {e}")
            return False
    
    def update_lifecycle_batch(
        self, 
        aga_id: str, 
        updates: List[tuple]  # [(lu_id, new_state), ...]
    ) -> int:
        """批量更新生命周期"""
        if not updates:
            return 0
        
        success_count = 0
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                
                for lu_id, new_state in updates:
                    cursor.execute('''
                        UPDATE knowledge 
                        SET lifecycle_state = ?, updated_at = ?
                        WHERE aga_id = ? AND lu_id = ?
                    ''', (new_state, now, aga_id, lu_id))
                    success_count += cursor.rowcount
                
                conn.commit()
            
            self._log_audit(aga_id, None, "update_lifecycle_batch", None, None, 
                           f"count={success_count}")
            return success_count
        except Exception as e:
            logger.error(f"Failed to update lifecycle batch: {e}")
            return success_count
    
    def delete_knowledge(self, aga_id: str, lu_id: str) -> bool:
        """删除知识"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM knowledge WHERE aga_id = ? AND lu_id = ?
                ''', (aga_id, lu_id))
                
                conn.commit()
            
            self._log_audit(aga_id, lu_id, "delete_knowledge", None, None, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete knowledge: {e}")
            return False
    
    def get_knowledge_by_lu_id(self, aga_id: str, lu_id: str) -> Optional[Dict[str, Any]]:
        """按 LU ID 获取知识"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM knowledge WHERE aga_id = ? AND lu_id = ?
                ''', (aga_id, lu_id))
                
                row = cursor.fetchone()
            
            if not row:
                return None
            
            return self._row_to_dict(row)
        except Exception as e:
            logger.error(f"Failed to get knowledge: {e}")
            return None
    
    def get_knowledge_count(self, aga_id: str, state: Optional[str] = None) -> int:
        """获取知识数量"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                if state:
                    cursor.execute('''
                        SELECT COUNT(*) as count FROM knowledge 
                        WHERE aga_id = ? AND lifecycle_state = ?
                    ''', (aga_id, state))
                else:
                    cursor.execute('''
                        SELECT COUNT(*) as count FROM knowledge WHERE aga_id = ?
                    ''', (aga_id,))
                
                return cursor.fetchone()['count']
        except Exception as e:
            logger.error(f"Failed to get knowledge count: {e}")
            return 0
    
    def increment_hit_count(self, aga_id: str, lu_ids: List[str]) -> bool:
        """批量增加命中计数"""
        if not lu_ids:
            return True
        
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                placeholders = ','.join(['?'] * len(lu_ids))
                cursor.execute(f'''
                    UPDATE knowledge 
                    SET hit_count = hit_count + 1
                    WHERE aga_id = ? AND lu_id IN ({placeholders})
                ''', [aga_id] + lu_ids)
                
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to increment hit count: {e}")
            return False
    
    def get_audit_log(self, aga_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM audit_log 
                    WHERE aga_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (aga_id, limit))
                
                rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get audit log: {e}")
            return []
    
    def _log_audit(
        self,
        aga_id: str,
        lu_id: Optional[str],
        action: str,
        old_state: Optional[str],
        new_state: Optional[str],
        details: Optional[str],
    ):
        """记录审计日志"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO audit_log 
                    (aga_id, lu_id, action, old_state, new_state, details, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (aga_id, lu_id, action, old_state, new_state, details, now))
                
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to log audit: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) as count FROM aga_configs')
                aga_count = cursor.fetchone()['count']
                
                cursor.execute('SELECT COUNT(*) as count FROM knowledge')
                knowledge_count = cursor.fetchone()['count']
                
                cursor.execute('''
                    SELECT lifecycle_state, COUNT(*) as count 
                    FROM knowledge GROUP BY lifecycle_state
                ''')
                state_dist = {row['lifecycle_state']: row['count'] for row in cursor.fetchall()}
                
                cursor.execute('SELECT COUNT(*) as count FROM audit_log')
                audit_count = cursor.fetchone()['count']
            
            return {
                'aga_instances': aga_count,
                'total_knowledge': knowledge_count,
                'state_distribution': state_dist,
                'audit_entries': audit_count,
                'db_path': self.db_path,
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    # v2.0: 增量同步支持
    
    def mark_dirty(self, aga_id: str, slot_indices: List[int]):
        """标记脏槽位（需要同步）"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                # 获取现有脏槽位
                cursor.execute('''
                    SELECT dirty_slots FROM sync_state WHERE aga_id = ?
                ''', (aga_id,))
                row = cursor.fetchone()
                
                if row and row['dirty_slots']:
                    existing = set(json.loads(row['dirty_slots']))
                else:
                    existing = set()
                
                existing.update(slot_indices)
                
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT OR REPLACE INTO sync_state (aga_id, last_sync_at, dirty_slots, version)
                    VALUES (?, ?, ?, COALESCE((SELECT version FROM sync_state WHERE aga_id = ?), 0) + 1)
                ''', (aga_id, now, json.dumps(list(existing)), aga_id))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark dirty: {e}")
    
    def get_dirty_slots(self, aga_id: str) -> List[int]:
        """获取脏槽位列表"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT dirty_slots FROM sync_state WHERE aga_id = ?
                ''', (aga_id,))
                row = cursor.fetchone()
                
                if row and row['dirty_slots']:
                    return json.loads(row['dirty_slots'])
                return []
        except Exception as e:
            logger.error(f"Failed to get dirty slots: {e}")
            return []
    
    def clear_dirty_slots(self, aga_id: str):
        """清除脏槽位标记"""
        try:
            with self._get_conn() as conn:
                cursor = conn.cursor()
                
                now = datetime.now().isoformat()
                cursor.execute('''
                    UPDATE sync_state 
                    SET dirty_slots = '[]', last_sync_at = ?
                    WHERE aga_id = ?
                ''', (now, aga_id))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to clear dirty slots: {e}")


class AGAPersistenceManager:
    """
    AGA 持久化管理器 v2.0
    
    核心设计原则：
    - DB 是治理层：管理 lifecycle、LU 元数据、审计日志
    - AGA 是执行层：只负责推理时的知识检索和融合
    - 增量同步：只同步变更，减少 IO
    """
    
    def __init__(
        self, 
        persistence: AGAPersistence, 
        aga_id: str = "default",
        auto_sync: bool = True,  # 是否自动同步变更
    ):
        """
        初始化管理器
        
        Args:
            persistence: 持久化实现
            aga_id: AGA 实例 ID
            auto_sync: 是否自动同步变更到数据库
        """
        self.persistence = persistence
        self.aga_id = aga_id
        self.auto_sync = auto_sync
        self._dirty_slots: Set[int] = set()
    
    def save_aga(self, aga: AuxiliaryGovernedAttention) -> bool:
        """
        保存 AGA 到数据库（完整保存）
        """
        # 保存配置
        state = aga.export_state()
        if not self.persistence.save_aga_config(self.aga_id, state['config']):
            return False
        
        # 批量保存知识
        records = []
        for i in range(aga.num_slots):
            if aga.slot_lifecycle[i] != LifecycleState.QUARANTINED:
                records.append(KnowledgeRecord(
                    slot_idx=i,
                    lu_id=aga.slot_lu_ids[i] or f"slot_{i}",
                    condition=aga.slot_conditions[i] or "",
                    decision=aga.slot_decisions[i] or "",
                    key_vector=aga.aux_keys[i].cpu().tolist(),
                    value_vector=aga.aux_values[i].cpu().tolist(),
                    lifecycle_state=aga.slot_lifecycle[i].value,
                    hit_count=aga.slot_hit_counts[i],
                ))
        
        if records:
            self.persistence.save_knowledge_batch(self.aga_id, records)
        
        self._dirty_slots.clear()
        return True
    
    def load_aga(self, aga: AuxiliaryGovernedAttention) -> bool:
        """
        从数据库加载 AGA 状态（仅活跃槽位）
        
        v2.0: 只加载活跃知识，提高启动速度
        """
        import torch
        
        # 加载活跃知识
        knowledge_list = self.persistence.load_active_knowledge(self.aga_id)
        
        if not knowledge_list:
            logger.info(f"No active knowledge found for AGA {self.aga_id}")
            return False
        
        for k in knowledge_list:
            slot_idx = k['slot_idx']
            if slot_idx >= aga.num_slots:
                continue
            
            key_vector = torch.tensor(k['key_vector'])
            value_vector = torch.tensor(k['value_vector'])
            
            aga.inject_knowledge(
                slot_idx=slot_idx,
                key_vector=key_vector,
                value_vector=value_vector,
                lu_id=k['lu_id'],
                lifecycle_state=LifecycleState(k['lifecycle_state']),
                condition=k['condition'],
                decision=k['decision'],
            )
        
        logger.info(f"Loaded {len(knowledge_list)} active knowledge entries for AGA {self.aga_id}")
        return True
    
    def sync_knowledge(
        self,
        aga: AuxiliaryGovernedAttention,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector,
        value_vector,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
    ) -> Optional[int]:
        """
        同步注入知识（同时写入 AGA 和数据库）
        
        Returns:
            槽位索引，失败返回 None
        """
        slot_idx = aga.find_free_slot()
        if slot_idx is None:
            logger.error("No free slot available")
            return None
        
        # 写入 AGA
        aga.inject_knowledge(
            slot_idx=slot_idx,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id=lu_id,
            lifecycle_state=lifecycle_state,
            condition=condition,
            decision=decision,
        )
        
        # 写入数据库
        import torch
        if isinstance(key_vector, torch.Tensor):
            key_list = key_vector.cpu().tolist()
        else:
            key_list = list(key_vector)
        
        if isinstance(value_vector, torch.Tensor):
            value_list = value_vector.cpu().tolist()
        else:
            value_list = list(value_vector)
        
        self.persistence.save_knowledge(
            aga_id=self.aga_id,
            slot_idx=slot_idx,
            lu_id=lu_id,
            condition=condition,
            decision=decision,
            key_vector=key_list,
            value_vector=value_list,
            lifecycle_state=lifecycle_state.value,
        )
        
        return slot_idx
    
    def sync_lifecycle_update(
        self,
        aga: AuxiliaryGovernedAttention,
        lu_id: str,
        new_state: LifecycleState,
    ) -> bool:
        """
        同步更新生命周期
        """
        # 更新 AGA
        slots = aga.get_slot_by_lu_id(lu_id)
        for slot_idx in slots:
            aga.update_lifecycle(slot_idx, new_state)
            if self.auto_sync:
                self._dirty_slots.add(slot_idx)
        
        # 更新数据库
        return self.persistence.update_lifecycle(self.aga_id, lu_id, new_state.value)
    
    def sync_quarantine(
        self,
        aga: AuxiliaryGovernedAttention,
        lu_id: str,
    ) -> bool:
        """
        同步隔离知识
        """
        # 隔离 AGA
        quarantined = aga.quarantine_by_lu_id(lu_id)
        if self.auto_sync:
            self._dirty_slots.update(quarantined)
        
        # 更新数据库
        return self.persistence.update_lifecycle(self.aga_id, lu_id, LifecycleState.QUARANTINED.value)
    
    def sync_hit_counts(self, aga: AuxiliaryGovernedAttention) -> bool:
        """
        同步命中计数到数据库
        
        建议定期调用，而非每次推理后调用
        """
        # 收集有命中的 LU IDs
        lu_ids_with_hits = []
        for i in range(aga.num_slots):
            if aga.slot_hit_counts[i] > 0 and aga.slot_lu_ids[i]:
                lu_ids_with_hits.append(aga.slot_lu_ids[i])
        
        if lu_ids_with_hits:
            return self.persistence.increment_hit_count(self.aga_id, lu_ids_with_hits)
        return True
    
    def flush_dirty(self, aga: AuxiliaryGovernedAttention) -> int:
        """
        刷新脏槽位到数据库
        
        Returns:
            同步的槽位数量
        """
        if not self._dirty_slots:
            return 0
        
        records = []
        for slot_idx in self._dirty_slots:
            if slot_idx < aga.num_slots and aga.slot_lu_ids[slot_idx]:
                records.append(KnowledgeRecord(
                    slot_idx=slot_idx,
                    lu_id=aga.slot_lu_ids[slot_idx],
                    condition=aga.slot_conditions[slot_idx] or "",
                    decision=aga.slot_decisions[slot_idx] or "",
                    key_vector=aga.aux_keys[slot_idx].cpu().tolist(),
                    value_vector=aga.aux_values[slot_idx].cpu().tolist(),
                    lifecycle_state=aga.slot_lifecycle[slot_idx].value,
                    hit_count=aga.slot_hit_counts[slot_idx],
                ))
        
        count = self.persistence.save_knowledge_batch(self.aga_id, records)
        self._dirty_slots.clear()
        return count
    
    # v2.0: 外置元数据场景支持
    
    def load_active_only(self, aga: AuxiliaryGovernedAttention) -> int:
        """
        仅加载活跃槽位（外置元数据场景）
        
        DB 管理 lifecycle/LU，AGA 只 load active slots。
        
        Returns:
            加载的槽位数量
        """
        active_knowledge = self.persistence.load_active_knowledge(self.aga_id)
        aga.load_active_slots_only(active_knowledge)
        return len(active_knowledge)
    
    def get_db_statistics(self) -> Dict[str, Any]:
        """获取数据库统计"""
        if hasattr(self.persistence, 'get_statistics'):
            return self.persistence.get_statistics()
        return {}
    
    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取审计日志"""
        if hasattr(self.persistence, 'get_audit_log'):
            return self.persistence.get_audit_log(self.aga_id, limit)
        return []
