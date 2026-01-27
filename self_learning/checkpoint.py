"""
Checkpoint 管理器

管理学习过程中的检查点
"""
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json
import os

from core.types import generate_id


class CheckpointManager:
    """
    Checkpoint 管理器
    
    特性：
    - 自动触发 Checkpoint
    - 检查点持久化
    - 检查点恢复
    """
    
    def __init__(
        self,
        storage_dir: str = "./data/checkpoints",
        time_interval_minutes: int = 30,
        knowledge_count_threshold: int = 5
    ):
        self.storage_dir = storage_dir
        self.time_interval_minutes = time_interval_minutes
        self.knowledge_count_threshold = knowledge_count_threshold
        
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
        
        # 检查点列表
        self.checkpoints: List[Dict[str, Any]] = []
        
        # 上次检查点时间
        self.last_checkpoint_time: Optional[datetime] = None
        
        # 当前知识计数
        self.current_knowledge_count: int = 0
        
        # 回调
        self.on_checkpoint_created: Optional[Callable] = None
    
    def should_checkpoint(
        self,
        force: bool = False,
        reason: Optional[str] = None
    ) -> bool:
        """
        判断是否应该创建检查点
        
        Args:
            force: 强制创建
            reason: 原因
            
        Returns:
            是否应该创建
        """
        if force:
            return True
        
        # 时间间隔检查
        if self.last_checkpoint_time:
            elapsed = datetime.now() - self.last_checkpoint_time
            if elapsed.total_seconds() >= self.time_interval_minutes * 60:
                return True
        
        # 知识数量检查
        if self.current_knowledge_count >= self.knowledge_count_threshold:
            return True
        
        return False
    
    def create_checkpoint(
        self,
        exploration_data: Dict[str, Any],
        reason: str = "auto"
    ) -> Dict[str, Any]:
        """
        创建检查点
        
        Args:
            exploration_data: 探索数据
            reason: 原因
            
        Returns:
            检查点数据
        """
        checkpoint_id = generate_id("ckpt")
        
        checkpoint = {
            'checkpoint_id': checkpoint_id,
            'reason': reason,
            'created_at': datetime.now().isoformat(),
            'exploration_data': exploration_data,
            'knowledge_count': self.current_knowledge_count,
        }
        
        # 保存到文件
        self._save_checkpoint(checkpoint)
        
        # 更新状态
        self.checkpoints.append(checkpoint)
        self.last_checkpoint_time = datetime.now()
        self.current_knowledge_count = 0
        
        # 触发回调
        if self.on_checkpoint_created:
            self.on_checkpoint_created(checkpoint)
        
        print(f"[Checkpoint] 创建: {checkpoint_id} - {reason}")
        
        return checkpoint
    
    def _save_checkpoint(self, checkpoint: Dict[str, Any]):
        """保存检查点到文件"""
        filepath = os.path.join(
            self.storage_dir,
            f"{checkpoint['checkpoint_id']}.json"
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """加载检查点"""
        filepath = os.path.join(self.storage_dir, f"{checkpoint_id}.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """列出所有检查点"""
        checkpoints = []
        
        for filename in os.listdir(self.storage_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.storage_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                    checkpoints.append({
                        'checkpoint_id': checkpoint['checkpoint_id'],
                        'reason': checkpoint['reason'],
                        'created_at': checkpoint['created_at'],
                    })
        
        # 按时间排序
        checkpoints.sort(key=lambda x: x['created_at'], reverse=True)
        
        return checkpoints
    
    def increment_knowledge_count(self, count: int = 1):
        """增加知识计数"""
        self.current_knowledge_count += count
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_checkpoints': len(self.checkpoints),
            'current_knowledge_count': self.current_knowledge_count,
            'last_checkpoint_time': self.last_checkpoint_time.isoformat() if self.last_checkpoint_time else None,
            'time_interval_minutes': self.time_interval_minutes,
            'knowledge_count_threshold': self.knowledge_count_threshold,
        }

