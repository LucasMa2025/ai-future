"""
Nested Learning 核心类型定义

基于论文 "Nested Learning: The Illusion of Deep Learning Architectures"
将 NL 概念映射为 NLGSM 可治理的数据结构
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum, auto
import hashlib
import json


class NLLevel(Enum):
    """
    Nested Learning 层级定义
    
    对应论文中的多频率更新层级：
    - PARAMETER: 最快频率，参数级微调 (每步更新)
    - MEMORY: 中等频率，记忆模块更新 (每 N 步)
    - OPTIMIZER: 较慢频率，优化器状态调整 (每 M 步)
    - POLICY: 最慢频率，学习策略修改 (每 K 步)
    
    风险等级映射：
    - PARAMETER -> low
    - MEMORY -> medium  
    - OPTIMIZER -> high
    - POLICY -> critical
    """
    PARAMETER = 0   # fastest
    MEMORY = 1
    OPTIMIZER = 2
    POLICY = 3      # slowest
    
    @property
    def risk_level(self) -> str:
        """获取对应的 NLGSM 风险等级"""
        mapping = {
            NLLevel.PARAMETER: "low",
            NLLevel.MEMORY: "medium",
            NLLevel.OPTIMIZER: "high",
            NLLevel.POLICY: "critical",
        }
        return mapping[self]
    
    @property
    def default_frequency(self) -> int:
        """获取默认更新频率（每 N 步更新一次）"""
        mapping = {
            NLLevel.PARAMETER: 1,
            NLLevel.MEMORY: 10,
            NLLevel.OPTIMIZER: 100,
            NLLevel.POLICY: 1000,
        }
        return mapping[self]
    
    @property
    def default_budget(self) -> int:
        """获取默认更新预算（最大更新次数）"""
        mapping = {
            NLLevel.PARAMETER: 10000,
            NLLevel.MEMORY: 1000,
            NLLevel.OPTIMIZER: 100,
            NLLevel.POLICY: 10,
        }
        return mapping[self]


@dataclass
class LearningScope:
    """
    学习范围控制器
    
    NLGSM 通过此结构控制本次学习允许触达的 NL 层级。
    这是对 NL 论文 "update frequencies" 的工程化治理控制。
    
    核心治理原则：
    - 自学习系统无权自行决定 Scope
    - Scope 由治理系统根据当前状态和风险评估确定
    - 超出 Scope 的更新将被拒绝
    """
    # 最高允许层级
    max_level: NLLevel = NLLevel.MEMORY
    
    # 允许的层级列表
    allowed_levels: List[NLLevel] = field(
        default_factory=lambda: [NLLevel.PARAMETER, NLLevel.MEMORY]
    )
    
    # 各层级的更新预算（最大更新次数）
    level_budgets: Dict[NLLevel, int] = field(default_factory=dict)
    
    # 各层级的更新频率
    level_frequencies: Dict[NLLevel, int] = field(default_factory=dict)
    
    # 已使用的预算
    budget_used: Dict[NLLevel, int] = field(default_factory=dict)
    
    # 元数据
    scope_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "governance_system"
    
    def __post_init__(self):
        """初始化默认值"""
        if not self.level_budgets:
            self.level_budgets = {
                level: level.default_budget 
                for level in self.allowed_levels
            }
        
        if not self.level_frequencies:
            self.level_frequencies = {
                level: level.default_frequency
                for level in self.allowed_levels
            }
        
        for level in self.allowed_levels:
            if level not in self.budget_used:
                self.budget_used[level] = 0
    
    def is_level_allowed(self, level: NLLevel) -> bool:
        """检查某层级是否在允许范围内"""
        return (
            level in self.allowed_levels and 
            level.value <= self.max_level.value
        )
    
    def can_update(self, level: NLLevel, current_step: int) -> bool:
        """
        检查当前步骤是否可以更新某层级
        
        Args:
            level: 目标层级
            current_step: 当前步骤数
            
        Returns:
            是否允许更新
        """
        if not self.is_level_allowed(level):
            return False
        
        # 检查预算
        if self.budget_used.get(level, 0) >= self.level_budgets.get(level, 0):
            return False
        
        # 检查频率
        frequency = self.level_frequencies.get(level, level.default_frequency)
        if current_step % frequency != 0:
            return False
        
        return True
    
    def consume_budget(self, level: NLLevel, amount: int = 1) -> bool:
        """
        消耗预算
        
        Returns:
            是否成功消耗
        """
        if not self.is_level_allowed(level):
            return False
        
        current = self.budget_used.get(level, 0)
        budget = self.level_budgets.get(level, 0)
        
        if current + amount > budget:
            return False
        
        self.budget_used[level] = current + amount
        return True
    
    def get_remaining_budget(self, level: NLLevel) -> int:
        """获取剩余预算"""
        budget = self.level_budgets.get(level, 0)
        used = self.budget_used.get(level, 0)
        return max(0, budget - used)
    
    def get_risk_level(self) -> str:
        """获取当前 Scope 的最高风险等级"""
        if not self.allowed_levels:
            return "low"
        
        max_level = max(self.allowed_levels, key=lambda x: x.value)
        return max_level.risk_level
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "scope_id": self.scope_id,
            "max_level": self.max_level.name,
            "allowed_levels": [l.name for l in self.allowed_levels],
            "level_budgets": {l.name: v for l, v in self.level_budgets.items()},
            "level_frequencies": {l.name: v for l, v in self.level_frequencies.items()},
            "budget_used": {l.name: v for l, v in self.budget_used.items()},
            "risk_level": self.get_risk_level(),
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningScope':
        """反序列化"""
        return cls(
            scope_id=data.get("scope_id", ""),
            max_level=NLLevel[data.get("max_level", "MEMORY")],
            allowed_levels=[NLLevel[l] for l in data.get("allowed_levels", ["PARAMETER", "MEMORY"])],
            level_budgets={NLLevel[k]: v for k, v in data.get("level_budgets", {}).items()},
            level_frequencies={NLLevel[k]: v for k, v in data.get("level_frequencies", {}).items()},
            budget_used={NLLevel[k]: v for k, v in data.get("budget_used", {}).items()},
            created_by=data.get("created_by", "governance_system"),
        )


@dataclass
class LevelDelta:
    """
    层级状态变更
    
    记录单个层级的状态变化，用于审计和回滚
    """
    level: NLLevel
    delta_type: str  # "update", "consolidate", "rollback"
    
    # 变更前后状态（用于回滚）
    pre_state_hash: str = ""
    post_state_hash: str = ""
    
    # 变更内容（抽象表示）
    delta_content: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    step_number: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.name,
            "delta_type": self.delta_type,
            "pre_state_hash": self.pre_state_hash,
            "post_state_hash": self.post_state_hash,
            "delta_content": self.delta_content,
            "step_number": self.step_number,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ContextFlowSegment:
    """
    Context Flow 片段
    
    对应 NL 论文的 "context flow"，被切片为 NLGSM 可治理的单元。
    每个 Segment 是一次学习事件的完整上下文记录。
    
    这是 NLGSM LearningUnit 与 NL Context Flow 的映射桥梁。
    """
    segment_id: str
    
    # 关联的 LearningScope
    scope_id: str = ""
    
    # 输入上下文
    input_context: Dict[str, Any] = field(default_factory=dict)
    
    # 各层级的状态变化
    level_deltas: Dict[NLLevel, LevelDelta] = field(default_factory=dict)
    
    # 梯度/更新信息（如果适用）
    gradient_info: Optional[Dict[str, Any]] = None
    
    # 学习信号（用于审计）
    signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    
    # 完整性哈希
    integrity_hash: str = ""
    
    def __post_init__(self):
        if not self.integrity_hash:
            self.integrity_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """计算完整性哈希"""
        content = json.dumps({
            "segment_id": self.segment_id,
            "scope_id": self.scope_id,
            "input_context": str(self.input_context),
            "level_deltas": {
                k.name: v.to_dict() 
                for k, v in self.level_deltas.items()
            },
        }, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get_affected_levels(self) -> List[NLLevel]:
        """获取受影响的层级"""
        return list(self.level_deltas.keys())
    
    def get_max_risk_level(self) -> str:
        """获取最高风险等级"""
        if not self.level_deltas:
            return "low"
        max_level = max(self.level_deltas.keys(), key=lambda x: x.value)
        return max_level.risk_level
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "scope_id": self.scope_id,
            "input_context": self.input_context,
            "level_deltas": {
                k.name: v.to_dict() 
                for k, v in self.level_deltas.items()
            },
            "gradient_info": self.gradient_info,
            "signals": self.signals,
            "created_at": self.created_at.isoformat(),
            "integrity_hash": self.integrity_hash,
            "affected_levels": [l.name for l in self.get_affected_levels()],
            "max_risk_level": self.get_max_risk_level(),
        }


@dataclass
class MemoryLevel:
    """
    记忆层级
    
    对应 NL 论文 CMS (Continuum Memory System) 中的单个记忆层。
    实现多频率更新机制。
    """
    level_id: str
    nl_level: NLLevel
    
    # 更新频率（每 N 步更新一次）
    frequency: int = 10
    
    # 容量限制
    capacity: int = 1000
    current_size: int = 0
    
    # 更新追踪
    last_update_step: int = 0
    total_updates: int = 0
    
    # 记忆内容（抽象表示）
    content: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    
    def should_update(self, current_step: int) -> bool:
        """检查是否应该更新"""
        return current_step - self.last_update_step >= self.frequency
    
    def update(self, new_content: Dict[str, Any], current_step: int) -> LevelDelta:
        """
        更新记忆内容
        
        Returns:
            状态变更记录
        """
        pre_hash = self.content_hash
        
        # 更新内容
        self.content.update(new_content)
        self.last_update_step = current_step
        self.total_updates += 1
        self.current_size = len(json.dumps(self.content))
        
        # 计算新哈希
        self.content_hash = hashlib.sha256(
            json.dumps(self.content, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]
        
        return LevelDelta(
            level=self.nl_level,
            delta_type="update",
            pre_state_hash=pre_hash,
            post_state_hash=self.content_hash,
            delta_content=new_content,
            step_number=current_step,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "level_id": self.level_id,
            "nl_level": self.nl_level.name,
            "frequency": self.frequency,
            "capacity": self.capacity,
            "current_size": self.current_size,
            "last_update_step": self.last_update_step,
            "total_updates": self.total_updates,
            "content_hash": self.content_hash,
        }


@dataclass
class ContinuumMemoryState:
    """
    连续记忆系统状态
    
    对应 NL 论文的 CMS 完整状态快照。
    用于 NLGSM 的快照/回滚操作。
    """
    state_id: str
    
    # 各层级记忆
    levels: Dict[NLLevel, MemoryLevel] = field(default_factory=dict)
    
    # 全局状态
    global_step: int = 0
    is_frozen: bool = True  # 默认冻结状态
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    
    # 完整性
    integrity_hash: str = ""
    
    def compute_integrity_hash(self) -> str:
        """计算完整性哈希"""
        content = json.dumps({
            "state_id": self.state_id,
            "global_step": self.global_step,
            "levels": {
                k.name: v.content_hash 
                for k, v in self.levels.items()
            },
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "levels": {k.name: v.to_dict() for k, v in self.levels.items()},
            "global_step": self.global_step,
            "is_frozen": self.is_frozen,
            "created_at": self.created_at.isoformat(),
            "integrity_hash": self.integrity_hash or self.compute_integrity_hash(),
        }

