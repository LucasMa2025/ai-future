"""
连续记忆系统 (Continuum Memory System)

基于 NL 论文的多频率记忆机制实现。
提供 fast/mid/slow 多层级记忆管理。
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import json
import hashlib
import uuid

from .types import (
    NLLevel, 
    MemoryLevel, 
    ContinuumMemoryState,
    LevelDelta,
    LearningScope,
)


class ContinuumMemorySystem:
    """
    连续记忆系统
    
    实现 NL 论文的 CMS 核心功能：
    1. 多频率更新机制
    2. 层级间信息传递
    3. 记忆固化（consolidation）
    4. 状态快照与回滚
    
    与 NLGSM 治理的对接：
    - freeze/unfreeze 与 NLGSM 状态机同步
    - 所有变更产生 LevelDelta 用于审计
    - 支持 Scope 控制的受限更新
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        state_id: Optional[str] = None
    ):
        self.state_id = state_id or f"cms_{uuid.uuid4().hex[:8]}"
        self.config = config or {}
        
        # 初始化各层级记忆
        self.levels: Dict[NLLevel, MemoryLevel] = {}
        self._init_levels()
        
        # 全局状态
        self.global_step: int = 0
        self.is_frozen: bool = True  # 默认冻结
        
        # 变更历史（用于审计）
        self.delta_history: List[LevelDelta] = []
        
        # 快照
        self.snapshots: Dict[str, ContinuumMemoryState] = {}
    
    def _init_levels(self):
        """初始化记忆层级"""
        level_configs = self.config.get("levels", {})
        
        for nl_level in NLLevel:
            level_config = level_configs.get(nl_level.name, {})
            
            self.levels[nl_level] = MemoryLevel(
                level_id=f"{self.state_id}_{nl_level.name.lower()}",
                nl_level=nl_level,
                frequency=level_config.get("frequency", nl_level.default_frequency),
                capacity=level_config.get("capacity", 10000),
            )
    
    # ==================== 核心操作 ====================
    
    def step(
        self,
        context: Dict[str, Any],
        scope: LearningScope
    ) -> List[LevelDelta]:
        """
        执行一步学习
        
        这是 CMS 的核心方法，实现多频率更新逻辑。
        
        Args:
            context: 输入上下文（数据、梯度等）
            scope: 学习范围控制
            
        Returns:
            产生的层级变更列表
        """
        if self.is_frozen:
            raise RuntimeError("CMS is frozen, cannot perform learning step")
        
        self.global_step += 1
        deltas = []
        
        # 按层级顺序处理（从快到慢）
        for nl_level in sorted(NLLevel, key=lambda x: x.value):
            # 检查是否可以更新
            if not scope.can_update(nl_level, self.global_step):
                continue
            
            memory = self.levels[nl_level]
            
            # 检查频率
            if not memory.should_update(self.global_step):
                continue
            
            # 执行层级更新
            delta = self._update_level(nl_level, context, scope)
            
            if delta:
                deltas.append(delta)
                self.delta_history.append(delta)
                scope.consume_budget(nl_level)
        
        # 执行层级间固化（如果需要）
        consolidation_deltas = self._consolidate()
        deltas.extend(consolidation_deltas)
        
        return deltas
    
    def _update_level(
        self,
        nl_level: NLLevel,
        context: Dict[str, Any],
        scope: LearningScope
    ) -> Optional[LevelDelta]:
        """
        更新单个层级
        
        根据层级类型执行不同的更新策略
        """
        memory = self.levels[nl_level]
        
        # 根据层级类型构建更新内容
        update_content = self._build_update_content(nl_level, context)
        
        if not update_content:
            return None
        
        # 执行更新
        delta = memory.update(update_content, self.global_step)
        
        return delta
    
    def _build_update_content(
        self,
        nl_level: NLLevel,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        根据层级构建更新内容
        
        这是将原始 context 映射到特定层级记忆的关键逻辑
        """
        if nl_level == NLLevel.PARAMETER:
            # 参数层：存储即时梯度/更新信息
            return {
                "step": self.global_step,
                "gradient_norm": context.get("gradient_norm"),
                "loss": context.get("loss"),
                "timestamp": datetime.now().isoformat(),
            }
        
        elif nl_level == NLLevel.MEMORY:
            # 记忆层：存储中期模式和关联
            findings = context.get("findings", [])
            if findings:
                return {
                    "step": self.global_step,
                    "patterns": findings[-5:],  # 最近的发现
                    "associations": context.get("associations", {}),
                }
        
        elif nl_level == NLLevel.OPTIMIZER:
            # 优化器层：存储优化策略状态
            return {
                "step": self.global_step,
                "learning_rate": context.get("learning_rate"),
                "momentum_state": context.get("momentum", {}),
                "strategy_params": context.get("strategy", {}),
            }
        
        elif nl_level == NLLevel.POLICY:
            # 策略层：存储学习策略本身
            return {
                "step": self.global_step,
                "policy_update": context.get("policy_update"),
                "meta_gradient": context.get("meta_gradient"),
            }
        
        return None
    
    def _consolidate(self) -> List[LevelDelta]:
        """
        层级间固化
        
        实现 NL 论文的 consolidation 机制：
        - 快层级的稳定模式向慢层级传递
        - 慢层级的结构化知识向快层级提供指导
        """
        deltas = []
        
        # 简化的固化逻辑：每 100 步执行一次
        if self.global_step % 100 != 0:
            return deltas
        
        # PARAMETER -> MEMORY 固化
        param_memory = self.levels[NLLevel.PARAMETER]
        memory_memory = self.levels[NLLevel.MEMORY]
        
        if param_memory.content.get("patterns"):
            # 将参数层的稳定模式传递到记忆层
            consolidation_content = {
                "consolidated_from": "PARAMETER",
                "step": self.global_step,
                "stable_patterns": param_memory.content.get("patterns", []),
            }
            
            delta = memory_memory.update(consolidation_content, self.global_step)
            delta.delta_type = "consolidate"
            deltas.append(delta)
            self.delta_history.append(delta)
        
        return deltas
    
    # ==================== 状态管理 ====================
    
    def freeze(self):
        """冻结 CMS，禁止学习操作"""
        self.is_frozen = True
    
    def unfreeze(self):
        """解冻 CMS，允许学习操作"""
        self.is_frozen = False
    
    def create_snapshot(self, snapshot_id: Optional[str] = None) -> ContinuumMemoryState:
        """
        创建状态快照
        
        用于 NLGSM 的 Frozen 状态快照和回滚
        """
        snapshot_id = snapshot_id or f"snap_{self.global_step}_{uuid.uuid4().hex[:8]}"
        
        # 深拷贝各层级
        levels_copy = {}
        for nl_level, memory in self.levels.items():
            levels_copy[nl_level] = MemoryLevel(
                level_id=memory.level_id,
                nl_level=memory.nl_level,
                frequency=memory.frequency,
                capacity=memory.capacity,
                current_size=memory.current_size,
                last_update_step=memory.last_update_step,
                total_updates=memory.total_updates,
                content=memory.content.copy(),
                content_hash=memory.content_hash,
            )
        
        state = ContinuumMemoryState(
            state_id=snapshot_id,
            levels=levels_copy,
            global_step=self.global_step,
            is_frozen=self.is_frozen,
        )
        state.integrity_hash = state.compute_integrity_hash()
        
        self.snapshots[snapshot_id] = state
        
        return state
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        从快照恢复
        
        用于 NLGSM 的 Rollback 操作
        """
        if snapshot_id not in self.snapshots:
            return False
        
        state = self.snapshots[snapshot_id]
        
        # 恢复各层级
        for nl_level, memory_state in state.levels.items():
            self.levels[nl_level] = MemoryLevel(
                level_id=memory_state.level_id,
                nl_level=memory_state.nl_level,
                frequency=memory_state.frequency,
                capacity=memory_state.capacity,
                current_size=memory_state.current_size,
                last_update_step=memory_state.last_update_step,
                total_updates=memory_state.total_updates,
                content=memory_state.content.copy(),
                content_hash=memory_state.content_hash,
            )
        
        self.global_step = state.global_step
        self.is_frozen = True  # 恢复后默认冻结
        
        # 记录回滚操作
        for nl_level in state.levels:
            rollback_delta = LevelDelta(
                level=nl_level,
                delta_type="rollback",
                pre_state_hash=self.levels[nl_level].content_hash,
                post_state_hash=state.levels[nl_level].content_hash,
                delta_content={"restored_from": snapshot_id},
                step_number=self.global_step,
            )
            self.delta_history.append(rollback_delta)
        
        return True
    
    # ==================== 查询接口 ====================
    
    def get_state(self) -> ContinuumMemoryState:
        """获取当前状态"""
        return ContinuumMemoryState(
            state_id=self.state_id,
            levels=self.levels.copy(),
            global_step=self.global_step,
            is_frozen=self.is_frozen,
        )
    
    def get_level_state(self, nl_level: NLLevel) -> Dict[str, Any]:
        """获取单个层级状态"""
        if nl_level not in self.levels:
            return {}
        return self.levels[nl_level].to_dict()
    
    def get_recent_deltas(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的变更"""
        return [d.to_dict() for d in self.delta_history[-count:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        level_stats = {}
        for nl_level, memory in self.levels.items():
            level_stats[nl_level.name] = {
                "total_updates": memory.total_updates,
                "current_size": memory.current_size,
                "last_update_step": memory.last_update_step,
            }
        
        return {
            "state_id": self.state_id,
            "global_step": self.global_step,
            "is_frozen": self.is_frozen,
            "total_deltas": len(self.delta_history),
            "snapshots_count": len(self.snapshots),
            "levels": level_stats,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化完整状态"""
        return {
            "state_id": self.state_id,
            "global_step": self.global_step,
            "is_frozen": self.is_frozen,
            "levels": {k.name: v.to_dict() for k, v in self.levels.items()},
            "config": self.config,
            "statistics": self.get_statistics(),
        }

