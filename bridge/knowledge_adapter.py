"""
Knowledge Adapter - 深度知识内化核心组件

将审计后的知识真正注入到 LLM 内部，而非仅在决策层操作。

原理：
1. 在 Transformer 的 FFN 层后插入 Adapter
2. Adapter 以低秩形式存储知识
3. 通过残差连接将知识融入 LLM 的表示空间

与 Decision Head 的区别：
- Decision Head: 只影响最终决策，不影响 LLM 生成
- Knowledge Adapter: 影响 LLM 的内部表示，从而影响生成

技术参考：
- Adapter-Transformers (Houlsby et al., 2019)
- LoRA (Hu et al., 2021)
- Knowledge Editing (Meng et al., 2022)
"""
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KnowledgeSlot:
    """知识槽位元数据"""
    slot_index: int
    constraint_id: str
    learning_unit_id: str
    condition: str
    decision: str
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'slot_index': self.slot_index,
            'constraint_id': self.constraint_id,
            'learning_unit_id': self.learning_unit_id,
            'condition': self.condition,
            'decision': self.decision,
            'created_at': self.created_at.isoformat(),
            'is_active': self.is_active,
        }


class KnowledgeAdapter(nn.Module):
    """
    Knowledge Adapter - 可插入 Transformer 层的知识注入模块
    
    架构：
    
    Input (hidden_states)
         │
         ▼
    ┌─────────────────┐
    │  Down Project   │  [hidden_dim → bottleneck_dim]
    │  (W_down)       │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Knowledge      │  注意力检索知识槽
    │  Retrieval      │  [bottleneck_dim × num_slots]
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Up Project     │  [bottleneck_dim → hidden_dim]
    │  (W_up)         │
    └────────┬────────┘
             │
             × gate (可学习门控)
             │
             ▼
    Output (knowledge_delta)
    
    最终输出 = FFN(x) + knowledge_delta
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 64,
        num_knowledge_slots: int = 100,
        init_scale: float = 0.01,
        use_gate: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_knowledge_slots = num_knowledge_slots
        self.use_gate = use_gate
        
        # 下投影
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        nn.init.normal_(self.down_proj.weight, std=init_scale)
        
        # 知识存储
        # Keys: 用于检索匹配
        self.knowledge_keys = nn.Parameter(
            torch.randn(num_knowledge_slots, bottleneck_dim) * init_scale
        )
        # Values: 实际知识内容
        self.knowledge_values = nn.Parameter(
            torch.zeros(num_knowledge_slots, bottleneck_dim)
        )
        
        # 上投影
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        nn.init.normal_(self.up_proj.weight, std=init_scale)
        
        # 门控参数（初始化为较小值，让模型逐步学习使用知识）
        if use_gate:
            self.gate = nn.Parameter(torch.tensor([-2.0]))  # sigmoid(-2) ≈ 0.12
        else:
            self.register_buffer('gate', torch.tensor([1.0]))
        
        # 知识槽元数据
        self.slot_metadata: Dict[str, KnowledgeSlot] = {}
        self.next_free_slot = 0
        
        # 槽位使用掩码（用于软删除）
        self.register_buffer(
            'slot_mask',
            torch.ones(num_knowledge_slots)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim] 或 [batch, hidden_dim]
            
        Returns:
            knowledge_delta: 与输入相同形状的知识增量
        """
        original_shape = hidden_states.shape
        
        # 处理不同形状
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)  # [batch, 1, hidden]
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # 下投影
        query = self.down_proj(hidden_states)  # [batch, seq, bottleneck]
        
        # 知识检索（软注意力）
        # query: [batch, seq, bottleneck]
        # knowledge_keys: [num_slots, bottleneck]
        attention_scores = torch.matmul(
            query, 
            self.knowledge_keys.T
        ) / math.sqrt(self.bottleneck_dim)  # [batch, seq, num_slots]
        
        # 应用槽位掩码（被删除的槽位不参与计算）
        attention_scores = attention_scores * self.slot_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, seq, num_slots]
        
        # 聚合知识
        knowledge = torch.matmul(
            attention_weights,
            self.knowledge_values
        )  # [batch, seq, bottleneck]
        
        # 上投影
        knowledge_delta = self.up_proj(knowledge)  # [batch, seq, hidden]
        
        # 门控
        if self.use_gate:
            gate_value = torch.sigmoid(self.gate)
            knowledge_delta = gate_value * knowledge_delta
        
        # 恢复原始形状
        if len(original_shape) == 2:
            knowledge_delta = knowledge_delta.squeeze(1)
        
        return knowledge_delta
    
    def inject_knowledge(
        self,
        constraint_id: str,
        learning_unit_id: str,
        knowledge_vector: torch.Tensor,
        condition: str = "",
        decision: str = "",
        slot_index: Optional[int] = None,
    ) -> int:
        """
        注入知识到指定槽位
        
        Args:
            constraint_id: 约束 ID
            learning_unit_id: Learning Unit ID
            knowledge_vector: 知识向量 [bottleneck_dim]
            condition: 约束条件描述
            decision: 决策类型
            slot_index: 指定槽位（可选）
            
        Returns:
            使用的槽位索引
        """
        if slot_index is None:
            slot_index = self._allocate_slot()
        
        if slot_index >= self.num_knowledge_slots:
            raise ValueError(f"Slot index {slot_index} exceeds capacity {self.num_knowledge_slots}")
        
        # 注入知识向量
        with torch.no_grad():
            # 确保维度正确
            if knowledge_vector.shape[0] != self.bottleneck_dim:
                raise ValueError(
                    f"Knowledge vector dim {knowledge_vector.shape[0]} != bottleneck_dim {self.bottleneck_dim}"
                )
            
            self.knowledge_values[slot_index] = knowledge_vector.to(self.knowledge_values.device)
            self.slot_mask[slot_index] = 1.0  # 激活槽位
        
        # 记录元数据
        self.slot_metadata[constraint_id] = KnowledgeSlot(
            slot_index=slot_index,
            constraint_id=constraint_id,
            learning_unit_id=learning_unit_id,
            condition=condition,
            decision=decision,
        )
        
        return slot_index
    
    def remove_knowledge(self, constraint_id: str) -> bool:
        """
        移除知识（软删除，用于回滚）
        
        Args:
            constraint_id: 约束 ID
            
        Returns:
            是否成功
        """
        if constraint_id not in self.slot_metadata:
            return False
        
        slot = self.slot_metadata[constraint_id]
        slot_index = slot.slot_index
        
        with torch.no_grad():
            # 清零知识向量
            self.knowledge_values[slot_index] = torch.zeros(self.bottleneck_dim)
            # 掩码设为0（软删除）
            self.slot_mask[slot_index] = 0.0
        
        # 更新元数据
        slot.is_active = False
        
        return True
    
    def _allocate_slot(self) -> int:
        """分配空闲槽位"""
        # 优先使用被删除的槽位
        inactive_slots = (self.slot_mask == 0).nonzero(as_tuple=True)[0]
        if len(inactive_slots) > 0:
            return inactive_slots[0].item()
        
        # 使用下一个空闲槽位
        if self.next_free_slot < self.num_knowledge_slots:
            slot = self.next_free_slot
            self.next_free_slot += 1
            return slot
        
        raise RuntimeError("No free knowledge slots available")
    
    def get_active_knowledge_count(self) -> int:
        """获取活跃知识数量"""
        return int(self.slot_mask.sum().item())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'hidden_dim': self.hidden_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'num_slots': self.num_knowledge_slots,
            'active_slots': self.get_active_knowledge_count(),
            'next_free_slot': self.next_free_slot,
            'gate_value': torch.sigmoid(self.gate).item() if self.use_gate else 1.0,
            'metadata_count': len(self.slot_metadata),
        }
    
    def export_state(self) -> Dict[str, Any]:
        """导出状态（用于持久化）"""
        return {
            'weights': {
                'down_proj': self.down_proj.weight.detach().cpu().numpy().tolist(),
                'up_proj': self.up_proj.weight.detach().cpu().numpy().tolist(),
                'knowledge_keys': self.knowledge_keys.detach().cpu().numpy().tolist(),
                'knowledge_values': self.knowledge_values.detach().cpu().numpy().tolist(),
                'gate': self.gate.detach().cpu().numpy().tolist() if self.use_gate else [1.0],
                'slot_mask': self.slot_mask.detach().cpu().numpy().tolist(),
            },
            'metadata': {
                cid: slot.to_dict() 
                for cid, slot in self.slot_metadata.items()
            },
            'next_free_slot': self.next_free_slot,
        }
    
    def import_state(self, state: Dict[str, Any]):
        """导入状态"""
        weights = state['weights']
        
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.tensor(weights['down_proj']))
            self.up_proj.weight.copy_(torch.tensor(weights['up_proj']))
            self.knowledge_keys.copy_(torch.tensor(weights['knowledge_keys']))
            self.knowledge_values.copy_(torch.tensor(weights['knowledge_values']))
            if self.use_gate:
                self.gate.copy_(torch.tensor(weights['gate']))
            self.slot_mask.copy_(torch.tensor(weights['slot_mask']))
        
        self.next_free_slot = state['next_free_slot']
        
        # 恢复元数据
        self.slot_metadata = {}
        for cid, slot_dict in state['metadata'].items():
            self.slot_metadata[cid] = KnowledgeSlot(
                slot_index=slot_dict['slot_index'],
                constraint_id=slot_dict['constraint_id'],
                learning_unit_id=slot_dict['learning_unit_id'],
                condition=slot_dict['condition'],
                decision=slot_dict['decision'],
                is_active=slot_dict['is_active'],
            )


class KnowledgeAdapterLayer(nn.Module):
    """
    包装层 - 将 Knowledge Adapter 与原始 FFN 组合
    
    使用方式：替换 Transformer 层的 FFN
    
    output = original_ffn(x) + adapter(x)
    """
    
    def __init__(
        self,
        original_ffn: nn.Module,
        adapter: KnowledgeAdapter,
        adapter_position: str = "post",  # "pre", "post", "parallel"
    ):
        super().__init__()
        self.original_ffn = original_ffn
        self.adapter = adapter
        self.adapter_position = adapter_position
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.adapter_position == "pre":
            # Adapter 在 FFN 之前
            adapter_output = self.adapter(hidden_states)
            return self.original_ffn(hidden_states + adapter_output)
        
        elif self.adapter_position == "post":
            # Adapter 在 FFN 之后（推荐）
            ffn_output = self.original_ffn(hidden_states)
            adapter_output = self.adapter(hidden_states)
            return ffn_output + adapter_output
        
        elif self.adapter_position == "parallel":
            # 并行
            ffn_output = self.original_ffn(hidden_states)
            adapter_output = self.adapter(hidden_states)
            return ffn_output + adapter_output
        
        else:
            raise ValueError(f"Unknown adapter position: {self.adapter_position}")


class KnowledgeAdapterManager:
    """
    Knowledge Adapter 管理器
    
    管理多个 Adapter 实例，协调知识注入和回滚
    """
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 64,
        num_slots_per_adapter: int = 100,
        num_adapters: int = 4,  # 通常插入到后几层
    ):
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # 创建 Adapter 实例
        self.adapters: List[KnowledgeAdapter] = [
            KnowledgeAdapter(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                num_knowledge_slots=num_slots_per_adapter,
            )
            for _ in range(num_adapters)
        ]
        
        # 内化记录
        self.internalization_history: List[Dict[str, Any]] = []
        
        # 版本控制
        self.current_version = 0
        self.version_snapshots: Dict[int, List[Dict]] = {}
    
    def inject_knowledge_to_all(
        self,
        constraint_id: str,
        learning_unit_id: str,
        knowledge_vector: torch.Tensor,
        condition: str = "",
        decision: str = "",
    ) -> List[int]:
        """
        将知识注入到所有 Adapter
        
        Returns:
            各 Adapter 使用的槽位索引列表
        """
        slot_indices = []
        
        for adapter in self.adapters:
            # 每个 Adapter 使用相同的知识向量
            slot_idx = adapter.inject_knowledge(
                constraint_id=constraint_id,
                learning_unit_id=learning_unit_id,
                knowledge_vector=knowledge_vector.clone(),
                condition=condition,
                decision=decision,
            )
            slot_indices.append(slot_idx)
        
        # 记录
        self.internalization_history.append({
            'constraint_id': constraint_id,
            'learning_unit_id': learning_unit_id,
            'slot_indices': slot_indices,
            'timestamp': datetime.now().isoformat(),
        })
        
        return slot_indices
    
    def remove_knowledge_from_all(self, constraint_id: str) -> bool:
        """从所有 Adapter 移除知识"""
        success = True
        for adapter in self.adapters:
            if not adapter.remove_knowledge(constraint_id):
                success = False
        return success
    
    def create_version_snapshot(self) -> int:
        """创建版本快照"""
        self.current_version += 1
        self.version_snapshots[self.current_version] = [
            adapter.export_state() for adapter in self.adapters
        ]
        return self.current_version
    
    def rollback_to_version(self, version: int) -> bool:
        """回滚到指定版本"""
        if version not in self.version_snapshots:
            return False
        
        states = self.version_snapshots[version]
        for adapter, state in zip(self.adapters, states):
            adapter.import_state(state)
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'num_adapters': len(self.adapters),
            'current_version': self.current_version,
            'versions_stored': list(self.version_snapshots.keys()),
            'adapters': [adapter.get_statistics() for adapter in self.adapters],
            'total_active_knowledge': sum(
                adapter.get_active_knowledge_count() for adapter in self.adapters
            ),
        }


def create_knowledge_encoder(
    llm_hidden_dim: int,
    bottleneck_dim: int,
    use_projection: bool = True,
) -> Callable[[str, Any], torch.Tensor]:
    """
    创建知识编码函数
    
    将文本描述编码为知识向量
    
    Args:
        llm_hidden_dim: LLM 隐藏维度
        bottleneck_dim: 目标维度
        use_projection: 是否使用投影层
        
    Returns:
        编码函数
    """
    if use_projection:
        projection = nn.Linear(llm_hidden_dim, bottleneck_dim, bias=False)
        nn.init.orthogonal_(projection.weight)
    else:
        projection = None
    
    def encode(text: str, llm_adapter) -> torch.Tensor:
        """
        编码文本为知识向量
        
        Args:
            text: 约束描述文本
            llm_adapter: LLM 适配器（用于获取 embedding）
            
        Returns:
            知识向量 [bottleneck_dim]
        """
        # 方案1：如果 LLM 适配器支持 embedding
        if hasattr(llm_adapter, 'get_embeddings'):
            try:
                embedding = llm_adapter.get_embeddings(text)
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
            except Exception:
                # 回退到哈希方案
                embedding_tensor = _hash_to_vector(text, llm_hidden_dim)
        else:
            # 方案2：使用确定性哈希
            embedding_tensor = _hash_to_vector(text, llm_hidden_dim)
        
        # 投影到 bottleneck 维度
        if projection is not None:
            with torch.no_grad():
                knowledge_vector = projection(embedding_tensor)
        else:
            knowledge_vector = embedding_tensor[:bottleneck_dim]
        
        # 归一化
        knowledge_vector = F.normalize(knowledge_vector, dim=-1)
        
        return knowledge_vector
    
    return encode


def _hash_to_vector(text: str, dim: int) -> torch.Tensor:
    """
    使用哈希生成确定性向量
    
    比纯随机更好，因为相同文本会产生相同向量
    """
    # 多次哈希以生成足够维度
    vectors = []
    for i in range(dim // 32 + 1):
        h = hashlib.sha256(f"{text}_{i}".encode()).digest()
        # 将字节转换为浮点数
        for j in range(0, min(32, dim - len(vectors)), 4):
            value = int.from_bytes(h[j:j+4], 'little', signed=True)
            vectors.append(value / (2**31))  # 归一化到 [-1, 1]
    
    return torch.tensor(vectors[:dim], dtype=torch.float32)

