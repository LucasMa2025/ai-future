"""
AGA (Auxiliary Governed Attention) 核心模块

实现热插拔式知识注入系统，无需向量化训练。

核心特性：
- 零训练注入：知识直接写入 buffer，无需梯度计算
- 热插拔设计：运行时动态添加/移除知识
- 治理控制：生命周期状态、熵门控、可追溯性
- 即时隔离：问题知识可立即移除影响
"""
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class LifecycleState(str, Enum):
    """知识槽位生命周期状态"""
    PROBATIONARY = "probationary"  # 试用期 (r=0.3)
    CONFIRMED = "confirmed"        # 已确认 (r=1.0)
    DEPRECATED = "deprecated"      # 已弃用 (r=0.1)
    QUARANTINED = "quarantined"    # 已隔离 (r=0.0)


@dataclass
class KnowledgeSlotInfo:
    """知识槽位信息"""
    slot_idx: int
    lu_id: Optional[str]
    lifecycle_state: LifecycleState
    reliability: float
    key_norm: float
    value_norm: float
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    hit_count: int = 0  # 被激活次数


@dataclass
class AGADiagnostics:
    """AGA 诊断信息"""
    entropy: torch.Tensor
    gate: torch.Tensor
    aux_attn_weights: torch.Tensor
    slot_reliability: torch.Tensor
    active_slots: int
    top_activated_slots: List[int] = field(default_factory=list)


class AuxiliaryGovernedAttention(nn.Module):
    """
    辅助治理注意力 (AGA)
    
    热插拔式知识注入模块，无需训练即可使用。
    
    工作原理：
    1. 知识以 key-value 对形式存储在 buffer 中
    2. 推理时，模型的隐藏状态作为 query 检索相关知识
    3. 通过熵门控和生命周期可靠性控制知识贡献
    4. 融合到主注意力输出中
    
    关键设计：
    - aux_keys/aux_values 是 buffer（非 Parameter），不参与训练
    - 已隔离槽位通过 log(0)=-∞ 在 softmax 中被完全屏蔽
    - 可靠性缓存避免重复计算
    """
    
    # 生命周期到可靠性的映射
    LIFECYCLE_RELIABILITY = {
        LifecycleState.PROBATIONARY: 0.3,
        LifecycleState.CONFIRMED: 1.0,
        LifecycleState.DEPRECATED: 0.1,
        LifecycleState.QUARANTINED: 0.0,
    }
    
    def __init__(
        self,
        hidden_dim: int = 4096,
        bottleneck_dim: int = 64,
        num_slots: int = 100,
        num_heads: int = 32,
        tau_low: float = 0.5,
        tau_high: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.num_slots = num_slots
        
        # 查询投影（可训练用于适配，但也可使用默认值）
        self.q_proj = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        
        # 辅助键值存储 - 使用 buffer（不可训练）
        self.register_buffer(
            'aux_keys',
            torch.randn(num_slots, bottleneck_dim) * 0.01
        )
        self.register_buffer(
            'aux_values',
            torch.zeros(num_slots, hidden_dim)
        )
        
        # 熵门控参数
        self.gate_w1 = nn.Parameter(torch.tensor(0.5))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        
        # 槽位元数据
        self.slot_lifecycle: List[LifecycleState] = [LifecycleState.QUARANTINED] * num_slots
        self.slot_lu_ids: List[Optional[str]] = [None] * num_slots
        self.slot_created_at: List[Optional[datetime]] = [None] * num_slots
        self.slot_hit_counts: List[int] = [0] * num_slots
        
        # 可靠性缓存
        self._cached_reliability: Optional[torch.Tensor] = None
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._cached_reliability = None
    
    def _get_reliability_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取可靠性张量（带缓存）"""
        if self._cached_reliability is None:
            self._cached_reliability = torch.tensor(
                [self.LIFECYCLE_RELIABILITY[state] for state in self.slot_lifecycle],
                device=device, dtype=dtype
            )
        return self._cached_reliability.to(device=device, dtype=dtype)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        primary_attention_output: torch.Tensor,
        primary_attention_weights: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[AGADiagnostics]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            primary_attention_output: [batch, seq, hidden_dim]
            primary_attention_weights: [batch, heads, seq, seq] 可选
            return_diagnostics: 是否返回诊断信息
        
        Returns:
            fused_output, diagnostics
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # 1. 查询投影和注意力分数
        query = self.q_proj(hidden_states)
        attn_scores = torch.matmul(query, self.aux_keys.T) / math.sqrt(self.bottleneck_dim)
        
        # 2. 应用可靠性掩码
        reliability = self._get_reliability_tensor(device, dtype)
        reliability_mask = torch.log(reliability + 1e-10)
        attn_scores = attn_scores + reliability_mask.unsqueeze(0).unsqueeze(0)
        
        # 3. Softmax 和值检索
        attn_weights = F.softmax(attn_scores, dim=-1)
        aux_output = torch.matmul(attn_weights, self.aux_values)
        
        # 4. 计算熵门控
        if primary_attention_weights is not None:
            entropy = self._compute_entropy(primary_attention_weights)
        else:
            # 如果没有主注意力权重，使用默认中等熵值
            entropy = torch.ones(batch_size, seq_len, device=device, dtype=dtype) * 1.0
        
        gate = torch.sigmoid(self.gate_w1 * entropy + self.gate_bias)
        gate = self._apply_entropy_veto(gate, entropy)
        
        # 5. 融合输出
        fused = primary_attention_output + gate.unsqueeze(-1) * aux_output
        
        # 6. 更新命中计数（用于监控）
        if self.training is False:
            self._update_hit_counts(attn_weights)
        
        # 7. 诊断信息
        diagnostics = None
        if return_diagnostics:
            top_slots = self._get_top_activated_slots(attn_weights)
            diagnostics = AGADiagnostics(
                entropy=entropy.detach(),
                gate=gate.detach(),
                aux_attn_weights=attn_weights.detach(),
                slot_reliability=reliability.detach(),
                active_slots=self.get_active_slots(),
                top_activated_slots=top_slots,
            )
        
        return fused, diagnostics
    
    def _compute_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        avg_weights = attention_weights.mean(dim=1)
        log_weights = torch.log(avg_weights + 1e-10)
        entropy = -torch.sum(avg_weights * log_weights, dim=-1)
        return entropy
    
    def _apply_entropy_veto(self, gate: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """应用熵否决"""
        gate = torch.where(entropy < self.tau_low, torch.zeros_like(gate), gate)
        gate = torch.where(entropy > self.tau_high, torch.clamp(gate, max=0.8), gate)
        return gate
    
    def _update_hit_counts(self, attn_weights: torch.Tensor):
        """更新槽位命中计数"""
        # 取平均注意力权重，找出被显著激活的槽位
        avg_weights = attn_weights.mean(dim=(0, 1))  # [num_slots]
        threshold = 1.0 / self.num_slots * 2  # 2倍均匀分布
        activated = (avg_weights > threshold).cpu().tolist()
        for i, is_activated in enumerate(activated):
            if is_activated:
                self.slot_hit_counts[i] += 1
    
    def _get_top_activated_slots(self, attn_weights: torch.Tensor, top_k: int = 5) -> List[int]:
        """获取最活跃的槽位"""
        avg_weights = attn_weights.mean(dim=(0, 1))
        _, indices = torch.topk(avg_weights, min(top_k, self.num_slots))
        return indices.cpu().tolist()
    
    # ==================== 知识注入接口 ====================
    
    def inject_knowledge(
        self,
        slot_idx: int,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lu_id: str,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
    ) -> bool:
        """
        注入知识到指定槽位
        
        这是知识进入 AGA 系统的唯一入口。
        
        Args:
            slot_idx: 目标槽位
            key_vector: [bottleneck_dim] 条件编码
            value_vector: [hidden_dim] 修正信号
            lu_id: Learning Unit ID
            lifecycle_state: 初始生命周期状态
        
        Returns:
            是否成功
        """
        if self.training:
            raise RuntimeError("Cannot inject knowledge during training mode")
        
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(f"slot_idx must be in [0, {self.num_slots})")
        
        if key_vector.shape[-1] != self.bottleneck_dim:
            raise ValueError(f"key_vector last dim must be {self.bottleneck_dim}")
        
        if value_vector.shape[-1] != self.hidden_dim:
            raise ValueError(f"value_vector last dim must be {self.hidden_dim}")
        
        # 展平为 1D
        key_vector = key_vector.flatten()[-self.bottleneck_dim:]
        value_vector = value_vector.flatten()[-self.hidden_dim:]
        
        with torch.no_grad():
            self.aux_keys[slot_idx] = key_vector.to(self.aux_keys.device)
            self.aux_values[slot_idx] = value_vector.to(self.aux_values.device)
        
        self.slot_lifecycle[slot_idx] = lifecycle_state
        self.slot_lu_ids[slot_idx] = lu_id
        self.slot_created_at[slot_idx] = datetime.now()
        self.slot_hit_counts[slot_idx] = 0
        self._invalidate_cache()
        
        return True
    
    def find_free_slot(self) -> Optional[int]:
        """查找空闲槽位（已隔离状态）"""
        for i, state in enumerate(self.slot_lifecycle):
            if state == LifecycleState.QUARANTINED:
                return i
        return None
    
    def update_lifecycle(self, slot_idx: int, new_state: LifecycleState):
        """更新生命周期状态"""
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(f"slot_idx out of range")
        self.slot_lifecycle[slot_idx] = new_state
        self._invalidate_cache()
    
    def confirm_slot(self, slot_idx: int):
        """确认槽位（试用期 → 已确认）"""
        self.update_lifecycle(slot_idx, LifecycleState.CONFIRMED)
    
    def deprecate_slot(self, slot_idx: int):
        """弃用槽位"""
        self.update_lifecycle(slot_idx, LifecycleState.DEPRECATED)
    
    def quarantine_slot(self, slot_idx: int):
        """隔离槽位（立即移除影响）"""
        self.update_lifecycle(slot_idx, LifecycleState.QUARANTINED)
        with torch.no_grad():
            self.aux_values[slot_idx].zero_()
        self.slot_lu_ids[slot_idx] = None
    
    def quarantine_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 隔离所有相关槽位"""
        quarantined = []
        for i, lid in enumerate(self.slot_lu_ids):
            if lid == lu_id:
                self.quarantine_slot(i)
                quarantined.append(i)
        return quarantined
    
    # ==================== 查询接口 ====================
    
    def get_active_slots(self) -> int:
        """获取活跃槽位数"""
        return sum(1 for s in self.slot_lifecycle if s != LifecycleState.QUARANTINED)
    
    def get_slot_info(self, slot_idx: int) -> KnowledgeSlotInfo:
        """获取槽位详细信息"""
        state = self.slot_lifecycle[slot_idx]
        return KnowledgeSlotInfo(
            slot_idx=slot_idx,
            lu_id=self.slot_lu_ids[slot_idx],
            lifecycle_state=state,
            reliability=self.LIFECYCLE_RELIABILITY[state],
            key_norm=self.aux_keys[slot_idx].norm().item(),
            value_norm=self.aux_values[slot_idx].norm().item(),
            created_at=self.slot_created_at[slot_idx],
            hit_count=self.slot_hit_counts[slot_idx],
        )
    
    def get_slot_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 查找槽位"""
        return [i for i, lid in enumerate(self.slot_lu_ids) if lid == lu_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        state_counts = {}
        for state in LifecycleState:
            state_counts[state.value] = sum(1 for s in self.slot_lifecycle if s == state)
        
        return {
            'total_slots': self.num_slots,
            'active_slots': self.get_active_slots(),
            'state_distribution': state_counts,
            'avg_key_norm': self.aux_keys.norm(dim=1).mean().item(),
            'avg_value_norm': self.aux_values.norm(dim=1).mean().item(),
            'total_hits': sum(self.slot_hit_counts),
        }


class AGAAugmentedTransformerLayer(nn.Module):
    """
    AGA 增强的 Transformer 层包装器
    
    通过 Monkey Patch 方式替换原始层，实现热插拔。
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        aga_module: AuxiliaryGovernedAttention,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.aga = aga_module
        
        # 检测层类型
        self.has_input_layernorm = hasattr(original_layer, 'input_layernorm')
        self.has_post_attention_layernorm = hasattr(original_layer, 'post_attention_layernorm')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """前向传播，保留原始层结构"""
        # === 注意力块 ===
        residual = hidden_states
        
        if self.has_input_layernorm:
            hidden_states = self.original_layer.input_layernorm(hidden_states)
        
        attn_outputs = self.original_layer.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=True,
            **kwargs
        )
        
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1] if len(attn_outputs) > 1 else None
        
        # AGA 融合
        fused_output, _ = self.aga(
            hidden_states=hidden_states,
            primary_attention_output=attn_output,
            primary_attention_weights=attn_weights,
        )
        
        hidden_states = residual + fused_output
        
        # === MLP 块 ===
        residual = hidden_states
        
        if self.has_post_attention_layernorm:
            hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        
        if hasattr(self.original_layer, 'mlp'):
            hidden_states = self.original_layer.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return (hidden_states,) + attn_outputs[1:]


class AGAManager:
    """
    AGA 管理器
    
    管理多个 AGA 模块（多层挂载）的统一接口。
    """
    
    def __init__(self):
        self.aga_modules: Dict[int, AuxiliaryGovernedAttention] = {}
        self.original_layers: Dict[int, nn.Module] = {}
    
    def attach_to_model(
        self,
        model: nn.Module,
        layer_indices: List[int],
        hidden_dim: int,
        bottleneck_dim: int = 64,
        num_slots: int = 100,
        num_heads: int = 32,
    ) -> Dict[int, AuxiliaryGovernedAttention]:
        """
        将 AGA 挂载到模型的指定层
        
        Args:
            model: HuggingFace 模型
            layer_indices: 要挂载的层索引
            hidden_dim: 隐藏维度
            bottleneck_dim: 瓶颈维度
            num_slots: 每层槽位数
            num_heads: 注意力头数
        
        Returns:
            layer_idx -> AGA 模块的映射
        """
        layers = model.model.layers  # 假设是 LlamaModel 结构
        
        for idx in layer_indices:
            if idx >= len(layers):
                raise ValueError(f"Layer index {idx} out of range")
            
            # 创建 AGA 模块
            aga = AuxiliaryGovernedAttention(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                num_slots=num_slots,
                num_heads=num_heads,
            )
            aga.eval()
            aga.to(next(model.parameters()).device)
            
            # 保存原始层并替换
            self.original_layers[idx] = layers[idx]
            layers[idx] = AGAAugmentedTransformerLayer(layers[idx], aga)
            self.aga_modules[idx] = aga
        
        return self.aga_modules
    
    def detach_from_model(self, model: nn.Module, layer_indices: Optional[List[int]] = None):
        """
        从模型卸载 AGA
        
        Args:
            model: HuggingFace 模型
            layer_indices: 要卸载的层索引，None 表示全部
        """
        layers = model.model.layers
        indices = layer_indices or list(self.original_layers.keys())
        
        for idx in indices:
            if idx in self.original_layers:
                layers[idx] = self.original_layers[idx]
                del self.original_layers[idx]
                del self.aga_modules[idx]
    
    def inject_knowledge_to_all(
        self,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lu_id: str,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
    ) -> Dict[int, int]:
        """
        向所有 AGA 模块注入相同知识
        
        Returns:
            layer_idx -> slot_idx 的映射
        """
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            slot_idx = aga.find_free_slot()
            if slot_idx is not None:
                aga.inject_knowledge(slot_idx, key_vector, value_vector, lu_id, lifecycle_state)
                result[layer_idx] = slot_idx
        return result
    
    def quarantine_by_lu_id(self, lu_id: str) -> Dict[int, List[int]]:
        """按 LU ID 隔离所有相关槽位"""
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            quarantined = aga.quarantine_by_lu_id(lu_id)
            if quarantined:
                result[layer_idx] = quarantined
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取所有 AGA 模块的统计信息"""
        return {
            'attached_layers': list(self.aga_modules.keys()),
            'per_layer_stats': {
                idx: aga.get_statistics() 
                for idx, aga in self.aga_modules.items()
            },
        }

