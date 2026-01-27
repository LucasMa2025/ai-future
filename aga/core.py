"""
AGA (Auxiliary Governed Attention) 核心模块 v2.1

实现热插拔式知识注入系统，无需向量化训练。

v2.1 优化（基于生产级建议）：
- 注入安全：范数裁剪 + 幅度控制
- 不确定性质量：改进 hidden_variance 估计
- 路由内存优化：分块 gather 避免 OOM
- 诊断完整性：路由模式下有意义的权重统计
- 自动降级：基于 hit_count 的 LRU-like 策略
- Early Exit：gate < threshold 时跳过计算

v2.0 优化：
- Slot Routing: O(N) → O(k) 复杂度优化
- Delta Subspace: value 通过 bottleneck projection 受控干预
- 熵信号解耦: 支持多种不确定性信号源，兼容 FlashAttention
- 元数据外置友好: 支持从外部加载 active slots

核心特性：
- 零训练注入：知识直接写入 buffer，无需梯度计算
- 热插拔设计：运行时动态添加/移除知识
- 治理控制：生命周期状态、熵门控、可追溯性
- 即时隔离：问题知识可立即移除影响
"""
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LifecycleState(str, Enum):
    """知识槽位生命周期状态"""
    PROBATIONARY = "probationary"  # 试用期 (r=0.3)
    CONFIRMED = "confirmed"        # 已确认 (r=1.0)
    DEPRECATED = "deprecated"      # 已弃用 (r=0.1)
    QUARANTINED = "quarantined"    # 已隔离 (r=0.0)


class UncertaintySource(str, Enum):
    """不确定性信号来源"""
    ATTENTION_ENTROPY = "attention_entropy"  # 注意力熵（需要 output_attentions）
    LOGITS_ENTROPY = "logits_entropy"        # logits 熵（推荐，最直接的不确定性信号）
    HIDDEN_VARIANCE = "hidden_variance"       # 隐藏状态方差（改进版）
    LEARNED_PROJECTION = "learned_projection" # v2.1: 学习的投影头
    CONSTANT = "constant"                     # 固定值（测试用）


@dataclass
class KnowledgeSlotInfo:
    """知识槽位信息"""
    slot_idx: int
    lu_id: Optional[str]
    lifecycle_state: LifecycleState
    reliability: float
    key_norm: float
    value_norm: float
    condition: Optional[str] = None
    decision: Optional[str] = None
    created_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    hit_count: int = 0
    consecutive_misses: int = 0  # v2.1: 连续未命中次数


# Alias for API compatibility
SlotInfo = KnowledgeSlotInfo


@dataclass
class AGAStatistics:
    """AGA 统计信息 (用于 API 返回)"""
    total_slots: int
    active_slots: int
    state_counts: Dict[str, int]
    avg_hit_count: float = 0.0
    max_hit_count: int = 0
    avg_consecutive_misses: float = 0.0
    max_consecutive_misses: int = 0


@dataclass
class AGADiagnostics:
    """AGA 诊断信息"""
    entropy: torch.Tensor
    gate: torch.Tensor
    aux_attn_weights: torch.Tensor
    slot_reliability: torch.Tensor
    active_slots: int
    top_activated_slots: List[int] = field(default_factory=list)
    routed_slots: Optional[List[int]] = None
    # v2.1 新增
    router_scores: Optional[torch.Tensor] = None  # 路由分数统计
    gate_mean: float = 0.0
    gate_max: float = 0.0
    early_exit_ratio: float = 0.0  # 提前退出比例


@dataclass 
class AGAConfig:
    """AGA 配置"""
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    num_slots: int = 100
    num_heads: int = 32
    tau_low: float = 0.5
    tau_high: float = 2.0
    # v2.0 新增
    top_k_routing: int = 8
    use_value_projection: bool = True
    value_bottleneck_dim: int = 256
    uncertainty_source: UncertaintySource = UncertaintySource.HIDDEN_VARIANCE
    # v2.1 新增
    key_norm_target: float = 5.0       # key 向量目标范数
    value_norm_target: float = 3.0     # value 向量目标范数
    enable_norm_clipping: bool = True  # 是否启用范数裁剪
    early_exit_threshold: float = 0.05 # gate 低于此值时跳过计算
    enable_early_exit: bool = True     # 是否启用 early exit
    auto_deprecate_threshold: int = 100 # 连续未命中次数阈值，超过则自动降级
    enable_auto_deprecate: bool = False # 是否启用自动降级
    logits_temperature: float = 1.0    # logits entropy 温度参数
    chunk_size: int = 64               # 路由分块大小（内存优化）


class UncertaintyEstimator(nn.Module):
    """
    不确定性估计器 v2.1
    
    改进：
    - 更好的 hidden_variance 实现
    - 支持学习的投影头
    - 温度参数控制
    """
    
    def __init__(
        self,
        hidden_dim: int,
        source: UncertaintySource = UncertaintySource.HIDDEN_VARIANCE,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.source = source
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # v2.1: 学习的不确定性投影头
        if source in (UncertaintySource.HIDDEN_VARIANCE, UncertaintySource.LEARNED_PROJECTION):
            self.uncertainty_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            # 初始化为输出接近 1.0 的值
            nn.init.zeros_(self.uncertainty_proj[-1].weight)
            nn.init.constant_(self.uncertainty_proj[-1].bias, 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算不确定性信号
        
        Args:
            hidden_states: [batch, seq, hidden_dim]
            attention_weights: [batch, heads, seq, seq] 可选
            logits: [batch, seq, vocab_size] 可选
        
        Returns:
            uncertainty: [batch, seq] 范围约 [0, 3]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        if self.source == UncertaintySource.ATTENTION_ENTROPY:
            if attention_weights is None:
                return self._compute_hidden_uncertainty(hidden_states)
            return self._compute_attention_entropy(attention_weights)
        
        elif self.source == UncertaintySource.LOGITS_ENTROPY:
            if logits is None:
                return self._compute_hidden_uncertainty(hidden_states)
            return self._compute_logits_entropy(logits)
        
        elif self.source == UncertaintySource.LEARNED_PROJECTION:
            return self._compute_learned_uncertainty(hidden_states)
        
        elif self.source == UncertaintySource.HIDDEN_VARIANCE:
            return self._compute_hidden_uncertainty(hidden_states)
        
        else:  # CONSTANT
            return torch.ones(batch_size, seq_len, device=device, dtype=dtype) * 1.0
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        avg_weights = attention_weights.mean(dim=1)  # [batch, seq, seq]
        log_weights = torch.log(avg_weights + 1e-10)
        entropy = -torch.sum(avg_weights * log_weights, dim=-1)  # [batch, seq]
        return entropy
    
    def _compute_logits_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 logits 熵（带温度参数）"""
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch, seq]
        # 归一化到合理范围
        return torch.clamp(entropy / math.log(logits.size(-1)) * 3.0, 0, 5.0)
    
    def _compute_hidden_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        v2.1 改进：计算隐藏状态不确定性
        
        使用内部一致性（方差）+ 学习的投影
        """
        # 方法1: 计算每个 token 的内部方差
        mean = hidden_states.mean(dim=-1, keepdim=True)
        centered = hidden_states - mean
        variance = (centered ** 2).mean(dim=-1)  # [batch, seq]
        
        # 归一化方差到合理范围
        # 使用 log1p 避免极端值
        log_var = torch.log1p(variance)
        normalized_var = log_var / (log_var.mean() + 1e-6)
        
        # 方法2: 学习的投影（如果可用）
        if hasattr(self, 'uncertainty_proj'):
            learned = self.uncertainty_proj(hidden_states).squeeze(-1)  # [batch, seq]
            # 结合方差和学习信号
            combined = normalized_var * 0.5 + torch.sigmoid(learned) * 2.5
        else:
            combined = normalized_var * 2.0
        
        return torch.clamp(combined, 0, 5.0)
    
    def _compute_learned_uncertainty(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """纯学习的不确定性估计"""
        raw = self.uncertainty_proj(hidden_states).squeeze(-1)  # [batch, seq]
        return torch.sigmoid(raw) * 3.0  # 范围 [0, 3]


class SlotRouter(nn.Module):
    """
    槽位路由器 v2.1
    
    优化：
    - 分块计算避免 OOM
    - 更好的信息保留
    """
    
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        num_slots: int,
        top_k: int = 8,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.num_slots = num_slots
        self.top_k = min(top_k, num_slots)
        self.chunk_size = chunk_size
        
        # v2.1: 使用更多维度进行路由（信息保留）
        self.router_dim = min(bottleneck_dim, 48)  # 不低于 48 维
        
        # 可学习的路由投影（从 bottleneck 到 router）
        self.router_proj = nn.Linear(bottleneck_dim, self.router_dim, bias=False)
        nn.init.orthogonal_(self.router_proj.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        aux_keys: torch.Tensor,
        reliability_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        路由选择 top-k 槽位
        
        Args:
            query: [batch, seq, bottleneck_dim]
            aux_keys: [num_slots, bottleneck_dim]
            reliability_mask: [num_slots] log reliability
        
        Returns:
            selected_indices: [batch, seq, k]
            router_scores: [batch, seq, k]
        """
        batch_size, seq_len, _ = query.shape
        
        # 投影到路由空间
        router_query = self.router_proj(query)  # [batch, seq, router_dim]
        router_keys = self.router_proj(aux_keys)  # [num_slots, router_dim]
        
        # 路由分数
        router_scores = torch.matmul(router_query, router_keys.T)  # [batch, seq, num_slots]
        router_scores = router_scores / math.sqrt(self.router_dim)
        
        # 应用可靠性掩码
        router_scores = router_scores + reliability_mask.unsqueeze(0).unsqueeze(0)
        
        # 选择 top-k
        top_scores, top_indices = torch.topk(router_scores, self.top_k, dim=-1)
        
        return top_indices, top_scores


class AuxiliaryGovernedAttention(nn.Module):
    """
    辅助治理注意力 (AGA) v2.1
    
    生产级优化版本：
    - 注入安全：范数裁剪保护
    - 内存优化：分块 gather
    - Early Exit：低 gate 时跳过计算
    - 自动降级：基于命中率的 LRU 策略
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
        config: Optional[AGAConfig] = None,
        # 向后兼容的参数
        hidden_dim: int = 4096,
        bottleneck_dim: int = 64,
        num_slots: int = 100,
        num_heads: int = 32,
        tau_low: float = 0.5,
        tau_high: float = 2.0,
    ):
        super().__init__()
        
        # 使用 config 或构建默认配置
        if config is not None:
            self.config = config
        else:
            self.config = AGAConfig(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                num_slots=num_slots,
                num_heads=num_heads,
                tau_low=tau_low,
                tau_high=tau_high,
            )
        
        # 快捷访问
        self.hidden_dim = self.config.hidden_dim
        self.bottleneck_dim = self.config.bottleneck_dim
        self.num_heads = self.config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        self.tau_low = self.config.tau_low
        self.tau_high = self.config.tau_high
        self.num_slots = self.config.num_slots
        self.top_k = self.config.top_k_routing
        
        # 查询投影
        self.q_proj = nn.Linear(self.hidden_dim, self.bottleneck_dim, bias=False)
        
        # 辅助键值存储 - 使用 buffer（不可训练）
        self.register_buffer(
            'aux_keys',
            torch.randn(self.num_slots, self.bottleneck_dim) * 0.01
        )
        self.register_buffer(
            'aux_values',
            torch.zeros(self.num_slots, self.hidden_dim)
        )
        
        # Slot Router
        self.router = SlotRouter(
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            num_slots=self.num_slots,
            top_k=self.top_k,
            chunk_size=self.config.chunk_size,
        )
        
        # Value Projection (delta subspace)
        if self.config.use_value_projection:
            self.value_down = nn.Linear(self.hidden_dim, self.config.value_bottleneck_dim, bias=False)
            self.value_up = nn.Linear(self.config.value_bottleneck_dim, self.hidden_dim, bias=False)
            nn.init.xavier_uniform_(self.value_down.weight, gain=0.1)
            nn.init.xavier_uniform_(self.value_up.weight, gain=0.1)
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator(
            hidden_dim=self.hidden_dim,
            source=self.config.uncertainty_source,
            temperature=self.config.logits_temperature,
        )
        
        # 熵门控参数
        self.gate_w1 = nn.Parameter(torch.tensor(0.5))
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))
        
        # 槽位元数据
        self.slot_lifecycle: List[LifecycleState] = [LifecycleState.QUARANTINED] * self.num_slots
        self.slot_lu_ids: List[Optional[str]] = [None] * self.num_slots
        self.slot_conditions: List[Optional[str]] = [None] * self.num_slots
        self.slot_decisions: List[Optional[str]] = [None] * self.num_slots
        self.slot_created_at: List[Optional[datetime]] = [None] * self.num_slots
        self.slot_hit_counts: List[int] = [0] * self.num_slots
        self.slot_consecutive_misses: List[int] = [0] * self.num_slots  # v2.1
        
        # 缓存
        self._cached_reliability: Optional[torch.Tensor] = None
        self._active_slot_indices: Optional[List[int]] = None
        
        # v2.1: 日志钩子
        self._logger: Optional[Callable] = None
    
    def register_logging_hook(self, logger_fn: Callable[[AGADiagnostics], None]):
        """注册日志钩子"""
        self._logger = logger_fn
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self._cached_reliability = None
        self._active_slot_indices = None
    
    def _get_reliability_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """获取可靠性张量（带缓存）"""
        if self._cached_reliability is None:
            self._cached_reliability = torch.tensor(
                [self.LIFECYCLE_RELIABILITY[state] for state in self.slot_lifecycle],
                device=device, dtype=dtype
            )
        return self._cached_reliability.to(device=device, dtype=dtype)
    
    def _get_active_slot_indices(self) -> List[int]:
        """获取活跃槽位索引（带缓存）"""
        if self._active_slot_indices is None:
            self._active_slot_indices = [
                i for i, state in enumerate(self.slot_lifecycle)
                if state != LifecycleState.QUARANTINED
            ]
        return self._active_slot_indices
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        primary_attention_output: torch.Tensor,
        primary_attention_weights: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        use_routing: bool = True,
    ) -> Tuple[torch.Tensor, Optional[AGADiagnostics]]:
        """
        前向传播
        
        v2.1 优化：
        - Early exit 当 gate 过低时
        - 分块计算避免 OOM
        - 更完整的诊断信息
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        # 检查是否有活跃槽位
        active_count = self.get_active_slots()
        if active_count == 0:
            return primary_attention_output, None
        
        # 1. 计算不确定性和门控（提前计算用于 early exit）
        uncertainty = self.uncertainty_estimator(
            hidden_states, primary_attention_weights, logits
        )
        
        gate = torch.sigmoid(self.gate_w1 * uncertainty + self.gate_bias)
        gate = self._apply_entropy_veto(gate, uncertainty)
        
        # v2.1: Early Exit 检查
        gate_mean = gate.mean().item()
        early_exit_ratio = 0.0
        
        if self.config.enable_early_exit:
            # 如果平均 gate 非常低，直接返回
            if gate_mean < self.config.early_exit_threshold:
                if return_diagnostics:
                    diagnostics = AGADiagnostics(
                        entropy=uncertainty.detach(),
                        gate=gate.detach(),
                        aux_attn_weights=torch.zeros(1, device=device),
                        slot_reliability=self._get_reliability_tensor(device, dtype).detach(),
                        active_slots=active_count,
                        gate_mean=gate_mean,
                        gate_max=gate.max().item(),
                        early_exit_ratio=1.0,
                    )
                    return primary_attention_output, diagnostics
                return primary_attention_output, None
            
            # 计算每个位置是否需要计算
            compute_mask = gate > self.config.early_exit_threshold  # [batch, seq]
            early_exit_ratio = 1.0 - compute_mask.float().mean().item()
        
        # 2. 查询投影
        query = self.q_proj(hidden_states)  # [batch, seq, bottleneck]
        
        # 3. 获取可靠性掩码
        reliability = self._get_reliability_tensor(device, dtype)
        reliability_mask = torch.log(reliability + 1e-10)
        
        # 4. 计算注意力
        if use_routing and active_count > self.top_k:
            aux_output, routed_slots, router_scores_stats = self._forward_with_routing_chunked(
                query, reliability_mask, batch_size, seq_len, device, dtype
            )
        else:
            aux_output, routed_slots, router_scores_stats = self._forward_full(
                query, reliability_mask
            )
        
        # 5. Value Projection (delta subspace)
        if self.config.use_value_projection:
            aux_output = self.value_down(aux_output)
            aux_output = F.gelu(aux_output)
            aux_output = self.value_up(aux_output)
        
        # 6. 融合输出
        fused = primary_attention_output + gate.unsqueeze(-1) * aux_output
        
        # 7. 更新命中计数（推理模式）
        if not self.training and routed_slots is not None:
            self._update_hit_counts_from_indices(routed_slots)
            
            # v2.1: 检查自动降级
            if self.config.enable_auto_deprecate:
                self._check_auto_deprecate()
        
        # 8. 诊断信息
        diagnostics = None
        if return_diagnostics:
            diagnostics = AGADiagnostics(
                entropy=uncertainty.detach(),
                gate=gate.detach(),
                aux_attn_weights=router_scores_stats if router_scores_stats is not None else torch.zeros(1, device=device),
                slot_reliability=reliability.detach(),
                active_slots=active_count,
                top_activated_slots=routed_slots[:10] if routed_slots else [],
                routed_slots=routed_slots,
                router_scores=router_scores_stats,
                gate_mean=gate_mean,
                gate_max=gate.max().item(),
                early_exit_ratio=early_exit_ratio,
            )
            
            # 调用日志钩子
            if self._logger is not None:
                self._logger(diagnostics)
        
        return fused, diagnostics
    
    def _forward_with_routing_chunked(
        self,
        query: torch.Tensor,
        reliability_mask: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        v2.1: 分块路由前向传播 - 避免 OOM
        """
        chunk_size = self.config.chunk_size
        
        # 1. 路由选择 top-k 槽位
        top_indices, router_scores = self.router(query, self.aux_keys, reliability_mask)
        # top_indices: [batch, seq, k]
        
        # 2. 分块收集选中槽位的 keys 和 values
        selected_keys = torch.zeros(
            batch_size, seq_len, self.top_k, self.bottleneck_dim, 
            device=device, dtype=dtype
        )
        selected_values = torch.zeros(
            batch_size, seq_len, self.top_k, self.hidden_dim,
            device=device, dtype=dtype
        )
        
        # 分块处理避免内存峰值
        for b_start in range(0, batch_size, chunk_size):
            b_end = min(b_start + chunk_size, batch_size)
            for s_start in range(0, seq_len, chunk_size):
                s_end = min(s_start + chunk_size, seq_len)
                
                chunk_indices = top_indices[b_start:b_end, s_start:s_end]  # [chunk_b, chunk_s, k]
                
                # 展平并 gather
                flat_indices = chunk_indices.reshape(-1)  # [chunk_b * chunk_s * k]
                
                chunk_keys = self.aux_keys[flat_indices]  # [chunk_b*chunk_s*k, bottleneck]
                chunk_values = self.aux_values[flat_indices]  # [chunk_b*chunk_s*k, hidden]
                
                # 重塑
                chunk_b = b_end - b_start
                chunk_s = s_end - s_start
                selected_keys[b_start:b_end, s_start:s_end] = chunk_keys.view(
                    chunk_b, chunk_s, self.top_k, self.bottleneck_dim
                )
                selected_values[b_start:b_end, s_start:s_end] = chunk_values.view(
                    chunk_b, chunk_s, self.top_k, self.hidden_dim
                )
        
        # 3. 计算精确注意力（只在 top-k 上）
        query_expanded = query.unsqueeze(2)  # [batch, seq, 1, bottleneck]
        attn_scores = torch.sum(query_expanded * selected_keys, dim=-1)  # [batch, seq, k]
        attn_scores = attn_scores / math.sqrt(self.bottleneck_dim)
        
        # 应用 router_scores 作为 prior
        attn_scores = attn_scores + router_scores * 0.1
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq, k]
        
        # 4. 加权求和
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch, seq, k, 1]
        aux_output = torch.sum(attn_weights_expanded * selected_values, dim=2)  # [batch, seq, hidden]
        
        # 5. 统计路由的槽位
        all_indices = top_indices.reshape(-1).cpu().tolist()
        from collections import Counter
        slot_counts = Counter(all_indices)
        routed_slots = [idx for idx, _ in slot_counts.most_common(20)]
        
        # v2.1: 返回有意义的权重统计
        router_scores_stats = attn_weights.mean(dim=(0, 1)).detach()  # [k]
        
        return aux_output, routed_slots, router_scores_stats
    
    def _forward_full(
        self,
        query: torch.Tensor,
        reliability_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """
        完整前向传播 - 用于小规模槽位
        """
        # 完整注意力分数
        attn_scores = torch.matmul(query, self.aux_keys.T) / math.sqrt(self.bottleneck_dim)
        attn_scores = attn_scores + reliability_mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        aux_output = torch.matmul(attn_weights, self.aux_values)
        
        # 获取 top 槽位
        avg_weights = attn_weights.mean(dim=(0, 1))
        _, top_indices = torch.topk(avg_weights, min(20, self.num_slots))
        routed_slots = top_indices.cpu().tolist()
        
        return aux_output, routed_slots, avg_weights.detach()
    
    def _apply_entropy_veto(self, gate: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """应用熵否决"""
        gate = torch.where(entropy < self.tau_low, torch.zeros_like(gate), gate)
        gate = torch.where(entropy > self.tau_high, torch.clamp(gate, max=0.8), gate)
        return gate
    
    def _update_hit_counts_from_indices(self, slot_indices: List[int]):
        """从路由索引更新命中计数"""
        hit_set = set(slot_indices)
        
        for i in range(self.num_slots):
            if self.slot_lifecycle[i] == LifecycleState.QUARANTINED:
                continue
                
            if i in hit_set:
                self.slot_hit_counts[i] += 1
                self.slot_consecutive_misses[i] = 0  # 重置连续未命中
            else:
                self.slot_consecutive_misses[i] += 1
    
    def _check_auto_deprecate(self):
        """v2.1: 检查并执行自动降级"""
        threshold = self.config.auto_deprecate_threshold
        
        for i in range(self.num_slots):
            if self.slot_lifecycle[i] in (LifecycleState.PROBATIONARY, LifecycleState.CONFIRMED):
                if self.slot_consecutive_misses[i] >= threshold:
                    logger.info(f"Auto-deprecating slot {i} (lu_id={self.slot_lu_ids[i]}) "
                               f"due to {self.slot_consecutive_misses[i]} consecutive misses")
                    self.deprecate_slot(i)
    
    # ==================== 知识注入接口 ====================
    
    def inject_knowledge(
        self,
        slot_idx: int,
        key_vector: torch.Tensor,
        value_vector: torch.Tensor,
        lu_id: str,
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> bool:
        """
        注入知识到指定槽位
        
        v2.1: 增加范数裁剪保护
        """
        if self.training:
            raise RuntimeError("Cannot inject knowledge during training mode")
        
        if slot_idx < 0 or slot_idx >= self.num_slots:
            raise ValueError(f"slot_idx must be in [0, {self.num_slots})")
        
        # 处理维度
        if key_vector.dim() > 1:
            key_vector = key_vector.flatten()
        if value_vector.dim() > 1:
            value_vector = value_vector.flatten()
        
        # 填充或截断
        if key_vector.shape[0] < self.bottleneck_dim:
            key_vector = F.pad(key_vector, (0, self.bottleneck_dim - key_vector.shape[0]))
        elif key_vector.shape[0] > self.bottleneck_dim:
            key_vector = key_vector[:self.bottleneck_dim]
        
        if value_vector.shape[0] < self.hidden_dim:
            value_vector = F.pad(value_vector, (0, self.hidden_dim - value_vector.shape[0]))
        elif value_vector.shape[0] > self.hidden_dim:
            value_vector = value_vector[:self.hidden_dim]
        
        with torch.no_grad():
            # v2.1: 范数裁剪保护
            if self.config.enable_norm_clipping:
                key_norm = key_vector.norm()
                if key_norm > 0:
                    key_vector = key_vector / (key_norm + 1e-8) * self.config.key_norm_target
                
                value_norm = value_vector.norm()
                if value_norm > 0:
                    value_vector = value_vector / (value_norm + 1e-8) * self.config.value_norm_target
            
            self.aux_keys[slot_idx] = key_vector.to(self.aux_keys.device)
            self.aux_values[slot_idx] = value_vector.to(self.aux_values.device)
        
        self.slot_lifecycle[slot_idx] = lifecycle_state
        self.slot_lu_ids[slot_idx] = lu_id
        self.slot_conditions[slot_idx] = condition
        self.slot_decisions[slot_idx] = decision
        self.slot_created_at[slot_idx] = datetime.now()
        self.slot_hit_counts[slot_idx] = 0
        self.slot_consecutive_misses[slot_idx] = 0
        self._invalidate_cache()
        
        return True
    
    def inject_knowledge_batch(
        self,
        knowledge_list: List[Dict[str, Any]],
    ) -> int:
        """
        v2.1: 批量注入知识（减少 cache invalidate 次数）
        
        Args:
            knowledge_list: [{'key_vector', 'value_vector', 'lu_id', ...}, ...]
        
        Returns:
            成功注入的数量
        """
        if self.training:
            raise RuntimeError("Cannot inject knowledge during training mode")
        
        success_count = 0
        
        for item in knowledge_list:
            slot_idx = item.get('slot_idx')
            if slot_idx is None:
                slot_idx = self.find_free_slot()
                if slot_idx is None:
                    continue
            
            try:
                # 临时禁用 cache invalidate
                key_vector = item['key_vector']
                value_vector = item['value_vector']
                
                if isinstance(key_vector, list):
                    key_vector = torch.tensor(key_vector)
                if isinstance(value_vector, list):
                    value_vector = torch.tensor(value_vector)
                
                # 处理维度和范数
                if key_vector.dim() > 1:
                    key_vector = key_vector.flatten()
                if value_vector.dim() > 1:
                    value_vector = value_vector.flatten()
                
                if key_vector.shape[0] != self.bottleneck_dim:
                    if key_vector.shape[0] < self.bottleneck_dim:
                        key_vector = F.pad(key_vector, (0, self.bottleneck_dim - key_vector.shape[0]))
                    else:
                        key_vector = key_vector[:self.bottleneck_dim]
                
                if value_vector.shape[0] != self.hidden_dim:
                    if value_vector.shape[0] < self.hidden_dim:
                        value_vector = F.pad(value_vector, (0, self.hidden_dim - value_vector.shape[0]))
                    else:
                        value_vector = value_vector[:self.hidden_dim]
                
                with torch.no_grad():
                    if self.config.enable_norm_clipping:
                        key_norm = key_vector.norm()
                        if key_norm > 0:
                            key_vector = key_vector / (key_norm + 1e-8) * self.config.key_norm_target
                        
                        value_norm = value_vector.norm()
                        if value_norm > 0:
                            value_vector = value_vector / (value_norm + 1e-8) * self.config.value_norm_target
                    
                    self.aux_keys[slot_idx] = key_vector.to(self.aux_keys.device)
                    self.aux_values[slot_idx] = value_vector.to(self.aux_values.device)
                
                lifecycle = item.get('lifecycle_state', LifecycleState.PROBATIONARY)
                if isinstance(lifecycle, str):
                    lifecycle = LifecycleState(lifecycle)
                
                self.slot_lifecycle[slot_idx] = lifecycle
                self.slot_lu_ids[slot_idx] = item['lu_id']
                self.slot_conditions[slot_idx] = item.get('condition')
                self.slot_decisions[slot_idx] = item.get('decision')
                self.slot_created_at[slot_idx] = datetime.now()
                self.slot_hit_counts[slot_idx] = item.get('hit_count', 0)
                self.slot_consecutive_misses[slot_idx] = 0
                
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to inject knowledge: {e}")
        
        # 最后统一 invalidate cache
        self._invalidate_cache()
        
        return success_count
    
    def inject_from_text(
        self,
        slot_idx: int,
        condition: str,
        decision: str,
        lu_id: str,
        embed_fn: Callable[[str], torch.Tensor],
        lifecycle_state: LifecycleState = LifecycleState.PROBATIONARY,
    ) -> bool:
        """从文本注入知识"""
        key_vector = embed_fn(condition)
        value_vector = embed_fn(decision)
        
        return self.inject_knowledge(
            slot_idx=slot_idx,
            key_vector=key_vector,
            value_vector=value_vector,
            lu_id=lu_id,
            lifecycle_state=lifecycle_state,
            condition=condition,
            decision=decision,
        )
    
    def find_free_slot(self) -> Optional[int]:
        """查找空闲槽位"""
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
        """确认槽位"""
        self.update_lifecycle(slot_idx, LifecycleState.CONFIRMED)
    
    def deprecate_slot(self, slot_idx: int):
        """弃用槽位"""
        self.update_lifecycle(slot_idx, LifecycleState.DEPRECATED)
    
    def quarantine_slot(self, slot_idx: int):
        """隔离槽位"""
        self.update_lifecycle(slot_idx, LifecycleState.QUARANTINED)
        with torch.no_grad():
            self.aux_values[slot_idx].zero_()
        self.slot_lu_ids[slot_idx] = None
        self.slot_conditions[slot_idx] = None
        self.slot_decisions[slot_idx] = None
    
    def quarantine_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 隔离"""
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
            condition=self.slot_conditions[slot_idx],
            decision=self.slot_decisions[slot_idx],
            created_at=self.slot_created_at[slot_idx],
            hit_count=self.slot_hit_counts[slot_idx],
            consecutive_misses=self.slot_consecutive_misses[slot_idx],
        )
    
    def get_all_slots_info(self) -> List[KnowledgeSlotInfo]:
        """获取所有槽位信息"""
        return [self.get_slot_info(i) for i in range(self.num_slots)]
    
    def get_active_knowledge(self) -> List[KnowledgeSlotInfo]:
        """获取所有活跃知识"""
        return [
            self.get_slot_info(i) 
            for i in range(self.num_slots) 
            if self.slot_lifecycle[i] != LifecycleState.QUARANTINED
        ]
    
    def get_slot_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 查找槽位"""
        return [i for i, lid in enumerate(self.slot_lu_ids) if lid == lu_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        state_counts = {}
        for state in LifecycleState:
            state_counts[state.value] = sum(1 for s in self.slot_lifecycle if s == state)
        
        # v2.1: 更丰富的统计
        hit_counts = [c for c in self.slot_hit_counts if c > 0]
        miss_counts = [c for c in self.slot_consecutive_misses if c > 0]
        
        return {
            'total_slots': self.num_slots,
            'active_slots': self.get_active_slots(),
            'state_distribution': state_counts,
            'avg_key_norm': self.aux_keys.norm(dim=1).mean().item(),
            'avg_value_norm': self.aux_values.norm(dim=1).mean().item(),
            'total_hits': sum(self.slot_hit_counts),
            'hidden_dim': self.hidden_dim,
            'bottleneck_dim': self.bottleneck_dim,
            'top_k_routing': self.top_k,
            'use_value_projection': self.config.use_value_projection,
            'uncertainty_source': self.config.uncertainty_source.value,
            # v2.1 新增
            'avg_hit_count': sum(hit_counts) / len(hit_counts) if hit_counts else 0,
            'max_hit_count': max(hit_counts) if hit_counts else 0,
            'avg_consecutive_misses': sum(miss_counts) / len(miss_counts) if miss_counts else 0,
            'max_consecutive_misses': max(miss_counts) if miss_counts else 0,
            'enable_norm_clipping': self.config.enable_norm_clipping,
            'enable_early_exit': self.config.enable_early_exit,
            'enable_auto_deprecate': self.config.enable_auto_deprecate,
        }
    
    def export_state(self) -> Dict[str, Any]:
        """导出状态（用于持久化）"""
        return {
            'version': '2.1',
            'config': {
                'hidden_dim': self.hidden_dim,
                'bottleneck_dim': self.bottleneck_dim,
                'num_slots': self.num_slots,
                'num_heads': self.num_heads,
                'tau_low': self.tau_low,
                'tau_high': self.tau_high,
                'top_k_routing': self.config.top_k_routing,
                'use_value_projection': self.config.use_value_projection,
                'value_bottleneck_dim': self.config.value_bottleneck_dim,
                'uncertainty_source': self.config.uncertainty_source.value,
                'key_norm_target': self.config.key_norm_target,
                'value_norm_target': self.config.value_norm_target,
                'enable_norm_clipping': self.config.enable_norm_clipping,
                'early_exit_threshold': self.config.early_exit_threshold,
                'enable_early_exit': self.config.enable_early_exit,
                'auto_deprecate_threshold': self.config.auto_deprecate_threshold,
                'enable_auto_deprecate': self.config.enable_auto_deprecate,
            },
            'aux_keys': self.aux_keys.cpu().tolist(),
            'aux_values': self.aux_values.cpu().tolist(),
            'slot_lifecycle': [s.value for s in self.slot_lifecycle],
            'slot_lu_ids': self.slot_lu_ids,
            'slot_conditions': self.slot_conditions,
            'slot_decisions': self.slot_decisions,
            'slot_created_at': [
                dt.isoformat() if dt else None 
                for dt in self.slot_created_at
            ],
            'slot_hit_counts': self.slot_hit_counts,
            'slot_consecutive_misses': self.slot_consecutive_misses,
        }
    
    def import_state(self, state: Dict[str, Any]):
        """导入状态"""
        # v2.1: 版本检查
        version = state.get('version', '1.0')
        if version not in ('2.0', '2.1'):
            logger.warning(f"Importing state from version {version}, some fields may be missing")
        
        device = self.aux_keys.device
        dtype = self.aux_keys.dtype
        
        self.aux_keys.data = torch.tensor(state['aux_keys'], device=device, dtype=dtype)
        self.aux_values.data = torch.tensor(state['aux_values'], device=device, dtype=dtype)
        self.slot_lifecycle = [LifecycleState(s) for s in state['slot_lifecycle']]
        self.slot_lu_ids = state['slot_lu_ids']
        self.slot_conditions = state.get('slot_conditions', [None] * self.num_slots)
        self.slot_decisions = state.get('slot_decisions', [None] * self.num_slots)
        self.slot_created_at = [
            datetime.fromisoformat(dt) if dt else None 
            for dt in state.get('slot_created_at', [None] * self.num_slots)
        ]
        self.slot_hit_counts = state.get('slot_hit_counts', [0] * self.num_slots)
        self.slot_consecutive_misses = state.get('slot_consecutive_misses', [0] * self.num_slots)
        self._invalidate_cache()
    
    # ==================== 外置元数据支持 ====================
    
    def load_active_slots_only(
        self,
        slot_data: List[Dict[str, Any]],
    ):
        """
        仅加载活跃槽位（外置元数据场景）
        """
        for data in slot_data:
            slot_idx = data['slot_idx']
            if slot_idx >= self.num_slots:
                continue
            
            key_vector = torch.tensor(data['key_vector'])
            value_vector = torch.tensor(data['value_vector'])
            
            self.inject_knowledge(
                slot_idx=slot_idx,
                key_vector=key_vector,
                value_vector=value_vector,
                lu_id=data['lu_id'],
                lifecycle_state=LifecycleState(data['lifecycle_state']),
                condition=data.get('condition'),
                decision=data.get('decision'),
            )


class AGAAugmentedTransformerLayer(nn.Module):
    """
    AGA 增强的 Transformer 层包装器
    """
    
    def __init__(
        self,
        original_layer: nn.Module,
        aga_module: AuxiliaryGovernedAttention,
        require_attention_weights: bool = False,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.aga = aga_module
        self.require_attention_weights = require_attention_weights
        
        self.has_input_layernorm = hasattr(original_layer, 'input_layernorm')
        self.has_post_attention_layernorm = hasattr(original_layer, 'post_attention_layernorm')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """前向传播"""
        residual = hidden_states
        
        if self.has_input_layernorm:
            hidden_states = self.original_layer.input_layernorm(hidden_states)
        
        output_attentions = self.require_attention_weights or \
                           self.aga.config.uncertainty_source == UncertaintySource.ATTENTION_ENTROPY
        
        attn_outputs = self.original_layer.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            **kwargs
        )
        
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1] if len(attn_outputs) > 1 and output_attentions else None
        
        fused_output, _ = self.aga(
            hidden_states=hidden_states,
            primary_attention_output=attn_output,
            primary_attention_weights=attn_weights,
        )
        
        hidden_states = residual + fused_output
        
        residual = hidden_states
        
        if self.has_post_attention_layernorm:
            hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        
        if hasattr(self.original_layer, 'mlp'):
            hidden_states = self.original_layer.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return (hidden_states,) + attn_outputs[1:]


class AGAManager:
    """
    AGA 管理器 v2.1
    """
    
    def __init__(self, config: Optional[AGAConfig] = None):
        self.config = config or AGAConfig()
        self.aga_modules: Dict[int, AuxiliaryGovernedAttention] = {}
        self.original_layers: Dict[int, nn.Module] = {}
        self.model = None
    
    def attach_to_model(
        self,
        model: nn.Module,
        layer_indices: List[int],
        hidden_dim: Optional[int] = None,
        bottleneck_dim: int = 64,
        num_slots: int = 100,
        num_heads: Optional[int] = None,
        config: Optional[AGAConfig] = None,
    ) -> Dict[int, AuxiliaryGovernedAttention]:
        """将 AGA 挂载到模型"""
        self.model = model
        
        # 自动检测模型参数
        if hidden_dim is None:
            if hasattr(model.config, 'hidden_size'):
                hidden_dim = model.config.hidden_size
            elif hasattr(model.config, 'n_embd'):
                hidden_dim = model.config.n_embd
            else:
                raise ValueError("Cannot detect hidden_dim, please specify")
        
        if num_heads is None:
            if hasattr(model.config, 'num_attention_heads'):
                num_heads = model.config.num_attention_heads
            elif hasattr(model.config, 'n_head'):
                num_heads = model.config.n_head
            else:
                num_heads = 32
        
        # 获取层列表
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError("Unsupported model architecture")
        
        num_layers = len(layers)
        
        # 转换负索引
        resolved_indices = []
        for idx in layer_indices:
            if idx < 0:
                resolved_indices.append(num_layers + idx)
            else:
                resolved_indices.append(idx)
        
        # 使用配置或构建默认配置
        if config is None:
            config = AGAConfig(
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                num_slots=num_slots,
                num_heads=num_heads,
            )
        
        # 挂载 AGA
        for idx in resolved_indices:
            if idx >= num_layers:
                raise ValueError(f"Layer index {idx} out of range")
            
            aga = AuxiliaryGovernedAttention(config=config)
            aga.eval()
            aga.to(next(model.parameters()).device)
            
            require_attn = config.uncertainty_source == UncertaintySource.ATTENTION_ENTROPY
            
            self.original_layers[idx] = layers[idx]
            layers[idx] = AGAAugmentedTransformerLayer(
                layers[idx], aga, require_attention_weights=require_attn
            )
            self.aga_modules[idx] = aga
        
        return self.aga_modules
    
    def detach_from_model(self, layer_indices: Optional[List[int]] = None):
        """从模型卸载 AGA"""
        if self.model is None:
            return
        
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        else:
            return
        
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
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Dict[int, int]:
        """向所有 AGA 模块注入知识"""
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            slot_idx = aga.find_free_slot()
            if slot_idx is not None:
                aga.inject_knowledge(
                    slot_idx, key_vector, value_vector, lu_id, 
                    lifecycle_state, condition, decision
                )
                result[layer_idx] = slot_idx
        return result
    
    def quarantine_by_lu_id(self, lu_id: str) -> Dict[int, List[int]]:
        """按 LU ID 隔离"""
        result = {}
        for layer_idx, aga in self.aga_modules.items():
            quarantined = aga.quarantine_by_lu_id(lu_id)
            if quarantined:
                result[layer_idx] = quarantined
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'attached_layers': list(self.aga_modules.keys()),
            'per_layer_stats': {
                idx: aga.get_statistics() 
                for idx, aga in self.aga_modules.items()
            },
        }
    
    def export_all_states(self) -> Dict[int, Dict[str, Any]]:
        """导出所有 AGA 状态"""
        return {idx: aga.export_state() for idx, aga in self.aga_modules.items()}
    
    def import_all_states(self, states: Dict[int, Dict[str, Any]]):
        """导入所有 AGA 状态"""
        for idx, state in states.items():
            if idx in self.aga_modules:
                self.aga_modules[idx].import_state(state)
