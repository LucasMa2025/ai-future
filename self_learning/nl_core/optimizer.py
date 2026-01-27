"""
表达性优化器 (Expressive Optimizer)

基于 NL 论文的可学习优化器实现。
优化器本身作为记忆模块，可以学习和适应。
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math


@dataclass
class OptimizerState:
    """优化器状态"""
    step: int = 0
    momentum: Dict[str, float] = field(default_factory=dict)
    velocity: Dict[str, float] = field(default_factory=dict)
    accumulated_gradients: Dict[str, float] = field(default_factory=dict)
    
    # 自适应参数
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "momentum_keys": list(self.momentum.keys()),
            "velocity_keys": list(self.velocity.keys()),
        }


class ExpressiveOptimizer(ABC):
    """
    表达性优化器基类
    
    实现 NL 论文的核心创新：
    - 优化器内部状态作为可学习的记忆模块
    - 支持自修改学习策略
    - 与 CMS 集成的多频率更新
    """
    
    @abstractmethod
    def compute_update(
        self,
        gradients: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算参数更新"""
        pass
    
    @abstractmethod
    def update_strategy(self, meta_context: Dict[str, Any]) -> None:
        """更新优化策略（自修改）"""
        pass
    
    @abstractmethod
    def get_state(self) -> OptimizerState:
        """获取优化器状态"""
        pass
    
    @abstractmethod
    def restore_state(self, state: OptimizerState) -> None:
        """恢复优化器状态"""
        pass


class DeepMomentumOptimizer(ExpressiveOptimizer):
    """
    深度动量优化器
    
    基于 NL 论文的 "Deep Momentum Gradient Descent" 概念。
    momentum 本身是一个可学习的模块。
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        momentum_depth: int = 3  # 动量的"深度"
    ):
        self.state = OptimizerState(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
        )
        self.momentum_depth = momentum_depth
        
        # 多层动量（NL 创新）
        self.deep_momentum: List[Dict[str, float]] = [
            {} for _ in range(momentum_depth)
        ]
        
        # 自适应学习率因子
        self.lr_factors: Dict[str, float] = {}
    
    def compute_update(
        self,
        gradients: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算参数更新
        
        实现深度动量：每层动量都参与更新计算
        """
        self.state.step += 1
        updates = {}
        
        for param_name, gradient in gradients.items():
            if not isinstance(gradient, (int, float)):
                continue
            
            # 初始化状态
            if param_name not in self.state.momentum:
                self.state.momentum[param_name] = 0.0
                self.state.velocity[param_name] = 0.0
                for i in range(self.momentum_depth):
                    self.deep_momentum[i][param_name] = 0.0
                self.lr_factors[param_name] = 1.0
            
            # 更新多层动量
            for i in range(self.momentum_depth):
                decay = self.state.beta1 ** (i + 1)
                self.deep_momentum[i][param_name] = (
                    decay * self.deep_momentum[i][param_name] +
                    (1 - decay) * gradient
                )
            
            # 聚合多层动量
            aggregated_momentum = sum(
                self.deep_momentum[i][param_name] / (i + 1)
                for i in range(self.momentum_depth)
            ) / self.momentum_depth
            
            # 更新速度 (Adam-like)
            self.state.velocity[param_name] = (
                self.state.beta2 * self.state.velocity[param_name] +
                (1 - self.state.beta2) * (gradient ** 2)
            )
            
            # 偏差校正
            m_hat = aggregated_momentum / (1 - self.state.beta1 ** self.state.step)
            v_hat = self.state.velocity[param_name] / (1 - self.state.beta2 ** self.state.step)
            
            # 计算更新
            lr = self.state.learning_rate * self.lr_factors[param_name]
            update = -lr * m_hat / (math.sqrt(v_hat) + self.state.epsilon)
            
            updates[param_name] = update
        
        return updates
    
    def update_strategy(self, meta_context: Dict[str, Any]) -> None:
        """
        更新优化策略
        
        实现 NL 论文的 "Self-Modifying Learning Module"：
        优化器学习如何修改自身的更新规则
        """
        # 基于 meta_context 调整策略参数
        
        # 1. 基于损失变化调整学习率
        loss_history = meta_context.get("loss_history", [])
        if len(loss_history) >= 2:
            loss_change = loss_history[-1] - loss_history[-2]
            if loss_change > 0:
                # 损失增加，降低学习率
                self.state.learning_rate *= 0.95
            else:
                # 损失降低，可以略微增加学习率
                self.state.learning_rate *= 1.01
        
        # 限制学习率范围
        self.state.learning_rate = max(1e-6, min(0.1, self.state.learning_rate))
        
        # 2. 基于梯度统计调整参数级学习率因子
        gradient_stats = meta_context.get("gradient_stats", {})
        for param_name, stats in gradient_stats.items():
            variance = stats.get("variance", 1.0)
            if param_name in self.lr_factors:
                # 高方差参数使用较小的学习率
                self.lr_factors[param_name] = 1.0 / (1.0 + math.sqrt(variance))
        
        # 3. 调整动量参数
        stability = meta_context.get("stability_score", 0.5)
        if stability < 0.3:
            # 不稳定时增加动量
            self.state.beta1 = min(0.99, self.state.beta1 * 1.01)
        elif stability > 0.7:
            # 稳定时减少动量以加快收敛
            self.state.beta1 = max(0.8, self.state.beta1 * 0.99)
    
    def get_state(self) -> OptimizerState:
        """获取优化器状态"""
        return self.state
    
    def restore_state(self, state: OptimizerState) -> None:
        """恢复优化器状态"""
        self.state = state
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "step": self.state.step,
            "learning_rate": self.state.learning_rate,
            "beta1": self.state.beta1,
            "beta2": self.state.beta2,
            "momentum_depth": self.momentum_depth,
            "tracked_params": len(self.state.momentum),
            "lr_factors_range": (
                min(self.lr_factors.values()) if self.lr_factors else None,
                max(self.lr_factors.values()) if self.lr_factors else None,
            ),
        }


class MetaLearningOptimizer(ExpressiveOptimizer):
    """
    元学习优化器
    
    实现更高级的自修改能力：
    - 学习优化策略的优化策略
    - 基于任务特征自适应
    """
    
    def __init__(
        self,
        inner_optimizer: ExpressiveOptimizer,
        meta_learning_rate: float = 0.01
    ):
        self.inner = inner_optimizer
        self.meta_lr = meta_learning_rate
        
        # 元学习状态
        self.meta_state = {
            "task_features": [],
            "strategy_history": [],
            "performance_history": [],
        }
    
    def compute_update(
        self,
        gradients: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """委托给内部优化器"""
        return self.inner.compute_update(gradients, context)
    
    def update_strategy(self, meta_context: Dict[str, Any]) -> None:
        """
        元级策略更新
        
        学习如何更好地调整内部优化器
        """
        # 记录任务特征
        task_features = {
            "loss": meta_context.get("loss"),
            "gradient_norm": meta_context.get("gradient_norm"),
            "step": self.inner.get_state().step,
        }
        self.meta_state["task_features"].append(task_features)
        
        # 基于历史调整元学习率
        if len(self.meta_state["performance_history"]) >= 10:
            recent_performance = self.meta_state["performance_history"][-10:]
            if all(p > recent_performance[0] for p in recent_performance[1:]):
                # 持续恶化，减少元学习率
                self.meta_lr *= 0.9
            elif all(p < recent_performance[0] for p in recent_performance[1:]):
                # 持续改进，可以增加元学习率
                self.meta_lr *= 1.1
        
        self.meta_lr = max(1e-4, min(0.1, self.meta_lr))
        
        # 调用内部优化器的策略更新
        self.inner.update_strategy(meta_context)
        
        # 记录性能
        if "loss" in meta_context:
            self.meta_state["performance_history"].append(meta_context["loss"])
    
    def get_state(self) -> OptimizerState:
        return self.inner.get_state()
    
    def restore_state(self, state: OptimizerState) -> None:
        self.inner.restore_state(state)
    
    def get_meta_statistics(self) -> Dict[str, Any]:
        """获取元学习统计"""
        return {
            "meta_learning_rate": self.meta_lr,
            "task_features_count": len(self.meta_state["task_features"]),
            "strategy_updates": len(self.meta_state["strategy_history"]),
            "inner_stats": self.inner.get_statistics() if hasattr(self.inner, "get_statistics") else {},
        }

