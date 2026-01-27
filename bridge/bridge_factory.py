"""
Bridge 工厂

根据配置选择加载 AGA Bridge（远程 API）或传统 Bridge。

配置方式：
1. 环境变量: BRIDGE_TYPE=aga|traditional, AGA_API_URL=http://...
2. 配置文件: bridge_config.yaml
3. 代码参数: BridgeFactory.create(bridge_type="aga", aga_api_url="...")

默认使用 AGA Bridge（远程 API 模式）。
"""
import os
from typing import Optional, Union, Dict, Any
from enum import Enum
from dataclasses import dataclass, field

import torch.nn as nn


class BridgeType(str, Enum):
    """Bridge 类型"""
    AGA = "aga"                    # AGA 热插拔方案（默认，远程 API）
    TRADITIONAL = "traditional"    # 传统训练方案
    DEEP = "deep"                  # 深度内化方案（KnowledgeAdapter）


@dataclass
class BridgeConfig:
    """Bridge 统一配置"""
    # 选择方案
    bridge_type: BridgeType = BridgeType.AGA
    
    # 通用配置
    hidden_dim: int = 4096
    authorized_writers: set = field(default_factory=lambda: {"audit_system"})
    
    # AGA 特有配置（远程 API 模式）
    aga_api_url: str = "http://localhost:8081"  # AGA 服务地址
    aga_api_timeout: float = 30.0               # API 超时时间
    aga_bottleneck_dim: int = 64
    aga_initial_lifecycle: str = "probationary"
    
    # 传统方案配置
    traditional_decision_classes: int = 10
    traditional_use_ewc: bool = True
    traditional_ewc_lambda: float = 1000.0
    
    # 深度内化配置
    deep_adapter_rank: int = 16
    deep_num_adapter_layers: int = 4
    
    @classmethod
    def from_env(cls) -> "BridgeConfig":
        """从环境变量加载配置"""
        config = cls()
        
        # Bridge 类型
        bridge_type = os.environ.get("BRIDGE_TYPE", "aga").lower()
        if bridge_type in [e.value for e in BridgeType]:
            config.bridge_type = BridgeType(bridge_type)
        
        # AGA API 配置
        if os.environ.get("AGA_API_URL"):
            config.aga_api_url = os.environ["AGA_API_URL"]
        if os.environ.get("AGA_API_TIMEOUT"):
            config.aga_api_timeout = float(os.environ["AGA_API_TIMEOUT"])
        if os.environ.get("AGA_BOTTLENECK_DIM"):
            config.aga_bottleneck_dim = int(os.environ["AGA_BOTTLENECK_DIM"])
        
        return config
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BridgeConfig":
        """从字典加载配置"""
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                if key == "bridge_type":
                    value = BridgeType(value)
                setattr(config, key, value)
        return config


class BridgeFactory:
    """
    Bridge 工厂
    
    示例用法：
    ```python
    # 使用默认配置（AGA 远程 API）
    bridge = BridgeFactory.create(aga_api_url="http://gpu-server:8081")
    
    # 使用编码器（需要模型和 tokenizer）
    encoder = KnowledgeEncoder(model, tokenizer)
    bridge = BridgeFactory.create(
        aga_api_url="http://gpu-server:8081",
        encoder=encoder,
    )
    
    # 指定使用传统方案
    bridge = BridgeFactory.create(bridge_type="traditional")
    
    # 使用配置对象
    config = BridgeConfig(
        bridge_type=BridgeType.AGA,
        aga_api_url="http://gpu-server:8081",
    )
    bridge = BridgeFactory.create(config=config)
    ```
    """
    
    @staticmethod
    def create(
        model: Optional[nn.Module] = None,
        tokenizer = None,
        bridge_type: Optional[Union[str, BridgeType]] = None,
        config: Optional[BridgeConfig] = None,
        aga_api_url: Optional[str] = None,
        encoder = None,
        **kwargs,
    ):
        """
        创建 Bridge 实例
        
        Args:
            model: HuggingFace 模型（用于编码器，可选）
            tokenizer: 分词器（用于编码器，可选）
            bridge_type: Bridge 类型，覆盖 config 中的设置
            config: 配置对象
            aga_api_url: AGA API 地址（覆盖配置）
            encoder: 知识编码器实例（可选）
            **kwargs: 额外配置参数
        
        Returns:
            Bridge 实例
        """
        # 解析配置
        if config is None:
            config = BridgeConfig.from_env()
        
        # 覆盖 bridge_type
        if bridge_type is not None:
            if isinstance(bridge_type, str):
                bridge_type = BridgeType(bridge_type.lower())
            config.bridge_type = bridge_type
        
        # 覆盖 AGA API URL
        if aga_api_url is not None:
            config.aga_api_url = aga_api_url
        
        # 应用额外参数
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建实例
        if config.bridge_type == BridgeType.AGA:
            return BridgeFactory._create_aga_bridge(model, tokenizer, config, encoder)
        elif config.bridge_type == BridgeType.TRADITIONAL:
            return BridgeFactory._create_traditional_bridge(config)
        elif config.bridge_type == BridgeType.DEEP:
            return BridgeFactory._create_deep_bridge(model, tokenizer, config)
        else:
            raise ValueError(f"Unknown bridge type: {config.bridge_type}")
    
    @staticmethod
    def _create_aga_bridge(model, tokenizer, config: BridgeConfig, encoder=None):
        """创建 AGA Bridge（远程 API 模式）"""
        from .aga_bridge import AGABridge, AGABridgeConfig, KnowledgeEncoder, LifecycleState
        
        # 创建 AGA Bridge 配置
        aga_config = AGABridgeConfig(
            aga_api_url=config.aga_api_url,
            api_timeout=config.aga_api_timeout,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.aga_bottleneck_dim,
            initial_lifecycle=LifecycleState(config.aga_initial_lifecycle),
        )
        
        # 创建编码器（如果未提供且有模型）
        if encoder is None and model is not None:
            encoder = KnowledgeEncoder(model, tokenizer, aga_config)
        
        # 创建 Bridge
        bridge = AGABridge(
            aga_api_url=config.aga_api_url,
            encoder=encoder,
            config=aga_config,
            authorized_writers=config.authorized_writers,
        )
        
        return bridge
    
    @staticmethod
    def _create_traditional_bridge(config: BridgeConfig):
        """创建传统 Bridge"""
        from .production_bridge import ProductionBridge
        
        return ProductionBridge(
            authorized_writers=config.authorized_writers,
            hidden_dim=config.hidden_dim,
        )
    
    @staticmethod
    def _create_deep_bridge(model, tokenizer, config: BridgeConfig):
        """创建深度内化 Bridge"""
        from .deep_internalization_service import DeepInternalizationService
        
        return DeepInternalizationService(
            hidden_dim=config.hidden_dim,
            num_layers=config.deep_num_adapter_layers,
            adapter_rank=config.deep_adapter_rank,
        )


# 便捷函数
def create_bridge(
    model: Optional[nn.Module] = None,
    tokenizer = None,
    bridge_type: str = "aga",
    aga_api_url: str = "http://localhost:8081",
    **kwargs,
):
    """
    创建 Bridge 的便捷函数
    
    Args:
        model: HuggingFace 模型（用于编码器，可选）
        tokenizer: 分词器（用于编码器，可选）
        bridge_type: "aga" | "traditional" | "deep"
        aga_api_url: AGA API 服务地址
        **kwargs: 额外配置
    
    Returns:
        Bridge 实例
    
    示例：
        # 远程 API 模式（推荐）
        bridge = create_bridge(aga_api_url="http://gpu-server:8081")
        
        # 带编码器
        bridge = create_bridge(
            model=model,
            tokenizer=tokenizer,
            aga_api_url="http://gpu-server:8081",
        )
        
        # 传统方案
        bridge = create_bridge(bridge_type="traditional")
    """
    return BridgeFactory.create(
        model=model,
        tokenizer=tokenizer,
        bridge_type=bridge_type,
        aga_api_url=aga_api_url,
        **kwargs,
    )
