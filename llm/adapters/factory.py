"""
LLM 适配器工厂和注册机制

提供统一的适配器创建和管理接口。
支持动态注册自定义适配器。
"""
from typing import Dict, Type, Optional, Any, List, Callable
from dataclasses import dataclass, field

from .base import BaseLLMAdapter, LLMConfig, LLMCapability


@dataclass
class AdapterInfo:
    """适配器信息"""
    name: str
    adapter_class: Type[BaseLLMAdapter]
    description: str = ""
    default_config: Optional[Dict[str, Any]] = None
    capabilities: List[LLMCapability] = field(default_factory=list)


class LLMAdapterRegistry:
    """
    LLM 适配器注册表
    
    管理所有可用的 LLM 适配器，支持动态注册。
    
    使用示例：
        # 注册自定义适配器
        registry = LLMAdapterRegistry()
        registry.register("my_adapter", MyAdapter, "My custom adapter")
        
        # 获取适配器类
        adapter_class = registry.get("my_adapter")
    """
    
    _instance: Optional["LLMAdapterRegistry"] = None
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._adapters: Dict[str, AdapterInfo] = {}
            cls._instance._initialized = False
        return cls._instance
    
    def _init_builtin_adapters(self):
        """初始化内置适配器"""
        if self._initialized:
            return
        
        # 延迟导入以避免循环依赖
        from .deepseek import DeepSeekAdapter
        from .ollama import OllamaAdapter
        from .vllm import VLLMAdapter
        from .openai_compat import OpenAICompatAdapter
        from .base import MockLLMAdapter
        
        self.register(
            "deepseek",
            DeepSeekAdapter,
            "DeepSeek LLM (API/本地部署)",
            default_config={
                "base_url": "http://localhost:8001/v1",
                "model": "deepseek-chat",
            },
        )
        
        self.register(
            "ollama",
            OllamaAdapter,
            "Ollama 本地模型 (Llama, Qwen, Mistral 等)",
            default_config={
                "base_url": "http://localhost:11434",
                "model": "llama3.2:latest",
            },
        )
        
        self.register(
            "vllm",
            VLLMAdapter,
            "vLLM 高性能推理服务",
            default_config={
                "base_url": "http://localhost:8001/v1",
                "model": "deepseek-coder-7b",
            },
        )
        
        self.register(
            "openai_compat",
            OpenAICompatAdapter,
            "OpenAI 兼容接口 (LMStudio, LocalAI 等)",
            default_config={
                "base_url": "http://localhost:1234/v1",
                "model": "local-model",
            },
        )
        
        self.register(
            "mock",
            MockLLMAdapter,
            "模拟适配器 (用于测试)",
            default_config={
                "model": "mock",
            },
        )
        
        self._initialized = True
    
    def register(
        self,
        name: str,
        adapter_class: Type[BaseLLMAdapter],
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        注册适配器
        
        Args:
            name: 适配器名称
            adapter_class: 适配器类
            description: 描述
            default_config: 默认配置
        """
        info = AdapterInfo(
            name=name,
            adapter_class=adapter_class,
            description=description,
            default_config=default_config,
            capabilities=list(adapter_class.supported_capabilities),
        )
        self._adapters[name] = info
    
    def unregister(self, name: str) -> bool:
        """
        取消注册适配器
        
        Args:
            name: 适配器名称
            
        Returns:
            是否成功
        """
        if name in self._adapters:
            del self._adapters[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Type[BaseLLMAdapter]]:
        """
        获取适配器类
        
        Args:
            name: 适配器名称
            
        Returns:
            适配器类，如果不存在则返回 None
        """
        self._init_builtin_adapters()
        
        info = self._adapters.get(name)
        return info.adapter_class if info else None
    
    def get_info(self, name: str) -> Optional[AdapterInfo]:
        """获取适配器信息"""
        self._init_builtin_adapters()
        return self._adapters.get(name)
    
    def list_adapters(self) -> List[AdapterInfo]:
        """列出所有已注册的适配器"""
        self._init_builtin_adapters()
        return list(self._adapters.values())
    
    def list_names(self) -> List[str]:
        """列出所有适配器名称"""
        self._init_builtin_adapters()
        return list(self._adapters.keys())
    
    def has(self, name: str) -> bool:
        """检查适配器是否已注册"""
        self._init_builtin_adapters()
        return name in self._adapters
    
    def get_by_capability(self, capability: LLMCapability) -> List[AdapterInfo]:
        """
        根据能力筛选适配器
        
        Args:
            capability: 所需能力
            
        Returns:
            具备该能力的适配器列表
        """
        self._init_builtin_adapters()
        
        result = []
        for info in self._adapters.values():
            if capability in info.capabilities:
                result.append(info)
        return result


class LLMAdapterFactory:
    """
    LLM 适配器工厂
    
    创建和管理 LLM 适配器实例。
    
    使用示例：
        # 创建适配器
        adapter = LLMAdapterFactory.create("deepseek", config={...})
        
        # 或使用配置对象
        config = LLMConfig(base_url="...", model="...")
        adapter = LLMAdapterFactory.create("ollama", config=config)
        
        # 从环境变量创建
        adapter = LLMAdapterFactory.from_env()
    """
    
    # 默认适配器类型
    DEFAULT_ADAPTER = "deepseek"
    
    # 适配器实例缓存（可选）
    _instances: Dict[str, BaseLLMAdapter] = {}
    
    @classmethod
    def create(
        cls,
        adapter_type: str,
        config: Optional[LLMConfig] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseLLMAdapter:
        """
        创建 LLM 适配器实例
        
        Args:
            adapter_type: 适配器类型 ("deepseek", "ollama", "vllm", etc.)
            config: LLMConfig 配置对象
            config_dict: 配置字典（会转换为 LLMConfig）
            **kwargs: 直接传递给适配器构造函数的参数
            
        Returns:
            适配器实例
            
        Raises:
            ValueError: 如果适配器类型未注册
        """
        registry = LLMAdapterRegistry()
        
        adapter_class = registry.get(adapter_type)
        if adapter_class is None:
            available = registry.list_names()
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. "
                f"Available: {available}"
            )
        
        # 构建配置
        if config is None:
            if config_dict:
                config = LLMConfig(**config_dict)
            else:
                # 使用默认配置
                info = registry.get_info(adapter_type)
                if info and info.default_config:
                    config = LLMConfig(**info.default_config)
                else:
                    config = LLMConfig()
        
        # 应用 kwargs 覆盖
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 创建实例
        adapter = adapter_class(config=config)
        
        return adapter
    
    @classmethod
    def create_and_initialize(
        cls,
        adapter_type: str,
        config: Optional[LLMConfig] = None,
        **kwargs
    ) -> BaseLLMAdapter:
        """
        创建并初始化适配器
        
        与 create() 相同，但会自动调用 initialize()
        """
        adapter = cls.create(adapter_type, config, **kwargs)
        adapter.initialize()
        return adapter
    
    @classmethod
    def from_env(
        cls,
        adapter_type_env: str = "LLM_ADAPTER_TYPE",
        base_url_env: str = "LLM_BASE_URL",
        api_key_env: str = "LLM_API_KEY",
        model_env: str = "LLM_MODEL",
    ) -> BaseLLMAdapter:
        """
        从环境变量创建适配器
        
        Args:
            adapter_type_env: 适配器类型环境变量名
            base_url_env: Base URL 环境变量名
            api_key_env: API Key 环境变量名
            model_env: 模型名环境变量名
            
        Returns:
            适配器实例
        """
        import os
        
        adapter_type = os.getenv(adapter_type_env, cls.DEFAULT_ADAPTER)
        
        config = LLMConfig(
            base_url=os.getenv(base_url_env, "http://localhost:8001/v1"),
            api_key=os.getenv(api_key_env, "not-needed"),
            model=os.getenv(model_env, "default"),
        )
        
        return cls.create(adapter_type, config=config)
    
    @classmethod
    def get_or_create(
        cls,
        adapter_type: str,
        instance_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
        **kwargs
    ) -> BaseLLMAdapter:
        """
        获取或创建适配器实例（单例模式）
        
        Args:
            adapter_type: 适配器类型
            instance_key: 实例键（用于区分同类型的不同实例）
            config: 配置
            **kwargs: 额外参数
            
        Returns:
            适配器实例
        """
        key = instance_key or f"{adapter_type}_default"
        
        if key not in cls._instances:
            cls._instances[key] = cls.create_and_initialize(
                adapter_type, config, **kwargs
            )
        
        return cls._instances[key]
    
    @classmethod
    def clear_instances(cls) -> None:
        """清除所有缓存的实例"""
        cls._instances.clear()
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的适配器类型"""
        return LLMAdapterRegistry().list_names()
    
    @classmethod
    def get_adapter_info(cls, adapter_type: str) -> Optional[AdapterInfo]:
        """获取适配器信息"""
        return LLMAdapterRegistry().get_info(adapter_type)
    
    @classmethod
    def register_adapter(
        cls,
        name: str,
        adapter_class: Type[BaseLLMAdapter],
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        注册自定义适配器
        
        Args:
            name: 适配器名称
            adapter_class: 适配器类
            description: 描述
            default_config: 默认配置
        """
        LLMAdapterRegistry().register(
            name, adapter_class, description, default_config
        )


# 便捷函数
def create_llm_adapter(
    adapter_type: str = "deepseek",
    **kwargs
) -> BaseLLMAdapter:
    """
    便捷函数：创建 LLM 适配器
    
    Args:
        adapter_type: 适配器类型
        **kwargs: 配置参数
        
    Returns:
        适配器实例
    """
    return LLMAdapterFactory.create(adapter_type, **kwargs)


def get_default_adapter() -> BaseLLMAdapter:
    """
    获取默认适配器（从环境变量配置）
    """
    return LLMAdapterFactory.from_env()

