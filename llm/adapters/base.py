"""
LLM 适配器基类

定义所有 LLM 适配器必须实现的统一接口。
支持同步/异步调用、流式输出、JSON 模式等。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, AsyncIterator, Iterator, Union
from enum import Enum
from datetime import datetime
import json


class LLMCapability(Enum):
    """LLM 能力枚举"""
    CHAT = "chat"                    # 基础聊天
    JSON_MODE = "json_mode"          # JSON 输出模式
    STREAMING = "streaming"          # 流式输出
    FUNCTION_CALL = "function_call"  # 函数调用
    VISION = "vision"                # 视觉理解
    EMBEDDING = "embedding"          # 文本嵌入
    FINE_TUNING = "fine_tuning"      # 微调能力
    WEIGHT_ACCESS = "weight_access"  # 权重访问（本地部署）


@dataclass
class LLMConfig:
    """
    LLM 配置
    
    统一的配置结构，适用于所有适配器
    """
    # 连接配置
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    
    # 模型配置
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    
    # 超时和重试
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 高级配置
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": "***" if self.api_key else None,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


@dataclass
class LLMResponse:
    """
    LLM 响应
    
    统一的响应结构
    """
    content: str
    model: str
    
    # Token 统计
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # 元数据
    finish_reason: str = "stop"
    created_at: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    
    # 原始响应（用于调试）
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


@dataclass
class LLMMessage:
    """聊天消息"""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class BaseLLMAdapter(ABC):
    """
    LLM 适配器基类
    
    所有 LLM 适配器必须继承此类并实现抽象方法。
    
    设计原则：
    1. 统一接口 - 所有适配器使用相同的方法签名
    2. 能力声明 - 适配器声明自己支持的能力
    3. 可注入性 - 支持依赖注入到自学习系统
    4. 可观测性 - 提供统计和监控接口
    
    使用示例：
        adapter = DeepSeekAdapter(config)
        response = adapter.chat([
            {"role": "user", "content": "Hello"}
        ])
    """
    
    # 适配器名称（子类必须定义）
    adapter_name: str = "base"
    
    # 支持的能力（子类可覆盖）
    supported_capabilities: List[LLMCapability] = [LLMCapability.CHAT]
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        
        # 统计
        self._total_calls = 0
        self._total_tokens = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
        
        # 初始化标志
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化适配器
        
        子类应实现连接测试、客户端初始化等逻辑。
        
        Returns:
            是否初始化成功
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        同步聊天
        
        Args:
            messages: 消息列表
            temperature: 温度（覆盖配置）
            max_tokens: 最大 token 数（覆盖配置）
            **kwargs: 其他参数
            
        Returns:
            LLMResponse
        """
        pass
    
    @abstractmethod
    async def achat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        异步聊天
        """
        pass
    
    def chat_json(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        聊天并返回 JSON
        
        如果适配器支持 JSON_MODE，会启用 JSON 模式。
        否则尝试从响应中解析 JSON。
        """
        if LLMCapability.JSON_MODE in self.supported_capabilities:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.chat(messages, temperature=temperature, **kwargs)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # 返回包装的响应
            return {"response": response.content, "_parse_error": True}
    
    async def achat_json(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """异步聊天并返回 JSON"""
        if LLMCapability.JSON_MODE in self.supported_capabilities:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = await self.achat(messages, temperature=temperature, **kwargs)
        
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return {"response": response.content, "_parse_error": True}
    
    def simple_query(self, query: str) -> str:
        """
        简单查询（便捷方法）
        """
        response = self.chat([{"role": "user", "content": query}])
        return response.content
    
    async def asimple_query(self, query: str) -> str:
        """异步简单查询"""
        response = await self.achat([{"role": "user", "content": query}])
        return response.content
    
    def stream_chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        **kwargs
    ) -> Iterator[str]:
        """
        流式聊天（默认实现：非流式）
        
        支持流式的适配器应覆盖此方法。
        """
        response = self.chat(messages, **kwargs)
        yield response.content
    
    async def astream_chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式聊天"""
        response = await self.achat(messages, **kwargs)
        yield response.content
    
    def test_connection(self) -> bool:
        """
        测试连接
        """
        try:
            response = self.simple_query("Respond with 'OK' only.")
            return len(response) > 0
        except Exception:
            return False
    
    def has_capability(self, capability: LLMCapability) -> bool:
        """检查是否支持某能力"""
        return capability in self.supported_capabilities
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_latency = (
            self._total_latency_ms / self._total_calls 
            if self._total_calls > 0 else 0
        )
        return {
            "adapter_name": self.adapter_name,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_errors": self._total_errors,
            "avg_latency_ms": avg_latency,
            "capabilities": [c.value for c in self.supported_capabilities],
            "initialized": self._initialized,
        }
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_errors = 0
        self._total_latency_ms = 0.0
    
    def _normalize_messages(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]]
    ) -> List[Dict[str, str]]:
        """标准化消息格式"""
        result = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                result.append(msg.to_dict())
            else:
                result.append(msg)
        return result
    
    def _record_call(self, response: LLMResponse) -> None:
        """记录调用统计"""
        self._total_calls += 1
        self._total_tokens += response.total_tokens
        self._total_latency_ms += response.latency_ms
    
    def _record_error(self) -> None:
        """记录错误"""
        self._total_errors += 1
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model={self.config.model}>"


class MockLLMAdapter(BaseLLMAdapter):
    """
    模拟 LLM 适配器
    
    用于测试和开发，不需要实际的 LLM 服务。
    """
    
    adapter_name = "mock"
    supported_capabilities = [
        LLMCapability.CHAT,
        LLMCapability.JSON_MODE,
        LLMCapability.STREAMING,
    ]
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        responses: Optional[Dict[str, str]] = None
    ):
        super().__init__(config)
        self.responses = responses or {}
        self._initialized = True
    
    def initialize(self) -> bool:
        self._initialized = True
        return True
    
    def chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        import time
        start = time.time()
        
        messages = self._normalize_messages(messages)
        last_content = messages[-1]["content"] if messages else ""
        
        # 查找匹配的响应
        response_content = self._get_mock_response(last_content, kwargs)
        
        latency = (time.time() - start) * 1000
        
        response = LLMResponse(
            content=response_content,
            model="mock",
            prompt_tokens=len(last_content) // 4,
            completion_tokens=len(response_content) // 4,
            total_tokens=(len(last_content) + len(response_content)) // 4,
            latency_ms=latency,
        )
        
        self._record_call(response)
        return response
    
    async def achat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        return self.chat(messages, temperature, max_tokens, **kwargs)
    
    def _get_mock_response(self, query: str, kwargs: Dict) -> str:
        """获取模拟响应"""
        # 检查是否需要 JSON
        if kwargs.get("response_format", {}).get("type") == "json_object":
            if "风险" in query or "risk" in query.lower():
                return json.dumps({
                    "risk_level": "medium",
                    "confidence": 0.75,
                    "factors": [
                        {"factor": "知识来源", "assessment": "可靠", "score": 0.8},
                    ],
                    "recommendation": "建议人工审核"
                }, ensure_ascii=False)
            
            if "探索" in query or "explore" in query.lower():
                return json.dumps({
                    "action": "reasoning",
                    "findings": [
                        {"type": "knowledge", "content": "发现新知识点"},
                    ],
                    "next_steps": ["深入分析"]
                }, ensure_ascii=False)
            
            if "知识" in query or "knowledge" in query.lower():
                return json.dumps({
                    "domain": "general",
                    "type": "knowledge",
                    "content": {"summary": "模拟知识内容"},
                    "confidence": 0.7,
                    "rationale": "基于模拟分析"
                }, ensure_ascii=False)
            
            return json.dumps({"response": "模拟 JSON 响应"}, ensure_ascii=False)
        
        # 普通文本响应
        for key, value in self.responses.items():
            if key.lower() in query.lower():
                return value
        
        return "这是一个模拟响应。"

