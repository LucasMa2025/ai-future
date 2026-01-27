"""
OpenAI 兼容 LLM 适配器

支持任何兼容 OpenAI API 的服务端点。
包括：LMStudio, LocalAI, text-generation-webui, FastChat 等。
"""
from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator
import time

from .base import (
    BaseLLMAdapter,
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMCapability,
)

try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AsyncOpenAI = None


class OpenAICompatAdapter(BaseLLMAdapter):
    """
    OpenAI 兼容 LLM 适配器
    
    支持任何实现 OpenAI API 的服务端点。
    
    适用场景：
    - LMStudio 本地运行
    - LocalAI 服务
    - text-generation-webui (使用 OpenAI 扩展)
    - FastChat 服务
    - 任何其他 OpenAI 兼容端点
    
    配置示例（LMStudio）：
        config = LLMConfig(
            base_url="http://localhost:1234/v1",
            model="local-model",
        )
    
    配置示例（LocalAI）：
        config = LLMConfig(
            base_url="http://localhost:8080/v1",
            model="gpt-3.5-turbo",  # LocalAI 别名
        )
    """
    
    adapter_name = "openai_compat"
    supported_capabilities = [
        LLMCapability.CHAT,
        LLMCapability.STREAMING,
    ]
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        if config is None:
            config = LLMConfig()
        
        if base_url:
            config.base_url = base_url
        if api_key:
            config.api_key = api_key
        if model:
            config.model = model
        
        super().__init__(config)
        
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
    
    def initialize(self) -> bool:
        """初始化 OpenAI 客户端"""
        if not HAS_OPENAI:
            raise ImportError(
                "OpenAI SDK not installed. "
                "Please install: pip install openai"
            )
        
        try:
            self._client = OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key or "not-needed",
                timeout=self.config.timeout,
            )
            
            self._async_client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key or "not-needed",
                timeout=self.config.timeout,
            )
            
            self._initialized = True
            return True
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Failed to initialize OpenAI compat client: {e}")
    
    def chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """同步聊天"""
        if not self._initialized:
            self.initialize()
        
        messages = self._normalize_messages(messages)
        start_time = time.time()
        
        try:
            request_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            }
            
            # 传递额外参数（不同后端可能支持不同参数）
            for key in ["top_p", "stop", "response_format", "seed"]:
                if key in kwargs:
                    request_kwargs[key] = kwargs[key]
            
            response = self._client.chat.completions.create(**request_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model if response.model else self.config.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason or "stop",
                latency_ms=latency_ms,
            )
            
            self._record_call(llm_response)
            return llm_response
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"OpenAI compat chat failed: {e}")
    
    async def achat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """异步聊天"""
        if not self._initialized:
            self.initialize()
        
        messages = self._normalize_messages(messages)
        start_time = time.time()
        
        try:
            request_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            }
            
            for key in ["top_p", "stop", "response_format", "seed"]:
                if key in kwargs:
                    request_kwargs[key] = kwargs[key]
            
            response = await self._async_client.chat.completions.create(**request_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model if response.model else self.config.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason or "stop",
                latency_ms=latency_ms,
            )
            
            self._record_call(llm_response)
            return llm_response
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"OpenAI compat async chat failed: {e}")
    
    def stream_chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        **kwargs
    ) -> Iterator[str]:
        """流式聊天"""
        if not self._initialized:
            self.initialize()
        
        messages = self._normalize_messages(messages)
        
        try:
            request_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": True,
            }
            
            response = self._client.chat.completions.create(**request_kwargs)
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"OpenAI compat stream chat failed: {e}")
    
    async def astream_chat(
        self,
        messages: List[Union[Dict[str, str], LLMMessage]],
        **kwargs
    ) -> AsyncIterator[str]:
        """异步流式聊天"""
        if not self._initialized:
            self.initialize()
        
        messages = self._normalize_messages(messages)
        
        try:
            request_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "stream": True,
            }
            
            response = await self._async_client.chat.completions.create(**request_kwargs)
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"OpenAI compat async stream chat failed: {e}")
    
    def get_available_models(self) -> Dict[str, str]:
        """获取可用模型列表"""
        return {self.config.model: f"{self.config.model} (configured)"}

