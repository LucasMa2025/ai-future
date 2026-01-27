"""
DeepSeek LLM 适配器

支持 DeepSeek API 和本地部署（vLLM/TensorRT-LLM）。
"""
from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator
import time
import json

from .base import (
    BaseLLMAdapter,
    LLMConfig,
    LLMResponse,
    LLMMessage,
    LLMCapability,
)

# 可选依赖
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AsyncOpenAI = None


class DeepSeekAdapter(BaseLLMAdapter):
    """
    DeepSeek LLM 适配器
    
    特性：
    - 支持 DeepSeek API
    - 支持本地部署（使用 OpenAI 兼容接口）
    - 同步和异步调用
    - 流式输出
    - JSON 模式
    
    配置示例（API）：
        config = LLMConfig(
            base_url="https://api.deepseek.com/v1",
            api_key="your-api-key",
            model="deepseek-chat",
        )
    
    配置示例（本地部署）：
        config = LLMConfig(
            base_url="http://localhost:8001/v1",
            api_key="not-needed",
            model="deepseek-coder-7b",
        )
    """
    
    adapter_name = "deepseek"
    supported_capabilities = [
        LLMCapability.CHAT,
        LLMCapability.JSON_MODE,
        LLMCapability.STREAMING,
        LLMCapability.FUNCTION_CALL,
    ]
    
    # DeepSeek 模型列表
    MODELS = {
        "deepseek-chat": "DeepSeek Chat (通用对话)",
        "deepseek-coder": "DeepSeek Coder (代码生成)",
        "deepseek-reasoner": "DeepSeek R1 (推理增强)",
    }
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 DeepSeek 适配器
        
        Args:
            config: LLM 配置
            base_url: API 地址（覆盖 config）
            api_key: API 密钥（覆盖 config）
            model: 模型名称（覆盖 config）
        """
        # 创建或更新配置
        if config is None:
            config = LLMConfig()
        
        if base_url:
            config.base_url = base_url
        if api_key:
            config.api_key = api_key
        if model:
            config.model = model
        
        # 默认 DeepSeek 配置
        if config.base_url == "http://localhost:8000/v1":
            config.base_url = "http://localhost:8001/v1"
        if config.model == "default":
            config.model = "deepseek-chat"
        
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
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
            
            self._async_client = AsyncOpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
            )
            
            self._initialized = True
            return True
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Failed to initialize DeepSeek client: {e}")
    
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
            # 构建请求参数
            request_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": temperature or self.config.temperature,
                "max_tokens": max_tokens or self.config.max_tokens,
            }
            
            # 处理 response_format
            if "response_format" in kwargs:
                request_kwargs["response_format"] = kwargs["response_format"]
            
            # 处理其他参数
            if "top_p" in kwargs:
                request_kwargs["top_p"] = kwargs["top_p"]
            if "stop" in kwargs:
                request_kwargs["stop"] = kwargs["stop"]
            
            # 调用 API
            response = self._client.chat.completions.create(**request_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 构建响应
            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason or "stop",
                latency_ms=latency_ms,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None,
            )
            
            self._record_call(llm_response)
            return llm_response
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"DeepSeek chat failed: {e}")
    
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
            
            if "response_format" in kwargs:
                request_kwargs["response_format"] = kwargs["response_format"]
            
            response = await self._async_client.chat.completions.create(**request_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
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
            raise RuntimeError(f"DeepSeek async chat failed: {e}")
    
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
            raise RuntimeError(f"DeepSeek stream chat failed: {e}")
    
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
            raise RuntimeError(f"DeepSeek async stream chat failed: {e}")
    
    def get_available_models(self) -> Dict[str, str]:
        """获取可用模型列表"""
        return self.MODELS.copy()

