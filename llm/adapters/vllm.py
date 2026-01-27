"""
vLLM LLM 适配器

支持 vLLM 本地部署的高性能推理服务。
vLLM 是一个高吞吐量、低延迟的 LLM 推理引擎。

安装 vLLM: pip install vllm
启动服务: python -m vllm.entrypoints.openai.api_server --model <model_name>
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

# 可选依赖 - vLLM 使用 OpenAI 兼容接口
try:
    from openai import OpenAI, AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None
    AsyncOpenAI = None

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class VLLMAdapter(BaseLLMAdapter):
    """
    vLLM LLM 适配器
    
    特性：
    - 高吞吐量推理
    - 低延迟响应
    - 支持 PagedAttention
    - 支持连续批处理
    - OpenAI 兼容 API
    - 支持 LoRA 动态加载
    
    vLLM 启动命令示例：
        python -m vllm.entrypoints.openai.api_server \\
            --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 \\
            --port 8001 \\
            --tensor-parallel-size 2
    
    配置示例：
        config = LLMConfig(
            base_url="http://localhost:8001/v1",
            model="deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        )
    
    知识桥接能力：
    - 支持 LoRA 适配器动态加载/卸载
    - 支持模型权重热更新（需要特定配置）
    """
    
    adapter_name = "vllm"
    supported_capabilities = [
        LLMCapability.CHAT,
        LLMCapability.JSON_MODE,
        LLMCapability.STREAMING,
        LLMCapability.WEIGHT_ACCESS,
        LLMCapability.FINE_TUNING,
    ]
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 vLLM 适配器
        
        Args:
            config: LLM 配置
            base_url: vLLM 服务地址
            model: 模型名称
        """
        if config is None:
            config = LLMConfig()
        
        if base_url:
            config.base_url = base_url
        elif config.base_url == "http://localhost:8000/v1":
            config.base_url = "http://localhost:8001/v1"
        
        if model:
            config.model = model
        
        super().__init__(config)
        
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None
        self._http_client = None
        
        # vLLM 特定状态
        self._loaded_loras: List[str] = []
    
    def initialize(self) -> bool:
        """初始化客户端"""
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
            
            # 创建 HTTP 客户端用于 vLLM 特定 API
            if HAS_HTTPX:
                # 提取 base URL（去掉 /v1）
                base = self.config.base_url.replace("/v1", "")
                self._http_client = httpx.Client(
                    base_url=base,
                    timeout=self.config.timeout,
                )
            
            self._initialized = True
            return True
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Failed to initialize vLLM client: {e}")
    
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
            
            # vLLM 特定参数
            if "best_of" in kwargs:
                request_kwargs["best_of"] = kwargs["best_of"]
            if "use_beam_search" in kwargs:
                request_kwargs["use_beam_search"] = kwargs["use_beam_search"]
            if "top_k" in kwargs:
                request_kwargs["top_k"] = kwargs["top_k"]
            if "presence_penalty" in kwargs:
                request_kwargs["presence_penalty"] = kwargs["presence_penalty"]
            if "frequency_penalty" in kwargs:
                request_kwargs["frequency_penalty"] = kwargs["frequency_penalty"]
            
            # 处理 LoRA
            if "lora_name" in kwargs:
                # vLLM 支持通过模型名称指定 LoRA
                request_kwargs["model"] = kwargs["lora_name"]
            
            if "response_format" in kwargs:
                request_kwargs["response_format"] = kwargs["response_format"]
            
            response = self._client.chat.completions.create(**request_kwargs)
            
            latency_ms = (time.time() - start_time) * 1000
            
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
            raise RuntimeError(f"vLLM chat failed: {e}")
    
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
            raise RuntimeError(f"vLLM async chat failed: {e}")
    
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
            raise RuntimeError(f"vLLM stream chat failed: {e}")
    
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
            raise RuntimeError(f"vLLM async stream chat failed: {e}")
    
    # ==================== vLLM 特定方法 ====================
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if not self._initialized:
            self.initialize()
        
        if not self._http_client:
            return {"error": "HTTP client not available"}
        
        try:
            response = self._http_client.get("/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def load_lora(self, lora_name: str, lora_path: str) -> bool:
        """
        动态加载 LoRA 适配器
        
        注意：需要 vLLM 以 --enable-lora 启动
        
        Args:
            lora_name: LoRA 名称（用于后续引用）
            lora_path: LoRA 权重路径
            
        Returns:
            是否成功
        """
        if not self._http_client:
            return False
        
        try:
            # vLLM 通过启动参数加载 LoRA，运行时动态加载需要特定版本支持
            # 这里提供接口框架，实际实现依赖 vLLM 版本
            response = self._http_client.post(
                "/v1/load_lora",
                json={
                    "lora_name": lora_name,
                    "lora_path": lora_path,
                }
            )
            
            if response.status_code == 200:
                self._loaded_loras.append(lora_name)
                return True
            return False
        
        except Exception:
            return False
    
    def unload_lora(self, lora_name: str) -> bool:
        """卸载 LoRA 适配器"""
        if not self._http_client:
            return False
        
        try:
            response = self._http_client.post(
                "/v1/unload_lora",
                json={"lora_name": lora_name}
            )
            
            if response.status_code == 200:
                if lora_name in self._loaded_loras:
                    self._loaded_loras.remove(lora_name)
                return True
            return False
        
        except Exception:
            return False
    
    def get_loaded_loras(self) -> List[str]:
        """获取已加载的 LoRA 列表"""
        return self._loaded_loras.copy()
    
    def get_server_stats(self) -> Dict[str, Any]:
        """
        获取 vLLM 服务器统计信息
        
        Returns:
            服务器状态和性能指标
        """
        if not self._http_client:
            return {"error": "HTTP client not available"}
        
        try:
            response = self._http_client.get("/metrics")
            if response.status_code == 200:
                return {"metrics": response.text}
            return {"error": f"Status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_available_models(self) -> Dict[str, str]:
        """获取可用模型列表"""
        models = {
            self.config.model: f"{self.config.model} (current)",
        }
        
        # 尝试获取服务器上的模型列表
        info = self.get_model_info()
        if "data" in info:
            for m in info["data"]:
                model_id = m.get("id", "")
                if model_id and model_id not in models:
                    models[model_id] = model_id
        
        return models

