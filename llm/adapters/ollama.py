"""
Ollama LLM 适配器

支持 Ollama 本地部署的开源模型。
Ollama 是一个本地运行 LLM 的工具，支持 Llama、Mistral、Qwen 等模型。

安装 Ollama: https://ollama.ai/
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
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class OllamaAdapter(BaseLLMAdapter):
    """
    Ollama LLM 适配器
    
    特性：
    - 支持所有 Ollama 模型
    - 同步和异步调用
    - 流式输出
    - 本地部署，支持知识写入
    
    支持的模型（示例）：
    - llama3.3:70b
    - qwen2.5:72b
    - deepseek-r1:70b
    - mistral:7b
    - codellama:34b
    
    配置示例：
        config = LLMConfig(
            base_url="http://localhost:11434",
            model="llama3.3:70b",
        )
    
    知识桥接能力：
    - 支持通过 Modelfile 自定义系统提示词
    - 支持创建带有特定知识的模型副本
    - 支持 LoRA 微调（需要额外配置）
    """
    
    adapter_name = "ollama"
    supported_capabilities = [
        LLMCapability.CHAT,
        LLMCapability.STREAMING,
        LLMCapability.EMBEDDING,
        LLMCapability.WEIGHT_ACCESS,  # 本地模型支持权重访问
    ]
    
    # 常用模型
    POPULAR_MODELS = {
        "llama3.3:70b": "Llama 3.3 70B (Meta)",
        "llama3.2:latest": "Llama 3.2 (Meta)",
        "qwen2.5:72b": "Qwen 2.5 72B (Alibaba)",
        "qwen2.5-coder:32b": "Qwen 2.5 Coder 32B",
        "deepseek-r1:70b": "DeepSeek R1 70B",
        "deepseek-coder-v2:latest": "DeepSeek Coder V2",
        "mistral:7b": "Mistral 7B",
        "mixtral:8x7b": "Mixtral 8x7B",
        "codellama:34b": "Code Llama 34B",
        "phi3:latest": "Phi-3 (Microsoft)",
    }
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化 Ollama 适配器
        
        Args:
            config: LLM 配置
            base_url: Ollama 服务地址
            model: 模型名称
        """
        if config is None:
            config = LLMConfig()
        
        if base_url:
            config.base_url = base_url
        elif config.base_url == "http://localhost:8000/v1":
            config.base_url = "http://localhost:11434"
        
        if model:
            config.model = model
        elif config.model == "default":
            config.model = "llama3.2:latest"
        
        super().__init__(config)
        
        self._sync_client = None
        self._async_client = None
    
    def initialize(self) -> bool:
        """初始化 HTTP 客户端"""
        if not HAS_HTTPX:
            raise ImportError(
                "httpx not installed. "
                "Please install: pip install httpx"
            )
        
        try:
            self._sync_client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
            
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
            
            # 验证连接
            response = self._sync_client.get("/api/tags")
            if response.status_code != 200:
                raise RuntimeError(f"Ollama server not responding: {response.status_code}")
            
            self._initialized = True
            return True
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Failed to initialize Ollama client: {e}")
    
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
            # Ollama API 格式
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            }
            
            # 添加额外选项
            if "top_p" in kwargs:
                payload["options"]["top_p"] = kwargs["top_p"]
            if "seed" in kwargs:
                payload["options"]["seed"] = kwargs["seed"]
            
            response = self._sync_client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            # 解析响应
            llm_response = LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason="stop" if data.get("done") else "length",
                latency_ms=latency_ms,
                raw_response=data,
            )
            
            self._record_call(llm_response)
            return llm_response
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Ollama chat failed: {e}")
    
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
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature,
                    "num_predict": max_tokens or self.config.max_tokens,
                }
            }
            
            response = await self._async_client.post("/api/chat", json=payload)
            response.raise_for_status()
            
            data = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            llm_response = LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason="stop" if data.get("done") else "length",
                latency_ms=latency_ms,
            )
            
            self._record_call(llm_response)
            return llm_response
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Ollama async chat failed: {e}")
    
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
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                }
            }
            
            with self._sync_client.stream("POST", "/api/chat", json=payload) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Ollama stream chat failed: {e}")
    
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
            payload = {
                "model": self.config.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                }
            }
            
            async with self._async_client.stream("POST", "/api/chat", json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
        
        except Exception as e:
            self._record_error()
            raise RuntimeError(f"Ollama async stream chat failed: {e}")
    
    # ==================== 知识桥接方法 ====================
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        列出本地可用模型
        
        Returns:
            模型列表
        """
        if not self._initialized:
            self.initialize()
        
        response = self._sync_client.get("/api/tags")
        response.raise_for_status()
        
        data = response.json()
        return data.get("models", [])
    
    def pull_model(self, model_name: str) -> bool:
        """
        拉取模型到本地
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否成功
        """
        if not self._initialized:
            self.initialize()
        
        try:
            response = self._sync_client.post(
                "/api/pull",
                json={"name": model_name},
                timeout=3600,  # 拉取可能需要很长时间
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def create_model_with_knowledge(
        self,
        base_model: str,
        new_model_name: str,
        system_prompt: str,
        knowledge_content: Optional[str] = None,
    ) -> bool:
        """
        创建带有特定知识的模型副本
        
        这是知识桥接的核心方法。通过 Modelfile 注入系统提示词和知识内容。
        
        Args:
            base_model: 基础模型名称
            new_model_name: 新模型名称
            system_prompt: 系统提示词
            knowledge_content: 可选的知识内容
            
        Returns:
            是否成功
        """
        if not self._initialized:
            self.initialize()
        
        # 构建 Modelfile
        modelfile_lines = [f"FROM {base_model}"]
        
        # 添加系统提示词
        if knowledge_content:
            full_system = f"{system_prompt}\n\n以下是你需要掌握的新知识：\n{knowledge_content}"
        else:
            full_system = system_prompt
        
        # 转义引号
        full_system = full_system.replace('"', '\\"')
        modelfile_lines.append(f'SYSTEM "{full_system}"')
        
        modelfile = "\n".join(modelfile_lines)
        
        try:
            response = self._sync_client.post(
                "/api/create",
                json={
                    "name": new_model_name,
                    "modelfile": modelfile,
                },
                timeout=300,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to create model: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """删除本地模型"""
        if not self._initialized:
            self.initialize()
        
        try:
            response = self._sync_client.delete(
                "/api/delete",
                json={"name": model_name},
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_embeddings(self, text: str) -> List[float]:
        """
        获取文本嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        if not self._initialized:
            self.initialize()
        
        response = self._sync_client.post(
            "/api/embeddings",
            json={
                "model": self.config.model,
                "prompt": text,
            }
        )
        response.raise_for_status()
        
        data = response.json()
        return data.get("embedding", [])
    
    def get_available_models(self) -> Dict[str, str]:
        """获取可用模型列表"""
        models = self.POPULAR_MODELS.copy()
        
        # 添加本地已下载的模型
        try:
            local_models = self.list_models()
            for m in local_models:
                name = m.get("name", "")
                if name and name not in models:
                    models[name] = f"{name} (local)"
        except Exception:
            pass
        
        return models

