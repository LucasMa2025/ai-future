"""
DeepSeek LLM 客户端

支持本地部署和 API 调用
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import asyncio

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    OpenAI = None
    AsyncOpenAI = None

from core.exceptions import LLMConnectionError, LLMResponseError


class DeepSeekClient:
    """
    DeepSeek LLM 客户端
    
    特性：
    - 支持本地部署（vLLM/TensorRT-LLM）
    - 支持 DeepSeek API
    - 同步和异步调用
    - 自动重试
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "not-needed",  # 本地部署不需要
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # 初始化客户端
        if OpenAI is None:
            raise ImportError("Please install openai: pip install openai")
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout
        )
        
        # 统计
        self.total_calls = 0
        self.total_tokens = 0
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        同步聊天调用
        
        Args:
            messages: 消息列表
            temperature: 温度
            max_tokens: 最大 token 数
            response_format: 响应格式（如 {"type": "json_object"}）
            
        Returns:
            响应文本
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            
            self.total_calls += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise LLMResponseError(f"LLM call failed: {e}")
    
    async def achat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> str:
        """
        异步聊天调用
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
            
            if response_format:
                kwargs["response_format"] = response_format
            
            response = await self.async_client.chat.completions.create(**kwargs)
            
            self.total_calls += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content
        
        except Exception as e:
            raise LLMResponseError(f"Async LLM call failed: {e}")
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        聊天并返回 JSON
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 尝试提取 JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise LLMResponseError(f"Failed to parse JSON response: {response}")
    
    def simple_query(self, query: str) -> str:
        """
        简单查询
        """
        return self.chat([{"role": "user", "content": query}])
    
    def test_connection(self) -> bool:
        """
        测试连接
        """
        try:
            response = self.simple_query("Hello, respond with 'OK' only.")
            return len(response) > 0
        except Exception:
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'model': self.model,
            'base_url': self.base_url,
        }


class MockDeepSeekClient:
    """
    模拟 DeepSeek 客户端（用于测试）
    """
    
    def __init__(self, **kwargs):
        self.total_calls = 0
        self.total_tokens = 0
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """模拟聊天"""
        self.total_calls += 1
        
        # 根据消息内容返回模拟响应
        last_message = messages[-1]['content'] if messages else ""
        
        if "风险评估" in last_message or "risk" in last_message.lower():
            return json.dumps({
                "risk_level": "medium",
                "confidence": 0.75,
                "factors": [
                    {"factor": "知识来源", "assessment": "可靠", "score": 0.8},
                    {"factor": "推理链路", "assessment": "合理", "score": 0.7},
                ],
                "recommendation": "建议人工审核"
            })
        
        if "探索" in last_message or "explore" in last_message.lower():
            return json.dumps({
                "findings": [
                    {"type": "knowledge", "content": "发现新知识点"},
                    {"type": "pattern", "content": "识别到模式"},
                ],
                "next_steps": ["深入分析", "验证假设"]
            })
        
        return "这是一个模拟响应。"
    
    async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """异步模拟聊天"""
        return self.chat(messages, **kwargs)
    
    def chat_json(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """模拟 JSON 聊天"""
        response = self.chat(messages, **kwargs)
        try:
            return json.loads(response)
        except:
            return {"response": response}
    
    def simple_query(self, query: str) -> str:
        """简单查询"""
        return self.chat([{"role": "user", "content": query}])
    
    def test_connection(self) -> bool:
        """测试连接"""
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'total_calls': self.total_calls,
            'total_tokens': self.total_tokens,
            'model': 'mock',
            'base_url': 'mock://localhost',
        }

