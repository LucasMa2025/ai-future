"""
知识转移服务 v4.0

实现:
1. 审批通过的 Learning Unit 通过 HTTP API 转移到 AGA Portal
2. AGA 生命周期管理 (通过 Portal API)
3. 知识隔离和回滚

架构说明:
    ┌─────────────────────────────────────────────────────────┐
    │  AIFuture Backend (治理系统 - 主权层)                   │
    │  ┌───────────────────────────────────────────────────┐ │
    │  │ ApprovalService                                   │ │
    │  │    ↓ submit_lu_approval() 审批通过                │ │
    │  │ KnowledgeTransferService                          │ │
    │  │    ↓ transfer_to_aga() 转移知识                   │ │
    │  │ AGAPortalClient                                   │ │
    │  │    ↓ inject_knowledge_text() 传递文本规则         │ │
    │  └─────────────────────────┬─────────────────────────┘ │
    │  ⚠️ 治理系统只传递文本，不涉及 KV 编码                 │
    └────────────────────────────┼────────────────────────────┘
                                 │ HTTP REST API (文本)
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │  AGA Portal (计算层 - 独立部署)                         │
    │  ┌───────────────────────────────────────────────────┐ │
    │  │ Portal API (aga.portal)                           │ │
    │  │  - POST /knowledge/inject-text  ← 推荐 API        │ │
    │  │  - PUT /lifecycle/update                          │ │
    │  │  - POST /lifecycle/quarantine                     │ │
    │  │  - GET /statistics, /audit                        │ │
    │  │                                                   │ │
    │  │ Encoder (编码器)                                  │ │
    │  │  - 将文本转换为 KV 向量                           │ │
    │  │  - 确保编码器一致性                               │ │
    │  └─────────────────────────┬─────────────────────────┘ │
    │                            │ Redis Pub/Sub             │
    │                            ▼                           │
    │  ┌───────────────────────────────────────────────────┐ │
    │  │ AGA Runtime (GPU 服务器)                          │ │
    │  │  - 订阅同步消息                                   │ │
    │  │  - 执行推理（使用相同编码器）                     │ │
    │  └───────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘

⚠️ 重要架构原则:
- 治理系统 = 规则与授权的"主权层"，只传递语义描述
- AGA Portal = 规则被执行的"计算层"，负责 KV 编码
- KV encoding 本质上是 Transformer 内部状态的构造，属于计算执行权
- 治理系统不应碰 KV，否则会与模型强耦合

设计目的:
1. 编码器一致性由 Portal 单点保证
2. 治理系统与模型解耦（模型升级无需改治理系统）
3. 规则文本可审计
4. 安全模型正确（治理系统不可信但可验证，AGA 可信但受限）
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Tuple
from uuid import UUID
from dataclasses import dataclass, field
import logging
import json
import hashlib

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from sqlalchemy.orm import Session

from ..models.learning_unit import LearningUnit, LUConstraint
from ..models.user import User
from ..core.enums import LearningUnitStatus
from ..core.exceptions import NotFoundError, BusinessError
from ..config import settings


if TYPE_CHECKING:
    from .learning_unit_service import LearningUnitService


logger = logging.getLogger(__name__)


# ============================================================
# AGA 生命周期状态（与 AGA Portal 对齐）
# ============================================================

class AGALifecycleState:
    """AGA 生命周期状态"""
    QUARANTINED = "quarantined"    # 已隔离 (r=0.0)
    PROBATIONARY = "probationary"  # 试用期 (r=0.3)
    CONFIRMED = "confirmed"        # 已确认 (r=1.0)
    DEPRECATED = "deprecated"      # 已弃用 (r=0.1)


# ============================================================
# AGA Portal HTTP 客户端
# ============================================================

class AGAPortalClient:
    """
    AGA Portal HTTP 客户端
    
    通过 HTTP REST API 与远程 AGA Portal 通信。
    
    API 端点参考: AGA/docs/Portal_API_Reference.md
    """
    
    def __init__(
        self,
        base_url: str = None,
        timeout: float = None,
        api_key: str = None,
        namespace: str = None,
    ):
        """
        初始化客户端
        
        Args:
            base_url: AGA Portal API 地址
            timeout: 请求超时（秒）
            api_key: API 密钥（可选）
            namespace: 默认命名空间
        """
        self.base_url = (base_url or settings.AGA_PORTAL_URL).rstrip("/")
        self.timeout = timeout or settings.AGA_PORTAL_TIMEOUT
        self.api_key = api_key or settings.AGA_PORTAL_API_KEY
        self.default_namespace = namespace or settings.AGA_PORTAL_NAMESPACE
        
        # 构建 headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=headers,
        )
        
        self._async_client = None  # 延迟初始化
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """获取异步客户端"""
        if self._async_client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._async_client
    
    def close(self):
        """关闭客户端"""
        self._client.close()
        if self._async_client:
            # 注意：异步客户端需要在异步上下文中关闭
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # ==================== 健康检查 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            response = self._client.get("/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"AGA Portal health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def is_healthy(self) -> bool:
        """检查 Portal 是否健康"""
        health = self.health_check()
        return health.get("status") == "healthy"
    
    # ==================== 知识管理 ====================
    
    @retry(
        stop=stop_after_attempt(settings.AGA_PORTAL_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.AGA_PORTAL_RETRY_DELAY),
    )
    def inject_knowledge_text(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        namespace: str = None,
        lifecycle_state: str = None,
        trust_tier: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        注入知识到 AGA Portal（文本形式，Portal 负责编码）
        
        ✅ 推荐 API：治理系统应使用此方法。
        
        架构原则：
        - 治理系统 = 规则与授权的"主权层"，只传递语义描述
        - AGA Portal = 规则被执行的"计算层"，负责 KV 编码
        
        Args:
            lu_id: Learning Unit ID
            condition: 触发条件描述（文本）
            decision: 决策描述（文本）
            namespace: 命名空间
            lifecycle_state: 初始状态
            trust_tier: 信任层级
            metadata: 扩展元数据
        
        Returns:
            注入结果
        """
        response = self._client.post(
            "/knowledge/inject-text",
            json={
                "lu_id": lu_id,
                "condition": condition,
                "decision": decision,
                "namespace": namespace or self.default_namespace,
                "lifecycle_state": lifecycle_state or settings.AGA_INITIAL_LIFECYCLE,
                "trust_tier": trust_tier,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return response.json()
    
    @retry(
        stop=stop_after_attempt(settings.AGA_PORTAL_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.AGA_PORTAL_RETRY_DELAY),
    )
    def inject_knowledge(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        namespace: str = None,
        lifecycle_state: str = None,
        trust_tier: str = None,
        metadata: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        注入知识到 AGA Portal（包含向量）
        
        ⚠️ 内部使用：此方法要求调用方提供预编码的向量。
        治理系统应使用 inject_knowledge_text() 方法。
        
        Args:
            lu_id: Learning Unit ID
            condition: 触发条件描述
            decision: 决策描述
            key_vector: 条件编码向量
            value_vector: 决策编码向量
            namespace: 命名空间
            lifecycle_state: 初始状态
            trust_tier: 信任层级
            metadata: 扩展元数据
        
        Returns:
            注入结果
        """
        response = self._client.post(
            "/knowledge/inject",
            json={
                "lu_id": lu_id,
                "condition": condition,
                "decision": decision,
                "key_vector": key_vector,
                "value_vector": value_vector,
                "namespace": namespace or self.default_namespace,
                "lifecycle_state": lifecycle_state or settings.AGA_INITIAL_LIFECYCLE,
                "trust_tier": trust_tier,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def batch_inject_text(
        self,
        items: List[Dict[str, Any]],
        namespace: str = None,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        批量注入知识（文本形式，Portal 负责编码）
        
        ✅ 推荐 API：治理系统应使用此方法。
        """
        response = self._client.post(
            "/knowledge/batch-text",
            json={
                "items": items,
                "namespace": namespace or self.default_namespace,
                "skip_duplicates": skip_duplicates,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = None,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """
        批量注入知识（包含向量）
        
        ⚠️ 内部使用：此方法要求调用方提供预编码的向量。
        治理系统应使用 batch_inject_text() 方法。
        """
        response = self._client.post(
            "/knowledge/batch",
            json={
                "items": items,
                "namespace": namespace or self.default_namespace,
                "skip_duplicates": skip_duplicates,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def get_knowledge(
        self,
        lu_id: str,
        namespace: str = None,
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """获取单个知识"""
        try:
            ns = namespace or self.default_namespace
            response = self._client.get(
                f"/knowledge/{ns}/{lu_id}",
                params={"include_vectors": include_vectors},
            )
            response.raise_for_status()
            result = response.json()
            return result.get("data")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def query_knowledge(
        self,
        namespace: str = None,
        lifecycle_states: List[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """查询知识列表"""
        params = {"limit": limit, "offset": offset}
        if lifecycle_states:
            params["lifecycle_states"] = ",".join(lifecycle_states)
        
        ns = namespace or self.default_namespace
        response = self._client.get(f"/knowledge/{ns}", params=params)
        response.raise_for_status()
        return response.json()
    
    def delete_knowledge(
        self,
        lu_id: str,
        namespace: str = None,
        reason: str = None,
    ) -> Dict[str, Any]:
        """删除知识"""
        ns = namespace or self.default_namespace
        params = {"reason": reason} if reason else {}
        response = self._client.delete(f"/knowledge/{ns}/{lu_id}", params=params)
        response.raise_for_status()
        return response.json()
    
    # ==================== 生命周期管理 ====================
    
    @retry(
        stop=stop_after_attempt(settings.AGA_PORTAL_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.AGA_PORTAL_RETRY_DELAY),
    )
    def update_lifecycle(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = None,
        reason: str = None,
    ) -> Dict[str, Any]:
        """更新生命周期状态"""
        response = self._client.put(
            "/lifecycle/update",
            json={
                "lu_id": lu_id,
                "new_state": new_state,
                "namespace": namespace or self.default_namespace,
                "reason": reason,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def confirm(self, lu_id: str, namespace: str = None, reason: str = None) -> Dict[str, Any]:
        """确认知识"""
        return self.update_lifecycle(lu_id, AGALifecycleState.CONFIRMED, namespace, reason)
    
    def deprecate(self, lu_id: str, namespace: str = None, reason: str = None) -> Dict[str, Any]:
        """弃用知识"""
        return self.update_lifecycle(lu_id, AGALifecycleState.DEPRECATED, namespace, reason)
    
    @retry(
        stop=stop_after_attempt(settings.AGA_PORTAL_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=settings.AGA_PORTAL_RETRY_DELAY),
    )
    def quarantine(
        self,
        lu_id: str,
        reason: str,
        namespace: str = None,
    ) -> Dict[str, Any]:
        """隔离知识"""
        response = self._client.post(
            "/lifecycle/quarantine",
            json={
                "lu_id": lu_id,
                "reason": reason,
                "namespace": namespace or self.default_namespace,
            },
        )
        response.raise_for_status()
        return response.json()
    
    # ==================== 统计和审计 ====================
    
    def get_statistics(self, namespace: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        if namespace:
            response = self._client.get(f"/statistics/{namespace}")
        else:
            response = self._client.get("/statistics")
        response.raise_for_status()
        return response.json()
    
    def get_audit_log(
        self,
        namespace: str = None,
        lu_id: str = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """获取审计日志"""
        params = {"limit": limit, "offset": offset}
        if namespace:
            params["namespace"] = namespace
        if lu_id:
            params["lu_id"] = lu_id
        
        response = self._client.get("/audit", params=params)
        response.raise_for_status()
        return response.json()


# ============================================================
# 知识编码器 - 基础接口与工厂
# ============================================================

from abc import ABC, abstractmethod
from enum import Enum


class EncoderType(str, Enum):
    """
    编码器类型枚举
    
    ⚠️ 编码器一致性说明：
    
    AGA 的 Key-Value 匹配机制要求：
    - **注入时的编码器** 与 **推理时的编码器** 必须一致
    - 不同编码器产生的向量空间不同，混用会导致匹配失败
    - 编码器不必与 AGA 绑定的 LLM（生成模型）一致
    
    推荐做法：
    1. 同一命名空间内的知识使用相同编码器
    2. 在 AGA Portal 配置中记录使用的编码器类型
    3. 迁移编码器时需要重新编码所有知识
    """
    HASH = "hash"                           # 哈希编码（测试用，无语义）
    OPENAI = "openai"                       # OpenAI text-embedding
    OPENAI_COMPATIBLE = "openai_compatible" # OpenAI 兼容 API（DeepSeek/Qwen/GLM/Moonshot 等）
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # HuggingFace SentenceTransformers
    COHERE = "cohere"                       # Cohere embed
    BAIDU_ERNIE = "baidu_ernie"             # 百度文心 ERNIE
    AZURE_OPENAI = "azure_openai"           # Azure OpenAI
    OLLAMA = "ollama"                       # Ollama 本地模型
    VLLM = "vllm"                           # vLLM 本地部署
    CUSTOM = "custom"                       # 自定义实现


class BaseKnowledgeEncoder(ABC):
    """
    知识编码器基础接口
    
    将约束文本编码为向量表示，用于 AGA 的知识注入。
    
    实现要求:
    - encode_text(): 编码单个文本
    - encode_constraint(): 编码约束对（条件+决策）
    - 支持批量编码以提高效率
    """
    
    def __init__(
        self,
        key_dim: int = None,
        value_dim: int = None,
    ):
        """
        初始化编码器
        
        Args:
            key_dim: Key 向量维度（条件编码）
            value_dim: Value 向量维度（决策编码）
        """
        self.key_dim = key_dim or settings.AGA_ENCODING_BOTTLENECK_DIM
        self.value_dim = value_dim or settings.AGA_ENCODING_HIDDEN_DIM
    
    @property
    @abstractmethod
    def encoder_type(self) -> EncoderType:
        """编码器类型"""
        pass
    
    @property
    @abstractmethod
    def native_dim(self) -> int:
        """原生编码维度"""
        pass
    
    @property
    def is_available(self) -> bool:
        """编码器是否可用"""
        return True
    
    @abstractmethod
    def encode_text(self, text: str) -> List[float]:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            编码向量
        """
        pass
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码文本（默认逐个编码，子类可优化）
        
        Args:
            texts: 文本列表
            
        Returns:
            编码向量列表
        """
        return [self.encode_text(t) for t in texts]
    
    def encode_constraint(
        self,
        condition: str,
        decision: str,
    ) -> Tuple[List[float], List[float]]:
        """
        编码约束为 key-value 向量对
        
        Args:
            condition: 条件文本
            decision: 决策文本
        
        Returns:
            (key_vector, value_vector)
        """
        key_vector = self.encode_text(condition)
        value_vector = self.encode_text(decision)
        
        # 调整维度
        key_vector = self._adjust_dim(key_vector, self.key_dim)
        value_vector = self._adjust_dim(value_vector, self.value_dim)
        
        return key_vector, value_vector
    
    def _adjust_dim(self, vector: List[float], target_dim: int) -> List[float]:
        """
        调整向量维度
        
        Args:
            vector: 原始向量
            target_dim: 目标维度
            
        Returns:
            调整后的向量
        """
        current_dim = len(vector)
        
        if current_dim == target_dim:
            return vector
        
        if current_dim > target_dim:
            # 截断 + 平均池化
            return self._reduce_dim(vector, target_dim)
        else:
            # 填充
            return vector + [0.0] * (target_dim - current_dim)
    
    def _reduce_dim(self, vector: List[float], target_dim: int) -> List[float]:
        """降维（平均池化）"""
        current_dim = len(vector)
        chunk_size = current_dim / target_dim
        
        result = []
        for i in range(target_dim):
            start = int(i * chunk_size)
            end = int((i + 1) * chunk_size)
            chunk = vector[start:end]
            result.append(sum(chunk) / len(chunk) if chunk else 0.0)
        
        return result


class HashKnowledgeEncoder(BaseKnowledgeEncoder):
    """
    哈希知识编码器
    
    将文本编码为确定性哈希向量。
    
    特点:
    - 不需要外部依赖
    - 确定性输出（相同输入产生相同输出）
    - 无语义理解能力
    
    适用场景:
    - 开发和测试
    - 离线环境
    - 简单的精确匹配场景
    """
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.HASH
    
    @property
    def native_dim(self) -> int:
        return 256  # SHAKE256 可生成任意长度
    
    def encode_text(self, text: str) -> List[float]:
        """使用 SHAKE256 哈希编码文本"""
        import struct
        
        # 计算哈希
        seed = hashlib.sha256(text.encode()).digest()
        
        # 扩展到目标维度
        dim = max(self.key_dim, self.value_dim)
        extended = hashlib.shake_256(seed).digest(dim * 4)
        
        # 转换为 float 列表（范围 [-0.1, 0.1]）
        vector = []
        for i in range(dim):
            val = struct.unpack('<I', extended[i*4:(i+1)*4])[0]
            normalized = (val / 0xFFFFFFFF - 0.5) * 0.2
            vector.append(normalized)
        
        return vector


class OpenAIEncoder(BaseKnowledgeEncoder):
    """
    OpenAI Embeddings 编码器
    
    使用 OpenAI 的 text-embedding 模型生成语义向量。
    
    支持模型:
    - text-embedding-3-small (1536 维，推荐)
    - text-embedding-3-large (3072 维)
    - text-embedding-ada-002 (1536 维，旧版)
    
    需要配置:
    - OPENAI_API_KEY: API 密钥
    - OPENAI_EMBEDDING_MODEL: 模型名称（可选）
    """
    
    DEFAULT_MODEL = "text-embedding-3-small"
    MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        base_url: str = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.api_key = api_key or getattr(settings, 'OPENAI_API_KEY', None)
        self.model = model or getattr(settings, 'OPENAI_EMBEDDING_MODEL', self.DEFAULT_MODEL)
        self.base_url = base_url or getattr(settings, 'OPENAI_BASE_URL', None)
        
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.OPENAI
    
    @property
    def native_dim(self) -> int:
        return self.MODEL_DIMS.get(self.model, 1536)
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _get_client(self):
        """延迟初始化 OpenAI 客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode_text(self, text: str) -> List[float]:
        """使用 OpenAI API 编码文本"""
        if not self.is_available:
            raise ValueError("OpenAI API key not configured")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        return response.data[0].embedding
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """批量编码（OpenAI API 支持批量）"""
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("OpenAI API key not configured")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        # 按原始顺序返回
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]


class AzureOpenAIEncoder(BaseKnowledgeEncoder):
    """
    Azure OpenAI Embeddings 编码器
    
    使用 Azure 部署的 OpenAI 模型。
    
    需要配置:
    - AZURE_OPENAI_API_KEY: API 密钥
    - AZURE_OPENAI_ENDPOINT: 端点 URL
    - AZURE_OPENAI_DEPLOYMENT: 部署名称
    - AZURE_OPENAI_API_VERSION: API 版本
    """
    
    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        deployment: str = None,
        api_version: str = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.api_key = api_key or getattr(settings, 'AZURE_OPENAI_API_KEY', None)
        self.endpoint = endpoint or getattr(settings, 'AZURE_OPENAI_ENDPOINT', None)
        self.deployment = deployment or getattr(settings, 'AZURE_OPENAI_DEPLOYMENT', None)
        self.api_version = api_version or getattr(settings, 'AZURE_OPENAI_API_VERSION', '2024-02-01')
        
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.AZURE_OPENAI
    
    @property
    def native_dim(self) -> int:
        return 1536  # 根据部署的模型而定
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.endpoint,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode_text(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("Azure OpenAI not configured")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.deployment,
            input=text,
        )
        
        return response.data[0].embedding
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("Azure OpenAI not configured")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.deployment,
            input=texts,
        )
        
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]


class SentenceTransformersEncoder(BaseKnowledgeEncoder):
    """
    HuggingFace Sentence Transformers 编码器
    
    使用本地运行的 Sentence Transformers 模型。
    
    推荐模型:
    - all-MiniLM-L6-v2 (384 维，快速)
    - all-mpnet-base-v2 (768 维，高质量)
    - paraphrase-multilingual-MiniLM-L12-v2 (384 维，多语言)
    - bge-small-zh-v1.5 (512 维，中文)
    - bge-large-zh-v1.5 (1024 维，中文高质量)
    
    需要安装:
    - pip install sentence-transformers
    """
    
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    MODEL_DIMS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "bge-small-zh-v1.5": 512,
        "bge-large-zh-v1.5": 1024,
        "text2vec-base-chinese": 768,
    }
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.model_name = model_name or getattr(settings, 'SENTENCE_TRANSFORMERS_MODEL', self.DEFAULT_MODEL)
        self.device = device or getattr(settings, 'SENTENCE_TRANSFORMERS_DEVICE', None)
        
        self._model = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.SENTENCE_TRANSFORMERS
    
    @property
    def native_dim(self) -> int:
        return self.MODEL_DIMS.get(self.model_name, 384)
    
    @property
    def is_available(self) -> bool:
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name, device=self.device)
            except ImportError:
                raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        return self._model
    
    def encode_text(self, text: str) -> List[float]:
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class CohereEncoder(BaseKnowledgeEncoder):
    """
    Cohere Embeddings 编码器
    
    使用 Cohere 的 embed 模型。
    
    支持模型:
    - embed-english-v3.0 (1024 维)
    - embed-multilingual-v3.0 (1024 维，多语言)
    - embed-english-light-v3.0 (384 维，轻量)
    
    需要配置:
    - COHERE_API_KEY: API 密钥
    """
    
    DEFAULT_MODEL = "embed-multilingual-v3.0"
    MODEL_DIMS = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
    }
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        input_type: str = "search_document",
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.api_key = api_key or getattr(settings, 'COHERE_API_KEY', None)
        self.model = model or getattr(settings, 'COHERE_EMBEDDING_MODEL', self.DEFAULT_MODEL)
        self.input_type = input_type
        
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.COHERE
    
    @property
    def native_dim(self) -> int:
        return self.MODEL_DIMS.get(self.model, 1024)
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def _get_client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise ImportError("请安装 cohere: pip install cohere")
        return self._client
    
    def encode_text(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("Cohere API key not configured")
        
        client = self._get_client()
        
        response = client.embed(
            texts=[text],
            model=self.model,
            input_type=self.input_type,
        )
        
        return response.embeddings[0]
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("Cohere API key not configured")
        
        client = self._get_client()
        
        response = client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
        )
        
        return response.embeddings


class BaiduErnieEncoder(BaseKnowledgeEncoder):
    """
    百度文心 ERNIE Embedding 编码器
    
    使用百度文心一言的 Embedding 服务。
    
    支持模型:
    - Embedding-V1 (384 维)
    
    需要配置:
    - BAIDU_API_KEY: API Key
    - BAIDU_SECRET_KEY: Secret Key
    """
    
    TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
    EMBEDDING_URL = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1"
    
    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.api_key = api_key or getattr(settings, 'BAIDU_API_KEY', None)
        self.secret_key = secret_key or getattr(settings, 'BAIDU_SECRET_KEY', None)
        
        self._access_token = None
        self._token_expires_at = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.BAIDU_ERNIE
    
    @property
    def native_dim(self) -> int:
        return 384
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.secret_key)
    
    def _get_access_token(self) -> str:
        """获取 access_token"""
        import time
        
        # 检查缓存的 token 是否有效
        if self._access_token and self._token_expires_at:
            if time.time() < self._token_expires_at - 60:  # 提前 60 秒刷新
                return self._access_token
        
        # 获取新 token
        response = httpx.post(
            self.TOKEN_URL,
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key,
            },
        )
        response.raise_for_status()
        data = response.json()
        
        self._access_token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 2592000)
        
        return self._access_token
    
    def encode_text(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("Baidu API credentials not configured")
        
        token = self._get_access_token()
        
        response = httpx.post(
            self.EMBEDDING_URL,
            params={"access_token": token},
            json={"input": [text]},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        
        if "error_code" in data:
            raise ValueError(f"Baidu API error: {data.get('error_msg', data['error_code'])}")
        
        return data["data"][0]["embedding"]
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("Baidu API credentials not configured")
        
        token = self._get_access_token()
        
        # 百度 API 每次最多 16 条
        batch_size = 16
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = httpx.post(
                self.EMBEDDING_URL,
                params={"access_token": token},
                json={"input": batch},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            
            if "error_code" in data:
                raise ValueError(f"Baidu API error: {data.get('error_msg', data['error_code'])}")
            
            # 按 index 排序
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in sorted_data])
        
        return all_embeddings


class OllamaEncoder(BaseKnowledgeEncoder):
    """
    Ollama 本地 Embedding 编码器
    
    使用 Ollama 运行的本地模型生成 embedding。
    
    支持模型:
    - nomic-embed-text (768 维)
    - mxbai-embed-large (1024 维)
    - all-minilm (384 维)
    
    需要配置:
    - OLLAMA_BASE_URL: Ollama API 地址 (默认 http://localhost:11434)
    - OLLAMA_EMBEDDING_MODEL: 模型名称
    """
    
    DEFAULT_MODEL = "nomic-embed-text"
    MODEL_DIMS = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
    }
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.base_url = (base_url or getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')).rstrip('/')
        self.model = model or getattr(settings, 'OLLAMA_EMBEDDING_MODEL', self.DEFAULT_MODEL)
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.OLLAMA
    
    @property
    def native_dim(self) -> int:
        return self.MODEL_DIMS.get(self.model, 768)
    
    @property
    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    def encode_text(self, text: str) -> List[float]:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        
        return data["embedding"]
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        # Ollama 不支持批量，逐个处理
        return [self.encode_text(t) for t in texts]


class OpenAICompatibleEncoder(BaseKnowledgeEncoder):
    """
    OpenAI 兼容 API 编码器
    
    支持所有提供 OpenAI 兼容 Embedding API 的服务商。
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │  支持的服务商                                                       │
    ├─────────────┬──────────────────────┬────────┬───────────────────────┤
    │  服务商     │  Embedding 模型       │  维度  │  API 端点              │
    ├─────────────┼──────────────────────┼────────┼───────────────────────┤
    │  DeepSeek   │  deepseek-embedding  │  1024  │  api.deepseek.com     │
    │  通义千问   │  text-embedding-v2   │  1536  │  dashscope.aliyuncs   │
    │  智谱 GLM   │  embedding-2/3       │  1024  │  open.bigmodel.cn     │
    │  Moonshot   │  moonshot-embedding  │  1024  │  api.moonshot.cn      │
    │  百川       │  baichuan-embedding  │  1024  │  api.baichuan-ai.com  │
    │  零一万物   │  yi-large-embedding  │  2048  │  api.01.ai            │
    │  MiniMax    │  embo-01             │  1536  │  api.minimax.chat     │
    │  SiliconFlow│  多种模型            │  vary  │  api.siliconflow.cn   │
    └─────────────┴──────────────────────┴────────┴───────────────────────┘
    
    配置方式:
    - OPENAI_COMPATIBLE_BASE_URL: API 端点
    - OPENAI_COMPATIBLE_API_KEY: API 密钥
    - OPENAI_COMPATIBLE_MODEL: 模型名称
    
    使用示例:
    ```python
    # DeepSeek
    encoder = OpenAICompatibleEncoder(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-xxx",
        model="deepseek-embedding",
    )
    
    # 通义千问
    encoder = OpenAICompatibleEncoder(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-xxx",
        model="text-embedding-v2",
    )
    
    # 智谱 GLM
    encoder = OpenAICompatibleEncoder(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key="xxx.xxx",
        model="embedding-3",
    )
    ```
    """
    
    # 预设的服务商配置
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-embedding",
            "dim": 1024,
        },
        "qwen": {
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "text-embedding-v2",
            "dim": 1536,
        },
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/paas/v4",
            "model": "embedding-3",
            "dim": 2048,
        },
        "moonshot": {
            "base_url": "https://api.moonshot.cn/v1",
            "model": "moonshot-embedding",
            "dim": 1024,
        },
        "baichuan": {
            "base_url": "https://api.baichuan-ai.com/v1",
            "model": "Baichuan-Text-Embedding",
            "dim": 1024,
        },
        "yi": {
            "base_url": "https://api.01.ai/v1",
            "model": "yi-large-embedding",
            "dim": 2048,
        },
        "minimax": {
            "base_url": "https://api.minimax.chat/v1",
            "model": "embo-01",
            "dim": 1536,
        },
        "siliconflow": {
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "BAAI/bge-large-zh-v1.5",
            "dim": 1024,
        },
    }
    
    # 常见模型的维度
    MODEL_DIMS = {
        # DeepSeek
        "deepseek-embedding": 1024,
        # 通义千问
        "text-embedding-v1": 1536,
        "text-embedding-v2": 1536,
        "text-embedding-v3": 1024,
        # 智谱 GLM
        "embedding-2": 1024,
        "embedding-3": 2048,
        # Moonshot
        "moonshot-embedding": 1024,
        # 百川
        "Baichuan-Text-Embedding": 1024,
        "baichuan-embedding": 1024,
        # 零一万物
        "yi-large-embedding": 2048,
        # MiniMax
        "embo-01": 1536,
        # SiliconFlow / BGE
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
    }
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        provider: str = None,
        native_dim: int = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        """
        初始化 OpenAI 兼容编码器
        
        Args:
            base_url: API 端点
            api_key: API 密钥
            model: 模型名称
            provider: 预设服务商名称（可选，简化配置）
            native_dim: 原生维度（如果模型不在预设列表中）
            key_dim: Key 向量目标维度
            value_dim: Value 向量目标维度
        """
        super().__init__(key_dim, value_dim)
        
        # 如果指定了 provider，使用预设配置
        if provider and provider in self.PROVIDERS:
            preset = self.PROVIDERS[provider]
            base_url = base_url or preset["base_url"]
            model = model or preset["model"]
            native_dim = native_dim or preset["dim"]
        
        self.base_url = base_url or getattr(settings, 'OPENAI_COMPATIBLE_BASE_URL', None)
        self.api_key = api_key or getattr(settings, 'OPENAI_COMPATIBLE_API_KEY', None)
        self.model = model or getattr(settings, 'OPENAI_COMPATIBLE_MODEL', 'text-embedding-v2')
        self._native_dim = native_dim
        self.provider = provider
        
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.OPENAI_COMPATIBLE
    
    @property
    def native_dim(self) -> int:
        if self._native_dim:
            return self._native_dim
        return self.MODEL_DIMS.get(self.model, 1024)
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)
    
    def _get_client(self):
        """延迟初始化 OpenAI 兼容客户端"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode_text(self, text: str) -> List[float]:
        if not self.is_available:
            raise ValueError("OpenAI compatible API not configured")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        return response.data[0].embedding
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        if not self.is_available:
            raise ValueError("OpenAI compatible API not configured")
        
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [d.embedding for d in sorted_data]
    
    @classmethod
    def from_provider(cls, provider: str, api_key: str, **kwargs) -> "OpenAICompatibleEncoder":
        """
        从预设服务商创建编码器
        
        Args:
            provider: 服务商名称 (deepseek/qwen/zhipu/moonshot/baichuan/yi/minimax/siliconflow)
            api_key: API 密钥
            **kwargs: 其他参数
            
        Returns:
            编码器实例
        """
        if provider not in cls.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(cls.PROVIDERS.keys())}")
        
        return cls(provider=provider, api_key=api_key, **kwargs)


class VLLMEncoder(BaseKnowledgeEncoder):
    """
    vLLM Embedding 编码器
    
    使用 vLLM 部署的本地模型生成 embedding。
    vLLM 提供高性能的批量推理，适合大规模知识注入。
    
    推荐模型:
    - BAAI/bge-large-zh-v1.5 (1024 维，中文)
    - BAAI/bge-m3 (1024 维，多语言)
    - intfloat/multilingual-e5-large (1024 维，多语言)
    - sentence-transformers/all-MiniLM-L6-v2 (384 维)
    
    vLLM 启动方式:
    ```bash
    python -m vllm.entrypoints.openai.api_server \\
        --model BAAI/bge-large-zh-v1.5 \\
        --host 0.0.0.0 --port 8000 \\
        --task embed
    ```
    
    需要配置:
    - VLLM_BASE_URL: vLLM API 地址 (默认 http://localhost:8000)
    - VLLM_EMBEDDING_MODEL: 模型名称
    """
    
    DEFAULT_MODEL = "BAAI/bge-large-zh-v1.5"
    MODEL_DIMS = {
        "BAAI/bge-large-zh-v1.5": 1024,
        "BAAI/bge-m3": 1024,
        "BAAI/bge-small-zh-v1.5": 512,
        "intfloat/multilingual-e5-large": 1024,
        "intfloat/e5-large-v2": 1024,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/all-mpnet-base-v2": 768,
    }
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        native_dim: int = None,
        key_dim: int = None,
        value_dim: int = None,
    ):
        super().__init__(key_dim, value_dim)
        
        self.base_url = (base_url or getattr(settings, 'VLLM_BASE_URL', 'http://localhost:8000')).rstrip('/')
        self.model = model or getattr(settings, 'VLLM_EMBEDDING_MODEL', self.DEFAULT_MODEL)
        self._native_dim = native_dim
        
        self._client = None
    
    @property
    def encoder_type(self) -> EncoderType:
        return EncoderType.VLLM
    
    @property
    def native_dim(self) -> int:
        if self._native_dim:
            return self._native_dim
        return self.MODEL_DIMS.get(self.model, 1024)
    
    @property
    def is_available(self) -> bool:
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            # 尝试 OpenAI 兼容的模型列表端点
            try:
                response = httpx.get(f"{self.base_url}/v1/models", timeout=5.0)
                return response.status_code == 200
            except:
                return False
    
    def _get_client(self):
        """使用 OpenAI 客户端（vLLM 兼容）"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key="EMPTY",  # vLLM 不需要 API key
                    base_url=f"{self.base_url}/v1",
                )
            except ImportError:
                raise ImportError("请安装 openai: pip install openai")
        return self._client
    
    def encode_text(self, text: str) -> List[float]:
        client = self._get_client()
        
        response = client.embeddings.create(
            model=self.model,
            input=text,
        )
        
        return response.data[0].embedding
    
    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量编码（vLLM 支持高效批量处理）
        """
        if not texts:
            return []
        
        client = self._get_client()
        
        # vLLM 支持大批量，但为安全起见分批处理
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            
            sorted_data = sorted(response.data, key=lambda x: x.index)
            all_embeddings.extend([d.embedding for d in sorted_data])
        
        return all_embeddings


# ============================================================
# 编码器一致性检查
# ============================================================

@dataclass
class EncoderSignature:
    """
    编码器签名
    
    用于验证编码器一致性，确保注入和查询使用相同的编码器。
    """
    encoder_type: str
    model: str
    native_dim: int
    provider: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_type": self.encoder_type,
            "model": self.model,
            "native_dim": self.native_dim,
            "provider": self.provider,
        }
    
    @classmethod
    def from_encoder(cls, encoder: BaseKnowledgeEncoder) -> "EncoderSignature":
        """从编码器实例创建签名"""
        model = getattr(encoder, 'model', None) or getattr(encoder, 'model_name', 'unknown')
        provider = getattr(encoder, 'provider', None)
        
        return cls(
            encoder_type=encoder.encoder_type.value,
            model=model,
            native_dim=encoder.native_dim,
            provider=provider,
        )
    
    def is_compatible(self, other: "EncoderSignature") -> bool:
        """
        检查两个编码器签名是否兼容
        
        兼容条件：
        1. 编码器类型相同
        2. 模型名称相同
        3. 原生维度相同
        """
        return (
            self.encoder_type == other.encoder_type and
            self.model == other.model and
            self.native_dim == other.native_dim
        )


def verify_encoder_consistency(
    encoder: BaseKnowledgeEncoder,
    expected_signature: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    验证编码器一致性
    
    Args:
        encoder: 当前编码器
        expected_signature: 期望的编码器签名（通常从 AGA Portal 获取）
        
    Returns:
        (is_consistent, message)
    """
    current = EncoderSignature.from_encoder(encoder)
    expected = EncoderSignature(
        encoder_type=expected_signature.get("encoder_type", ""),
        model=expected_signature.get("model", ""),
        native_dim=expected_signature.get("native_dim", 0),
        provider=expected_signature.get("provider"),
    )
    
    if current.is_compatible(expected):
        return True, "Encoder is consistent"
    
    # 构建详细的不一致信息
    mismatches = []
    if current.encoder_type != expected.encoder_type:
        mismatches.append(f"type: {current.encoder_type} vs {expected.encoder_type}")
    if current.model != expected.model:
        mismatches.append(f"model: {current.model} vs {expected.model}")
    if current.native_dim != expected.native_dim:
        mismatches.append(f"dim: {current.native_dim} vs {expected.native_dim}")
    
    return False, f"Encoder mismatch: {', '.join(mismatches)}"


# ============================================================
# 编码器工厂
# ============================================================

class KnowledgeEncoderFactory:
    """
    知识编码器工厂
    
    根据配置创建合适的编码器实例。
    
    使用方式:
    ```python
    # 使用默认编码器（根据配置自动选择）
    encoder = KnowledgeEncoderFactory.create()
    
    # 指定编码器类型
    encoder = KnowledgeEncoderFactory.create(EncoderType.OPENAI)
    
    # 自动选择可用的最佳编码器
    encoder = KnowledgeEncoderFactory.create_best_available()
    ```
    """
    
    # 编码器优先级（用于自动选择）
    ENCODER_PRIORITY = [
        EncoderType.OPENAI,
        EncoderType.AZURE_OPENAI,
        EncoderType.OPENAI_COMPATIBLE,
        EncoderType.COHERE,
        EncoderType.BAIDU_ERNIE,
        EncoderType.VLLM,
        EncoderType.SENTENCE_TRANSFORMERS,
        EncoderType.OLLAMA,
        EncoderType.HASH,
    ]
    
    # 编码器类映射
    ENCODER_CLASSES: Dict[EncoderType, type] = {
        EncoderType.HASH: HashKnowledgeEncoder,
        EncoderType.OPENAI: OpenAIEncoder,
        EncoderType.OPENAI_COMPATIBLE: OpenAICompatibleEncoder,
        EncoderType.AZURE_OPENAI: AzureOpenAIEncoder,
        EncoderType.SENTENCE_TRANSFORMERS: SentenceTransformersEncoder,
        EncoderType.COHERE: CohereEncoder,
        EncoderType.BAIDU_ERNIE: BaiduErnieEncoder,
        EncoderType.OLLAMA: OllamaEncoder,
        EncoderType.VLLM: VLLMEncoder,
    }
    
    # 自定义编码器注册
    _custom_encoders: Dict[str, type] = {}
    
    @classmethod
    def create(
        cls,
        encoder_type: EncoderType = None,
        **kwargs
    ) -> BaseKnowledgeEncoder:
        """
        创建编码器实例
        
        Args:
            encoder_type: 编码器类型，不指定则使用配置或默认值
            **kwargs: 传递给编码器构造函数的参数
            
        Returns:
            编码器实例
        """
        if encoder_type is None:
            # 从配置读取
            configured_type = getattr(settings, 'KNOWLEDGE_ENCODER_TYPE', 'hash')
            try:
                encoder_type = EncoderType(configured_type)
            except ValueError:
                logger.warning(f"Unknown encoder type '{configured_type}', using hash")
                encoder_type = EncoderType.HASH
        
        encoder_class = cls.ENCODER_CLASSES.get(encoder_type)
        
        if encoder_class is None:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        return encoder_class(**kwargs)
    
    @classmethod
    def create_best_available(cls, **kwargs) -> BaseKnowledgeEncoder:
        """
        创建最佳可用编码器
        
        按优先级尝试创建编码器，返回第一个可用的。
        
        Returns:
            可用的最佳编码器实例
        """
        for encoder_type in cls.ENCODER_PRIORITY:
            try:
                encoder = cls.create(encoder_type, **kwargs)
                if encoder.is_available:
                    logger.info(f"Using {encoder_type.value} encoder")
                    return encoder
            except Exception as e:
                logger.debug(f"Encoder {encoder_type.value} not available: {e}")
                continue
        
        # 回退到哈希编码器（始终可用）
        logger.warning("No semantic encoder available, falling back to hash encoder")
        return HashKnowledgeEncoder(**kwargs)
    
    @classmethod
    def register_custom(cls, name: str, encoder_class: type):
        """
        注册自定义编码器
        
        Args:
            name: 编码器名称
            encoder_class: 编码器类（必须继承 BaseKnowledgeEncoder）
        """
        if not issubclass(encoder_class, BaseKnowledgeEncoder):
            raise TypeError("encoder_class must be a subclass of BaseKnowledgeEncoder")
        
        cls._custom_encoders[name] = encoder_class
        logger.info(f"Registered custom encoder: {name}")
    
    @classmethod
    def create_from_provider(
        cls,
        provider: str,
        api_key: str,
        **kwargs
    ) -> BaseKnowledgeEncoder:
        """
        从服务商名称创建编码器
        
        简化国产大模型的配置，只需提供服务商名称和 API 密钥。
        
        支持的服务商:
        - deepseek: DeepSeek
        - qwen: 通义千问
        - zhipu: 智谱 GLM
        - moonshot: Moonshot/Kimi
        - baichuan: 百川
        - yi: 零一万物
        - minimax: MiniMax
        - siliconflow: SiliconFlow (支持多种模型)
        
        Args:
            provider: 服务商名称
            api_key: API 密钥
            **kwargs: 其他参数（如 model, key_dim, value_dim）
            
        Returns:
            配置好的编码器实例
            
        Example:
            encoder = KnowledgeEncoderFactory.create_from_provider("deepseek", "sk-xxx")
        """
        return OpenAICompatibleEncoder.from_provider(provider, api_key, **kwargs)
    
    @classmethod
    def get_encoder_signature(cls, encoder: BaseKnowledgeEncoder) -> EncoderSignature:
        """
        获取编码器签名
        
        用于记录和验证编码器一致性。
        """
        return EncoderSignature.from_encoder(encoder)
    
    @classmethod
    def list_available(cls) -> List[Dict[str, Any]]:
        """
        列出所有可用的编码器
        
        Returns:
            编码器信息列表
        """
        result = []
        
        for encoder_type in EncoderType:
            if encoder_type == EncoderType.CUSTOM:
                continue
            
            try:
                encoder = cls.create(encoder_type)
                result.append({
                    "type": encoder_type.value,
                    "native_dim": encoder.native_dim,
                    "is_available": encoder.is_available,
                    "requires_api_key": encoder_type in [
                        EncoderType.OPENAI, 
                        EncoderType.AZURE_OPENAI,
                        EncoderType.OPENAI_COMPATIBLE,
                        EncoderType.COHERE, 
                        EncoderType.BAIDU_ERNIE
                    ],
                    "requires_local_model": encoder_type in [
                        EncoderType.SENTENCE_TRANSFORMERS,
                        EncoderType.OLLAMA,
                        EncoderType.VLLM,
                    ],
                })
            except Exception as e:
                result.append({
                    "type": encoder_type.value,
                    "is_available": False,
                    "error": str(e),
                })
        
        return result


# 为了向后兼容，保留 SimpleKnowledgeEncoder 作为 HashKnowledgeEncoder 的别名
SimpleKnowledgeEncoder = HashKnowledgeEncoder


# ============================================================
# 知识转移服务
# ============================================================

class KnowledgeTransferService:
    """
    知识转移服务
    
    将审批通过的 Learning Unit 通过 HTTP API 转移到 AGA Portal。
    
    架构说明:
        ┌─────────────────────────────────────────────────────────────┐
        │  治理系统（主权层）                                          │
        │  - 定义规则和授权                                           │
        │  - 只传递文本形式的 condition/decision                       │
        │  - 不涉及 KV 编码                                           │
        └─────────────────────────────────────────────────────────────┘
                                    │
                                    │ HTTP API (文本)
                                    ▼
        ┌─────────────────────────────────────────────────────────────┐
        │  AGA Portal（计算层）                                        │
        │  - 接收文本规则                                              │
        │  - 使用编码器将文本转换为 KV 向量                            │
        │  - 确保编码器一致性（注入与推理使用同一编码器）               │
        └─────────────────────────────────────────────────────────────┘
    
    这种架构确保:
        1. 治理系统与模型解耦（不依赖模型维度/结构）
        2. 编码器一致性由 Portal 单点保证
        3. 规则文本可审计
        4. 模型升级时治理系统无需改动
    
    使用示例:
    ```python
    service = KnowledgeTransferService(db)
    
    # 转移知识（文本形式，Portal 负责编码）
    result = service.transfer_to_aga(lu_id)
    ```
    """
    
    def __init__(
        self,
        db: Session,
        lu_service: Optional["LearningUnitService"] = None,
        portal_client: Optional[AGAPortalClient] = None,
    ):
        """
        初始化知识转移服务
        
        Args:
            db: 数据库会话
            lu_service: Learning Unit 服务（可选，延迟加载）
            portal_client: AGA Portal 客户端（可选，根据配置创建）
        
        注意:
            编码器已移至 AGA Portal 侧，治理系统不再负责 KV 编码。
            这确保了编码器一致性和系统解耦。
        """
        self.db = db
        self._lu_service = lu_service
        
        # 初始化 Portal 客户端
        if portal_client:
            self._portal_client = portal_client
        elif settings.AGA_PORTAL_ENABLED:
            self._portal_client = AGAPortalClient()
        else:
            self._portal_client = None
        
        logger.info("KnowledgeTransferService initialized (encoding delegated to AGA Portal)")
        
        # 转移记录（内存缓存，实际应持久化）
        self._transfer_records: List[Dict[str, Any]] = []
    
    @property
    def lu_service(self) -> "LearningUnitService":
        """延迟加载 LearningUnitService"""
        if self._lu_service is None:
            from .learning_unit_service import LearningUnitService
            self._lu_service = LearningUnitService(self.db)
        return self._lu_service
    
    @property
    def portal_available(self) -> bool:
        """检查 Portal 是否可用"""
        if self._portal_client is None:
            return False
        return self._portal_client.is_healthy()
    
    # ==================== 知识转移 ====================
    
    def transfer_to_aga(
        self,
        lu_id: UUID,
        initial_lifecycle: str = AGALifecycleState.PROBATIONARY,
        namespace: str = None,
    ) -> Dict[str, Any]:
        """
        将 Learning Unit 转移到 AGA Portal
        
        Args:
            lu_id: Learning Unit ID
            initial_lifecycle: 初始生命周期状态
            namespace: 命名空间
        
        Returns:
            转移结果
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        # 检查状态
        if lu.status not in [
            LearningUnitStatus.APPROVED.value,
            LearningUnitStatus.CORRECTED.value
        ]:
            raise BusinessError(
                f"Cannot transfer LU in status {lu.status}. Must be APPROVED or CORRECTED.",
                code="INVALID_STATUS"
            )
        
        if lu.is_internalized:
            raise BusinessError(
                f"LU {lu_id} is already internalized",
                code="ALREADY_INTERNALIZED"
            )
        
        # 获取约束
        constraints = self.lu_service.get_constraints(lu_id)
        if not constraints:
            raise BusinessError(
                f"LU {lu_id} has no constraints to transfer",
                code="NO_CONSTRAINTS"
            )
        
        # 构建转移结果
        transfer_result = {
            "lu_id": str(lu_id),
            "title": lu.title,
            "constraint_count": len(constraints),
            "initial_lifecycle": initial_lifecycle,
            "namespace": namespace or settings.AGA_PORTAL_NAMESPACE,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "pending",
            "injected_ids": [],
            "errors": [],
        }
        
        try:
            if self._portal_client and self.portal_available:
                # 通过 HTTP API 转移
                injected_ids = self._transfer_via_portal(
                    lu, constraints, initial_lifecycle, namespace
                )
                transfer_result["injected_ids"] = injected_ids
                transfer_result["status"] = "success"
                
                logger.info(f"Transferred LU {lu_id} to AGA Portal with {len(injected_ids)} constraints")
            else:
                # Portal 不可用，模拟转移
                transfer_result["status"] = "simulated"
                transfer_result["injected_ids"] = [f"{lu_id}_c{i}" for i in range(len(constraints))]
                logger.warning(f"AGA Portal not available, simulating transfer for LU {lu_id}")
            
            # 更新 LU 状态
            self.lu_service.mark_internalized(
                lu_id,
                aga_slot_mapping={"injected_ids": transfer_result["injected_ids"]},
                lifecycle_state=initial_lifecycle,
            )
            
        except Exception as e:
            transfer_result["status"] = "failed"
            transfer_result["errors"].append(str(e))
            logger.error(f"Transfer failed for LU {lu_id}: {e}")
            raise
        
        # 记录转移
        self._transfer_records.append(transfer_result)
        
        return transfer_result
    
    def _transfer_via_portal(
        self,
        lu: LearningUnit,
        constraints: List[LUConstraint],
        initial_lifecycle: str,
        namespace: str = None,
    ) -> List[str]:
        """
        通过 AGA Portal API 转移知识
        
        架构说明：
        - 治理系统只传递文本形式的 condition/decision
        - AGA Portal 负责 KV 编码（确保编码器一致性）
        - 这符合"治理系统=主权层，AGA=计算层"的架构原则
        
        Args:
            lu: Learning Unit
            constraints: 约束列表
            initial_lifecycle: 初始生命周期状态
            namespace: 命名空间
        
        Returns:
            成功注入的知识 ID 列表
        """
        ns = namespace or settings.AGA_PORTAL_NAMESPACE
        injected_ids = []
        
        for i, constraint in enumerate(constraints):
            condition = constraint.condition or ""
            decision = constraint.decision or ""
            
            if not condition or not decision:
                logger.warning(f"Skipping constraint {i} with empty condition/decision")
                continue
            
            # 构建知识 ID
            knowledge_id = f"{lu.id}_c{i}"
            
            try:
                # 调用 Portal API 注入（文本形式，Portal 负责编码）
                result = self._portal_client.inject_knowledge_text(
                    lu_id=knowledge_id,
                    condition=condition,
                    decision=decision,
                    namespace=ns,
                    lifecycle_state=initial_lifecycle,
                    metadata={
                        "source_lu_id": str(lu.id),
                        "constraint_index": i,
                        "confidence": constraint.confidence,
                        "rationale": constraint.rationale,
                        "transferred_at": datetime.utcnow().isoformat(),
                    },
                )
                
                if result.get("success"):
                    injected_ids.append(knowledge_id)
                    logger.debug(f"Injected knowledge {knowledge_id} to Portal (text mode)")
                else:
                    logger.warning(f"Failed to inject {knowledge_id}: {result}")
                    
            except Exception as e:
                logger.error(f"Error injecting {knowledge_id}: {e}")
                # 继续处理其他约束
        
        return injected_ids
    
    # ==================== 生命周期管理 ====================
    
    def confirm_knowledge(self, lu_id: UUID, actor: Optional[User] = None) -> Dict[str, Any]:
        """
        确认知识（试用期 → 已确认）
        
        当知识经过验证后，提升其可靠性。
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(f"LU {lu_id} is not internalized", code="NOT_INTERNALIZED")
        
        if lu.lifecycle_state == AGALifecycleState.CONFIRMED:
            raise BusinessError(f"LU {lu_id} is already confirmed", code="ALREADY_CONFIRMED")
        
        old_state = lu.lifecycle_state
        
        # 更新数据库
        self.lu_service.update_lifecycle_state(lu_id, AGALifecycleState.CONFIRMED, actor)
        
        # 更新 AGA Portal
        if self._portal_client and self.portal_available:
            try:
                # 获取所有相关知识 ID
                aga_mapping = lu.aga_slot_mapping or {}
                injected_ids = aga_mapping.get("injected_ids", [])
                
                for kid in injected_ids:
                    self._portal_client.confirm(
                        kid,
                        reason=f"Confirmed by {actor.username if actor else 'system'}"
                    )
                
                logger.info(f"Confirmed {len(injected_ids)} knowledge items in Portal for LU {lu_id}")
            except Exception as e:
                logger.error(f"Failed to confirm in AGA Portal: {e}")
        
        return {
            "lu_id": str(lu_id),
            "old_state": old_state,
            "new_state": AGALifecycleState.CONFIRMED,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def deprecate_knowledge(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """
        弃用知识
        
        降低知识的可靠性，但不完全移除。
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(f"LU {lu_id} is not internalized", code="NOT_INTERNALIZED")
        
        old_state = lu.lifecycle_state
        
        # 更新数据库
        self.lu_service.update_lifecycle_state(lu_id, AGALifecycleState.DEPRECATED, actor)
        
        # 更新 AGA Portal
        if self._portal_client and self.portal_available:
            try:
                aga_mapping = lu.aga_slot_mapping or {}
                injected_ids = aga_mapping.get("injected_ids", [])
                
                for kid in injected_ids:
                    self._portal_client.deprecate(kid, reason=reason)
                
                logger.info(f"Deprecated {len(injected_ids)} knowledge items for LU {lu_id}")
            except Exception as e:
                logger.error(f"Failed to deprecate in AGA Portal: {e}")
        
        return {
            "lu_id": str(lu_id),
            "old_state": old_state,
            "new_state": AGALifecycleState.DEPRECATED,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def quarantine_knowledge(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """
        隔离知识（立即移除影响）
        
        当发现知识有问题时，立即隔离。
        """
        lu = self.lu_service.get_learning_unit_or_404(lu_id)
        
        if not lu.is_internalized:
            raise BusinessError(f"LU {lu_id} is not internalized", code="NOT_INTERNALIZED")
        
        old_state = lu.lifecycle_state
        
        # 更新数据库
        self.lu_service.quarantine(lu_id, reason, actor)
        
        # 更新 AGA Portal
        if self._portal_client and self.portal_available:
            try:
                aga_mapping = lu.aga_slot_mapping or {}
                injected_ids = aga_mapping.get("injected_ids", [])
                
                for kid in injected_ids:
                    self._portal_client.quarantine(kid, reason=reason)
                
                logger.warning(f"Quarantined {len(injected_ids)} knowledge items for LU {lu_id}")
            except Exception as e:
                logger.error(f"Failed to quarantine in AGA Portal: {e}")
        
        return {
            "lu_id": str(lu_id),
            "old_state": old_state,
            "new_state": AGALifecycleState.QUARANTINED,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def rollback_knowledge(
        self,
        lu_id: UUID,
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """
        回滚知识（等同于隔离）
        """
        return self.quarantine_knowledge(lu_id, f"Rollback: {reason}", actor)
    
    # ==================== 批量操作 ====================
    
    def batch_transfer(
        self,
        lu_ids: List[UUID],
        initial_lifecycle: str = AGALifecycleState.PROBATIONARY,
        namespace: str = None,
    ) -> Dict[str, Any]:
        """批量转移"""
        results = {
            "total": len(lu_ids),
            "success": 0,
            "failed": 0,
            "details": [],
        }
        
        for lu_id in lu_ids:
            try:
                result = self.transfer_to_aga(lu_id, initial_lifecycle, namespace)
                results["success"] += 1
                results["details"].append({
                    "lu_id": str(lu_id),
                    "status": "success",
                    "result": result,
                })
            except Exception as e:
                results["failed"] += 1
                results["details"].append({
                    "lu_id": str(lu_id),
                    "status": "failed",
                    "error": str(e),
                })
        
        return results
    
    def batch_confirm(self, lu_ids: List[UUID], actor: Optional[User] = None) -> Dict[str, Any]:
        """批量确认"""
        results = {"success": 0, "failed": 0, "details": []}
        
        for lu_id in lu_ids:
            try:
                result = self.confirm_knowledge(lu_id, actor)
                results["success"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "success"})
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "failed", "error": str(e)})
        
        return results
    
    def batch_quarantine(
        self,
        lu_ids: List[UUID],
        reason: str,
        actor: Optional[User] = None,
    ) -> Dict[str, Any]:
        """批量隔离"""
        results = {"success": 0, "failed": 0, "details": []}
        
        for lu_id in lu_ids:
            try:
                result = self.quarantine_knowledge(lu_id, reason, actor)
                results["success"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "success"})
            except Exception as e:
                results["failed"] += 1
                results["details"].append({"lu_id": str(lu_id), "status": "failed", "error": str(e)})
        
        return results
    
    # ==================== 查询接口 ====================
    
    def get_transfer_history(
        self,
        lu_id: Optional[UUID] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取转移历史"""
        history = self._transfer_records
        
        if lu_id:
            history = [r for r in history if r.get("lu_id") == str(lu_id)]
        
        if status:
            history = [r for r in history if r.get("status") == status]
        
        return sorted(history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_internalized_lus(
        self,
        lifecycle_state: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[LearningUnit], int]:
        """获取已内化的 Learning Units"""
        return self.lu_service.list_learning_units(
            is_internalized=True,
            skip=skip,
            limit=limit,
        )
    
    def get_aga_status(self, lu_id: UUID) -> Optional[Dict[str, Any]]:
        """获取 LU 在 AGA Portal 中的状态"""
        lu = self.lu_service.get_learning_unit(lu_id)
        if not lu or not lu.is_internalized:
            return None
        
        status = {
            "lu_id": str(lu_id),
            "is_internalized": True,
            "internalized_at": lu.internalized_at.isoformat() if lu.internalized_at else None,
            "lifecycle_state": lu.lifecycle_state,
            "aga_mapping": lu.aga_slot_mapping,
        }
        
        # 从 Portal 获取详细状态
        if self._portal_client and self.portal_available:
            try:
                aga_mapping = lu.aga_slot_mapping or {}
                injected_ids = aga_mapping.get("injected_ids", [])
                
                if injected_ids:
                    # 获取第一个知识的状态作为代表
                    portal_status = self._portal_client.get_knowledge(injected_ids[0])
                    if portal_status:
                        status["portal_status"] = portal_status
            except Exception as e:
                logger.error(f"Failed to get Portal status: {e}")
        
        return status
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 从数据库获取统计
        total_internalized = self.db.query(LearningUnit).filter(
            LearningUnit.is_internalized == True
        ).count()
        
        by_lifecycle = {}
        for state in [AGALifecycleState.PROBATIONARY, AGALifecycleState.CONFIRMED, 
                      AGALifecycleState.DEPRECATED, AGALifecycleState.QUARANTINED]:
            count = self.db.query(LearningUnit).filter(
                LearningUnit.is_internalized == True,
                LearningUnit.lifecycle_state == state,
            ).count()
            by_lifecycle[state] = count
        
        # 转移记录统计
        transfer_stats = {
            "total": len(self._transfer_records),
            "success": sum(1 for r in self._transfer_records if r["status"] == "success"),
            "failed": sum(1 for r in self._transfer_records if r["status"] == "failed"),
            "simulated": sum(1 for r in self._transfer_records if r["status"] == "simulated"),
        }
        
        # Portal 统计
        portal_stats = None
        if self._portal_client and self.portal_available:
            try:
                portal_stats = self._portal_client.get_statistics()
            except Exception as e:
                logger.error(f"Failed to get Portal statistics: {e}")
        
        return {
            "total_internalized": total_internalized,
            "by_lifecycle_state": by_lifecycle,
            "transfer_records": transfer_stats,
            "portal_available": self.portal_available,
            "portal_url": settings.AGA_PORTAL_URL if settings.AGA_PORTAL_ENABLED else None,
            "portal_stats": portal_stats,
        }
