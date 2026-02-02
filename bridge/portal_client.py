"""
AGA Portal 客户端

为外部系统提供与 AGA Portal 的 HTTP 集成。
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AGAClient:
    """
    同步 AGA 客户端
    
    用于外部治理系统与 AGA Portal 的集成。
    
    示例：
    
    ```python
    from aga.client import AGAClient
    
    # 创建客户端
    client = AGAClient("http://portal:8081")
    
    # 注入知识
    result = client.inject_knowledge(
        lu_id="knowledge_001",
        condition="当用户询问...",
        decision="应该回答...",
        key_vector=[0.1, 0.2, ...],
        value_vector=[0.3, 0.4, ...],
    )
    
    # 查询知识
    knowledge = client.query_knowledge(namespace="default", limit=10)
    
    # 更新生命周期
    client.update_lifecycle("knowledge_001", "confirmed", reason="验证通过")
    
    # 隔离知识
    client.quarantine("knowledge_001", reason="检测到异常")
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        初始化客户端
        
        Args:
            base_url: Portal API 地址
            timeout: 请求超时（秒）
            api_key: API 密钥（可选）
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("需要安装 httpx: pip install httpx")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )
    
    def close(self):
        """关闭客户端"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # ==================== 知识管理 ====================
    
    def inject_knowledge(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        注入知识
        
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
                "namespace": namespace,
                "lifecycle_state": lifecycle_state,
                "trust_tier": trust_tier,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """批量注入知识"""
        response = self._client.post(
            "/knowledge/batch",
            json={
                "items": items,
                "namespace": namespace,
                "skip_duplicates": skip_duplicates,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def get_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """获取单个知识"""
        try:
            response = self._client.get(
                f"/knowledge/{namespace}/{lu_id}",
                params={"include_vectors": include_vectors},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Get knowledge failed: {e}")
            return None
    
    def query_knowledge(
        self,
        namespace: str = "default",
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
    ) -> Dict[str, Any]:
        """查询知识列表"""
        params = {
            "limit": limit,
            "offset": offset,
            "include_vectors": include_vectors,
        }
        if lifecycle_states:
            params["lifecycle_states"] = ",".join(lifecycle_states)
        if trust_tiers:
            params["trust_tiers"] = ",".join(trust_tiers)
        
        response = self._client.get(f"/knowledge/{namespace}", params=params)
        response.raise_for_status()
        return response.json()
    
    def delete_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """删除知识"""
        response = self._client.delete(
            f"/knowledge/{namespace}/{lu_id}",
            params={"reason": reason} if reason else {},
        )
        response.raise_for_status()
        return response.json()
    
    # ==================== 生命周期管理 ====================
    
    def update_lifecycle(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """更新生命周期状态"""
        response = self._client.put(
            "/lifecycle/update",
            json={
                "lu_id": lu_id,
                "new_state": new_state,
                "namespace": namespace,
                "reason": reason,
            },
        )
        response.raise_for_status()
        return response.json()
    
    def confirm(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """确认知识"""
        return self.update_lifecycle(lu_id, "confirmed", namespace, reason)
    
    def deprecate(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """弃用知识"""
        return self.update_lifecycle(lu_id, "deprecated", namespace, reason)
    
    def quarantine(
        self,
        lu_id: str,
        reason: str,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """隔离知识"""
        response = self._client.post(
            "/lifecycle/quarantine",
            json={
                "lu_id": lu_id,
                "reason": reason,
                "namespace": namespace,
            },
        )
        response.raise_for_status()
        return response.json()
    
    # ==================== 统计和审计 ====================
    
    def get_statistics(
        self,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取统计信息"""
        if namespace:
            response = self._client.get(f"/statistics/{namespace}")
        else:
            response = self._client.get("/statistics")
        response.raise_for_status()
        return response.json()
    
    def get_audit_log(
        self,
        namespace: Optional[str] = None,
        lu_id: Optional[str] = None,
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
    
    # ==================== 健康检查 ====================
    
    def health(self) -> Dict[str, Any]:
        """健康检查"""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    def is_healthy(self) -> bool:
        """检查是否健康"""
        try:
            health = self.health()
            return health.get("status") == "healthy"
        except Exception:
            return False
    
    # ==================== 命名空间 ====================
    
    def list_namespaces(self) -> List[str]:
        """列出命名空间"""
        response = self._client.get("/namespaces")
        response.raise_for_status()
        data = response.json()
        return data.get("data", {}).get("namespaces", [])
    
    # ==================== Runtime ====================
    
    def list_runtimes(self) -> Dict[str, Any]:
        """列出所有 Runtime"""
        response = self._client.get("/runtimes")
        response.raise_for_status()
        return response.json()


class AsyncAGAClient:
    """
    异步 AGA 客户端
    
    异步版本，适用于高并发场景。
    
    示例：
    
    ```python
    from aga.client import AsyncAGAClient
    
    async with AsyncAGAClient("http://portal:8081") as client:
        # 并发注入
        tasks = [
            client.inject_knowledge(
                lu_id=f"knowledge_{i}",
                condition=f"条件 {i}",
                decision=f"决策 {i}",
                key_vector=[...],
                value_vector=[...],
            )
            for i in range(10)
        ]
        results = await asyncio.gather(*tasks)
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        try:
            import httpx
        except ImportError:
            raise ImportError("需要安装 httpx: pip install httpx")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )
    
    async def close(self):
        """关闭客户端"""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
    
    # ==================== 知识管理 ====================
    
    async def inject_knowledge(
        self,
        lu_id: str,
        condition: str,
        decision: str,
        key_vector: List[float],
        value_vector: List[float],
        namespace: str = "default",
        lifecycle_state: str = "probationary",
        trust_tier: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """注入知识"""
        response = await self._client.post(
            "/knowledge/inject",
            json={
                "lu_id": lu_id,
                "condition": condition,
                "decision": decision,
                "key_vector": key_vector,
                "value_vector": value_vector,
                "namespace": namespace,
                "lifecycle_state": lifecycle_state,
                "trust_tier": trust_tier,
                "metadata": metadata,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = "default",
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """批量注入知识"""
        response = await self._client.post(
            "/knowledge/batch",
            json={
                "items": items,
                "namespace": namespace,
                "skip_duplicates": skip_duplicates,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def get_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
        include_vectors: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """获取单个知识"""
        try:
            response = await self._client.get(
                f"/knowledge/{namespace}/{lu_id}",
                params={"include_vectors": include_vectors},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Get knowledge failed: {e}")
            return None
    
    async def query_knowledge(
        self,
        namespace: str = "default",
        lifecycle_states: Optional[List[str]] = None,
        trust_tiers: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
    ) -> Dict[str, Any]:
        """查询知识列表"""
        params = {
            "limit": limit,
            "offset": offset,
            "include_vectors": include_vectors,
        }
        if lifecycle_states:
            params["lifecycle_states"] = ",".join(lifecycle_states)
        if trust_tiers:
            params["trust_tiers"] = ",".join(trust_tiers)
        
        response = await self._client.get(f"/knowledge/{namespace}", params=params)
        response.raise_for_status()
        return response.json()
    
    async def delete_knowledge(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """删除知识"""
        response = await self._client.delete(
            f"/knowledge/{namespace}/{lu_id}",
            params={"reason": reason} if reason else {},
        )
        response.raise_for_status()
        return response.json()
    
    # ==================== 生命周期管理 ====================
    
    async def update_lifecycle(
        self,
        lu_id: str,
        new_state: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """更新生命周期状态"""
        response = await self._client.put(
            "/lifecycle/update",
            json={
                "lu_id": lu_id,
                "new_state": new_state,
                "namespace": namespace,
                "reason": reason,
            },
        )
        response.raise_for_status()
        return response.json()
    
    async def confirm(
        self,
        lu_id: str,
        namespace: str = "default",
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """确认知识"""
        return await self.update_lifecycle(lu_id, "confirmed", namespace, reason)
    
    async def quarantine(
        self,
        lu_id: str,
        reason: str,
        namespace: str = "default",
    ) -> Dict[str, Any]:
        """隔离知识"""
        response = await self._client.post(
            "/lifecycle/quarantine",
            json={
                "lu_id": lu_id,
                "reason": reason,
                "namespace": namespace,
            },
        )
        response.raise_for_status()
        return response.json()
    
    # ==================== 统计和健康 ====================
    
    async def get_statistics(
        self,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取统计信息"""
        if namespace:
            response = await self._client.get(f"/statistics/{namespace}")
        else:
            response = await self._client.get("/statistics")
        response.raise_for_status()
        return response.json()
    
    async def health(self) -> Dict[str, Any]:
        """健康检查"""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def is_healthy(self) -> bool:
        """检查是否健康"""
        try:
            health = await self.health()
            return health.get("status") == "healthy"
        except Exception:
            return False
