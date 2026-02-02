"""
知识转移服务 v3.2

实现:
1. 审批通过的 Learning Unit 通过 HTTP API 转移到 AGA Portal
2. AGA 生命周期管理 (通过 Portal API)
3. 知识隔离和回滚

架构说明:
    ┌─────────────────────────────────────────────────────────┐
    │  AIFuture Backend (治理系统)                            │
    │  ┌───────────────────────────────────────────────────┐ │
    │  │ ApprovalService                                   │ │
    │  │    ↓ submit_lu_approval() 审批通过                │ │
    │  │ KnowledgeTransferService                          │ │
    │  │    ↓ transfer_to_aga() 转移知识                   │ │
    │  │ AGAPortalClient (本模块定义)                      │ │
    │  │    ↓ inject_knowledge() HTTP 调用                 │ │
    │  └─────────────────────────┬─────────────────────────┘ │
    └────────────────────────────┼────────────────────────────┘
                                 │ HTTP REST API
                                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │  AGA Portal (独立部署，可能在不同服务器)               │
    │  ┌───────────────────────────────────────────────────┐ │
    │  │ Portal API (aga.portal)                           │ │
    │  │  - POST /knowledge/inject                         │ │
    │  │  - PUT /lifecycle/update                          │ │
    │  │  - POST /lifecycle/quarantine                     │ │
    │  │  - GET /statistics, /audit                        │ │
    │  └─────────────────────────┬─────────────────────────┘ │
    │                            │ Redis Pub/Sub             │
    │                            ▼                           │
    │  ┌───────────────────────────────────────────────────┐ │
    │  │ AGA Runtime (GPU 服务器)                          │ │
    │  │  - 订阅同步消息                                   │ │
    │  │  - 执行推理                                       │ │
    │  └───────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘

⚠️ 重要说明:
- AIFuture (治理系统) 与 AGA Portal 是独立部署的服务
- 本模块仅通过 HTTP API 与 AGA Portal 通信
- 不依赖 aga 包的任何内部实现
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
        注入知识到 AGA Portal
        
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
    
    def batch_inject(
        self,
        items: List[Dict[str, Any]],
        namespace: str = None,
        skip_duplicates: bool = True,
    ) -> Dict[str, Any]:
        """批量注入知识"""
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
# 简易向量编码器
# ============================================================

class SimpleKnowledgeEncoder:
    """
    简易知识编码器
    
    将约束文本编码为确定性哈希向量。
    
    注意：生产环境应使用真实的语义编码（如 Sentence Transformers），
    此实现仅用于演示和测试。
    """
    
    def __init__(
        self,
        hidden_dim: int = None,
        bottleneck_dim: int = None,
    ):
        self.hidden_dim = hidden_dim or settings.AGA_ENCODING_HIDDEN_DIM
        self.bottleneck_dim = bottleneck_dim or settings.AGA_ENCODING_BOTTLENECK_DIM
    
    def encode_constraint(
        self,
        condition: str,
        decision: str,
    ) -> Tuple[List[float], List[float]]:
        """
        编码约束为 key-value 向量
        
        使用确定性哈希生成伪随机向量，确保相同输入产生相同输出。
        
        Args:
            condition: 条件文本
            decision: 决策文本
        
        Returns:
            (key_vector, value_vector)
        """
        # 使用 SHA256 哈希作为伪随机种子
        key_seed = hashlib.sha256(f"key:{condition}".encode()).digest()
        value_seed = hashlib.sha256(f"value:{decision}".encode()).digest()
        
        # 生成向量
        key_vector = self._hash_to_vector(key_seed, self.bottleneck_dim)
        value_vector = self._hash_to_vector(value_seed, self.hidden_dim)
        
        return key_vector, value_vector
    
    def _hash_to_vector(self, seed: bytes, dim: int) -> List[float]:
        """将哈希种子扩展为指定维度的向量"""
        import struct
        
        # 使用 SHAKE256 扩展到所需字节数
        extended = hashlib.shake_256(seed).digest(dim * 4)
        
        # 转换为 float 列表（范围 [-0.1, 0.1]）
        vector = []
        for i in range(dim):
            # 取 4 字节转为无符号整数
            val = struct.unpack('<I', extended[i*4:(i+1)*4])[0]
            # 归一化到 [-0.1, 0.1]
            normalized = (val / 0xFFFFFFFF - 0.5) * 0.2
            vector.append(normalized)
        
        return vector


# ============================================================
# 知识转移服务
# ============================================================

class KnowledgeTransferService:
    """
    知识转移服务
    
    将审批通过的 Learning Unit 通过 HTTP API 转移到 AGA Portal。
    """
    
    def __init__(
        self,
        db: Session,
        lu_service: Optional["LearningUnitService"] = None,
        portal_client: Optional[AGAPortalClient] = None,
    ):
        self.db = db
        self._lu_service = lu_service
        
        # 初始化 Portal 客户端
        if portal_client:
            self._portal_client = portal_client
        elif settings.AGA_PORTAL_ENABLED:
            self._portal_client = AGAPortalClient()
        else:
            self._portal_client = None
        
        # 知识编码器
        self._encoder = SimpleKnowledgeEncoder()
        
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
            # 编码约束
            condition = constraint.condition or ""
            decision = constraint.decision or ""
            
            if not condition or not decision:
                logger.warning(f"Skipping constraint {i} with empty condition/decision")
                continue
            
            key_vector, value_vector = self._encoder.encode_constraint(condition, decision)
            
            # 构建知识 ID
            knowledge_id = f"{lu.id}_c{i}"
            
            try:
                # 调用 Portal API 注入
                result = self._portal_client.inject_knowledge(
                    lu_id=knowledge_id,
                    condition=condition,
                    decision=decision,
                    key_vector=key_vector,
                    value_vector=value_vector,
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
                    logger.debug(f"Injected knowledge {knowledge_id} to Portal")
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
