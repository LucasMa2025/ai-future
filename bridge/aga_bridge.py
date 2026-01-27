"""
AGA Bridge - 基于辅助治理注意力的知识桥接系统（远程 API 版本）

相比原始 Bridge 的优势：
1. 零训练注入：无需 EWC、无需梯度计算
2. 即时生效：知识注入后立即可用
3. 即时隔离：问题知识可立即移除
4. 完全可追溯：每个贡献可追溯到具体 LU
5. 无灾难性遗忘：不修改原始模型参数

架构说明：
┌─────────────────────────────────────────────────────────────────┐
│  Backend Server (CPU)                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AGABridge                                                │   │
│  │  - write_learning_unit()                                 │   │
│  │  - confirm/deprecate/quarantine_learning_unit()          │   │
│  │  - 通过 HTTP 调用远程 AGA API                           │   │
│  └─────────────────────────────────┬───────────────────────┘   │
└────────────────────────────────────┼────────────────────────────┘
                                     │ HTTP REST API
                                     v
┌─────────────────────────────────────────────────────────────────┐
│  AGA Server (GPU)                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FastAPI (aga/api.py)                                     │   │
│  │  - POST /inject                                          │   │
│  │  - POST /inject/batch                                    │   │
│  │  - POST /lifecycle/update                                │   │
│  │  - POST /quarantine/slot/{slot_idx}                      │   │
│  │  - POST /quarantine/lu                                   │   │
│  │  - GET /slot/free                                        │   │
│  │  - GET /slot/{slot_idx}                                  │   │
│  │  - GET /statistics                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          v                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ AGA Module (aga/core.py v2.1)                            │   │
│  │  - AuxiliaryGovernedAttention                            │   │
│  │  - 挂载到 Transformer 模型                               │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

使用方式：
    # 1. 启动 AGA 服务（GPU 服务器）
    python -m aga.api --host 0.0.0.0 --port 8081
    
    # 2. 后端使用 AGABridge（通过 HTTP 调用）
    bridge = AGABridge(aga_api_url="http://gpu-server:8081")
    result = bridge.write_learning_unit(lu, writer_id, approval)
"""
from typing import Dict, List, Any, Optional, Tuple, Union, Protocol
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# HTTP 客户端
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    HAS_HTTPX = False

# 尝试导入 core.types，如果失败则使用本地定义
try:
    from core.types import LearningUnit as CoreLearningUnit, AuditApproval as CoreAuditApproval
    from core.enums import BridgeWriteResult, AuditDecision
    HAS_CORE_TYPES = True
except ImportError:
    HAS_CORE_TYPES = False
    
    # 定义本地枚举
    class BridgeWriteResult(str, Enum):
        SUCCESS = "success"
        PERMISSION_DENIED = "permission_denied"
        INTERNALIZATION_FAILED = "internalization_failed"
        ALREADY_EXISTS = "already_exists"
        API_ERROR = "api_error"
    
    class AuditDecision(str, Enum):
        APPROVE = "approve"
        REJECT = "reject"
        CORRECT = "correct"

# 生命周期状态（本地定义，与 aga/core.py 保持一致）
class LifecycleState(str, Enum):
    EMPTY = "empty"
    PROBATIONARY = "probationary"
    CONFIRMED = "confirmed"
    DEPRECATED = "deprecated"
    QUARANTINED = "quarantined"

from .permission import PermissionValidator


logger = logging.getLogger(__name__)


# ============================================================
# 通用接口协议（支持多种输入格式）
# ============================================================

class ConstraintProtocol(Protocol):
    """约束协议"""
    condition: str
    decision: str
    confidence: float


class LearningUnitProtocol(Protocol):
    """Learning Unit 协议"""
    id: str
    proposed_constraints: List[ConstraintProtocol]


class AuditApprovalProtocol(Protocol):
    """审计批准协议"""
    approval_id: str
    decision: str
    
    def verify(self) -> bool:
        ...


# ============================================================
# 数据类
# ============================================================

@dataclass
class AGAWriteRecord:
    """AGA 写入记录"""
    lu_id: str
    writer_id: str
    approval_id: str
    slot_indices: List[int]  # 注入的槽位列表
    lifecycle_state: LifecycleState
    timestamp: datetime
    result: BridgeWriteResult
    error: str = ""


@dataclass
class AGABridgeConfig:
    """AGA Bridge 配置"""
    # AGA API 配置
    aga_api_url: str = "http://localhost:8081"
    api_timeout: float = 30.0
    
    # 治理配置
    initial_lifecycle: LifecycleState = LifecycleState.PROBATIONARY
    auto_confirm_after_hits: int = 100  # 命中N次后自动确认
    
    # 编码配置（用于本地编码器）
    hidden_dim: int = 4096
    bottleneck_dim: int = 64
    use_mean_pooling: bool = True


class KnowledgeEncoder:
    """
    知识编码器
    
    将 Learning Unit 的约束编码为 key-value 向量。
    
    注意：此编码器在 Backend 服务器上运行，需要访问 tokenizer。
    如果 Backend 没有 GPU，可以使用 CPU 编码。
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer = None,
        config: Optional[AGABridgeConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or AGABridgeConfig()
        self.device = torch.device("cpu")
        
        if model is not None:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                pass
    
    def encode_constraint(
        self,
        condition: str,
        decision: str,
    ) -> Tuple[List[float], List[float]]:
        """
        编码约束为 key-value 向量
        
        Args:
            condition: 条件文本（何时激活）
            decision: 决策文本（添加什么信号）
        
        Returns:
            key_vector: List[float], 长度为 bottleneck_dim
            value_vector: List[float], 长度为 hidden_dim
        """
        if self.model is None or self.tokenizer is None:
            # 无模型时使用确定性哈希编码
            return self._hash_encode(condition, decision)
        
        with torch.no_grad():
            # 编码条件
            condition_tokens = self.tokenizer(
                condition, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            
            # 编码决策
            decision_tokens = self.tokenizer(
                decision,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(self.device)
            
            # 获取嵌入
            embed_layer = self.model.model.embed_tokens
            condition_emb = embed_layer(condition_tokens.input_ids)
            decision_emb = embed_layer(decision_tokens.input_ids)
            
            # 池化
            if self.config.use_mean_pooling:
                condition_mask = condition_tokens.attention_mask.unsqueeze(-1)
                decision_mask = decision_tokens.attention_mask.unsqueeze(-1)
                
                condition_pooled = (condition_emb * condition_mask).sum(dim=1) / condition_mask.sum(dim=1)
                decision_pooled = (decision_emb * decision_mask).sum(dim=1) / decision_mask.sum(dim=1)
            else:
                condition_pooled = condition_emb[:, -1, :]
                decision_pooled = decision_emb[:, -1, :]
            
            # 投影到目标维度
            key_vector = condition_pooled[0, :self.config.bottleneck_dim]
            value_vector = decision_pooled[0]
            
            # 如果维度不够，用零填充
            if key_vector.shape[0] < self.config.bottleneck_dim:
                key_vector = F.pad(key_vector, (0, self.config.bottleneck_dim - key_vector.shape[0]))
            
            return key_vector.tolist(), value_vector.tolist()
    
    def _hash_encode(
        self,
        condition: str,
        decision: str,
    ) -> Tuple[List[float], List[float]]:
        """
        使用确定性哈希进行编码（无模型时的回退方案）
        
        生产环境应使用真实模型编码。
        """
        # 使用文本哈希作为随机种子
        seed = hash(f"{condition}:{decision}") % (2**32)
        torch.manual_seed(seed)
        
        key_vec = torch.randn(self.config.bottleneck_dim) * 0.1
        val_vec = torch.randn(self.config.hidden_dim) * 0.1
        
        return key_vec.tolist(), val_vec.tolist()
    
    def encode_learning_unit(
        self, 
        lu: Union[LearningUnitProtocol, Any]
    ) -> List[Tuple[List[float], List[float], str, str]]:
        """
        编码整个 Learning Unit
        
        Returns:
            List of (key_vector, value_vector, condition, decision)
        """
        encoded = []
        
        # 获取约束列表（兼容不同格式）
        constraints = getattr(lu, 'proposed_constraints', [])
        
        for constraint in constraints:
            condition = getattr(constraint, 'condition', '')
            decision = getattr(constraint, 'decision', '')
            
            if not condition or not decision:
                logger.warning(f"Skipping constraint with empty condition/decision")
                continue
            
            key, value = self.encode_constraint(condition, decision)
            encoded.append((key, value, condition, decision))
        
        return encoded


# ============================================================
# AGA HTTP 客户端
# ============================================================

class AGAAPIClient:
    """
    AGA HTTP API 客户端
    
    封装与远程 AGA 服务的所有通信。
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        if HAS_HTTPX:
            self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        else:
            self._client = None
    
    def _request(
        self, 
        method: str, 
        path: str, 
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """发送 HTTP 请求"""
        if self._client:
            # 使用 httpx
            if method == "GET":
                response = self._client.get(path)
            else:
                response = self._client.post(path, json=json_data)
            response.raise_for_status()
            return response.json()
        else:
            # 使用 urllib
            url = f"{self.base_url}{path}"
            
            if json_data:
                data = json.dumps(json_data).encode("utf-8")
                req = urllib.request.Request(
                    url, data=data,
                    headers={"Content-Type": "application/json"},
                    method=method,
                )
            else:
                req = urllib.request.Request(url, method=method)
            
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else str(e)
                raise RuntimeError(f"HTTP {e.code}: {error_body}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return self._request("GET", "/health")
    
    def inject_knowledge(
        self,
        slot_idx: int,
        key_vector: List[float],
        value_vector: List[float],
        lu_id: str,
        lifecycle_state: str = "probationary",
        condition: Optional[str] = None,
        decision: Optional[str] = None,
    ) -> Dict[str, Any]:
        """注入知识到指定槽位"""
        return self._request("POST", "/inject", {
            "slot_idx": slot_idx,
            "key_vector": key_vector,
            "value_vector": value_vector,
            "lu_id": lu_id,
            "lifecycle_state": lifecycle_state,
            "condition": condition,
            "decision": decision,
        })
    
    def batch_inject(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量注入知识"""
        return self._request("POST", "/inject/batch", {"items": items})
    
    def update_lifecycle(self, slot_idx: int, new_state: str) -> Dict[str, Any]:
        """更新槽位生命周期状态"""
        return self._request("POST", "/lifecycle/update", {
            "slot_idx": slot_idx,
            "new_state": new_state,
        })
    
    def quarantine_slot(self, slot_idx: int) -> Dict[str, Any]:
        """隔离指定槽位"""
        return self._request("POST", f"/quarantine/slot/{slot_idx}")
    
    def quarantine_by_lu_id(self, lu_id: str) -> Dict[str, Any]:
        """按 LU ID 隔离所有相关槽位"""
        return self._request("POST", "/quarantine/lu", {"lu_id": lu_id})
    
    def find_free_slot(self) -> Optional[int]:
        """查找空闲槽位"""
        result = self._request("GET", "/slot/free")
        return result.get("free_slot")
    
    def get_slot_info(self, slot_idx: int) -> Dict[str, Any]:
        """获取槽位详细信息"""
        return self._request("GET", f"/slot/{slot_idx}")
    
    def get_slots_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 查找所有槽位"""
        result = self._request("GET", f"/slots/by-lu/{lu_id}")
        return result.get("slots", [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取 AGA 统计信息"""
        return self._request("GET", "/statistics")
    
    def close(self):
        """关闭客户端连接"""
        if self._client and hasattr(self._client, 'close'):
            self._client.close()


# ============================================================
# AGA Bridge（远程 API 版本）
# ============================================================

class AGABridge:
    """
    AGA Bridge - 热插拔式知识桥接系统（远程 API 版本）
    
    通过 HTTP API 与远程 AGA 服务通信，实现知识注入。
    
    使用流程：
    1. 初始化时指定 AGA API 地址
    2. 接收审计通过的 Learning Unit
    3. 编码约束为 key-value 向量
    4. 通过 HTTP API 注入到远程 AGA 服务
    5. 根据使用情况调整生命周期
    
    使用示例：
        bridge = AGABridge(
            aga_api_url="http://gpu-server:8081",
            encoder=KnowledgeEncoder(model, tokenizer),
        )
        
        result = bridge.write_learning_unit(
            learning_unit=lu,
            writer_id="audit_system",
            audit_approval=approval,
        )
    """
    
    def __init__(
        self,
        aga_api_url: str = "http://localhost:8081",
        encoder: Optional[KnowledgeEncoder] = None,
        config: Optional[AGABridgeConfig] = None,
        authorized_writers: Optional[set] = None,
    ):
        """
        初始化 AGA Bridge
        
        Args:
            aga_api_url: AGA API 服务地址
            encoder: 知识编码器（可选，无则使用哈希编码）
            config: Bridge 配置
            authorized_writers: 授权写入者集合
        """
        self.config = config or AGABridgeConfig(aga_api_url=aga_api_url)
        
        # AGA API 客户端
        self.api_client = AGAAPIClient(
            base_url=self.config.aga_api_url,
            timeout=self.config.api_timeout,
        )
        
        # 知识编码器
        self.encoder = encoder or KnowledgeEncoder(config=self.config)
        
        # 权限验证器
        self.permission_validator = PermissionValidator(
            authorized_writers=authorized_writers or {"audit_system"}
        )
        
        # 写入记录
        self.write_records: List[AGAWriteRecord] = []
        
        # LU ID -> 槽位映射
        self.lu_slot_mapping: Dict[str, List[int]] = {}
        
        # 验证连接
        self._verify_connection()
    
    def _verify_connection(self) -> bool:
        """验证与 AGA API 的连接"""
        try:
            health = self.api_client.health_check()
            if health.get("status") == "healthy":
                logger.info(f"Connected to AGA API at {self.config.aga_api_url}")
                return True
            else:
                logger.warning(f"AGA API unhealthy: {health}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to AGA API: {e}")
            return False
    
    def write_learning_unit(
        self,
        learning_unit: Union[LearningUnitProtocol, Any],
        writer_id: str,
        audit_approval: Union[AuditApprovalProtocol, Any],
    ) -> BridgeWriteResult:
        """
        写入 Learning Unit
        
        通过 HTTP API 将编码后的知识注入到远程 AGA 服务。
        
        Args:
            learning_unit: 学习单元
            writer_id: 写入者 ID
            audit_approval: 审计批准
        
        Returns:
            写入结果
        """
        # 获取 LU ID
        lu_id = str(getattr(learning_unit, 'id', learning_unit))
        approval_id = str(getattr(audit_approval, 'approval_id', 'unknown'))
        
        logger.info(f"[AGA Bridge] Receiving write request for LU: {lu_id}")
        
        # 1. 权限验证
        try:
            self.permission_validator.validate_all(writer_id, audit_approval)
        except Exception as e:
            logger.warning(f"Permission denied: {e}")
            self._record_write(lu_id, writer_id, approval_id, 
                             [], BridgeWriteResult.PERMISSION_DENIED, str(e))
            return BridgeWriteResult.PERMISSION_DENIED
        
        logger.info("  ✓ Permission validated")
        
        # 2. 编码约束
        try:
            encoded_constraints = self.encoder.encode_learning_unit(learning_unit)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            self._record_write(lu_id, writer_id, approval_id,
                             [], BridgeWriteResult.INTERNALIZATION_FAILED, str(e))
            return BridgeWriteResult.INTERNALIZATION_FAILED
        
        logger.info(f"  ✓ Encoded {len(encoded_constraints)} constraints")
        
        # 3. 准备批量注入请求
        inject_items = []
        for i, (key_vec, val_vec, condition, decision) in enumerate(encoded_constraints):
            # 查找空闲槽位
            try:
                slot_idx = self.api_client.find_free_slot()
                if slot_idx is None:
                    logger.warning(f"No free slot available for constraint {i}")
                    continue
            except Exception as e:
                logger.error(f"Failed to find free slot: {e}")
                continue
            
            inject_items.append({
                "slot_idx": slot_idx,
                "key_vector": key_vec,
                "value_vector": val_vec,
                "lu_id": f"{lu_id}_c{i}",
                "lifecycle_state": self.config.initial_lifecycle.value,
                "condition": condition,
                "decision": decision,
            })
        
        if not inject_items:
            logger.error("No constraints to inject")
            self._record_write(lu_id, writer_id, approval_id,
                             [], BridgeWriteResult.INTERNALIZATION_FAILED, "No free slots")
            return BridgeWriteResult.INTERNALIZATION_FAILED
        
        # 4. 批量注入
        try:
            result = self.api_client.batch_inject(inject_items)
            
            if result.get("success", 0) == 0:
                raise RuntimeError(f"All injections failed: {result}")
            
            # 记录成功注入的槽位
            slot_indices = [
                item.get("slot_idx") 
                for item in result.get("results", [])
                if item.get("success")
            ]
            
        except Exception as e:
            logger.error(f"Injection failed: {e}")
            self._record_write(lu_id, writer_id, approval_id,
                             [], BridgeWriteResult.API_ERROR, str(e))
            return BridgeWriteResult.API_ERROR
        
        # 5. 记录映射
        self.lu_slot_mapping[lu_id] = slot_indices
        
        # 6. 记录写入
        self._record_write(
            lu_id, writer_id, approval_id,
            slot_indices, BridgeWriteResult.SUCCESS
        )
        
        logger.info(f"  ✓ Injected to {len(slot_indices)} slots via API")
        logger.info(f"  ✓ Write successful (lifecycle: {self.config.initial_lifecycle.value})")
        
        return BridgeWriteResult.SUCCESS
    
    def _record_write(
        self,
        lu_id: str,
        writer_id: str,
        approval_id: str,
        slot_indices: List[int],
        result: BridgeWriteResult,
        error: str = "",
    ):
        """记录写入"""
        record = AGAWriteRecord(
            lu_id=lu_id,
            writer_id=writer_id,
            approval_id=approval_id,
            slot_indices=slot_indices,
            lifecycle_state=self.config.initial_lifecycle,
            timestamp=datetime.now(),
            result=result,
            error=error,
        )
        self.write_records.append(record)
    
    # ==================== 治理接口 ====================
    
    def confirm_learning_unit(self, lu_id: str) -> bool:
        """确认 Learning Unit（试用期 → 已确认）"""
        if lu_id not in self.lu_slot_mapping:
            logger.warning(f"LU {lu_id} not found")
            return False
        
        try:
            # 获取所有相关槽位并更新状态
            slots = self.api_client.get_slots_by_lu_id(f"{lu_id}_c0")
            for slot_idx in slots:
                self.api_client.update_lifecycle(slot_idx, LifecycleState.CONFIRMED.value)
            
            logger.info(f"LU {lu_id} confirmed ({len(slots)} slots)")
            return True
        except Exception as e:
            logger.error(f"Failed to confirm LU {lu_id}: {e}")
            return False
    
    def deprecate_learning_unit(self, lu_id: str) -> bool:
        """弃用 Learning Unit"""
        if lu_id not in self.lu_slot_mapping:
            logger.warning(f"LU {lu_id} not found")
            return False
        
        try:
            # 遍历所有约束
            deprecated_count = 0
            for i in range(100):  # 最多检查100个约束
                slots = self.api_client.get_slots_by_lu_id(f"{lu_id}_c{i}")
                if not slots:
                    break
                for slot_idx in slots:
                    self.api_client.update_lifecycle(slot_idx, LifecycleState.DEPRECATED.value)
                    deprecated_count += 1
            
            logger.info(f"LU {lu_id} deprecated ({deprecated_count} slots)")
            return True
        except Exception as e:
            logger.error(f"Failed to deprecate LU {lu_id}: {e}")
            return False
    
    def quarantine_learning_unit(self, lu_id: str) -> bool:
        """隔离 Learning Unit（立即移除影响）"""
        if lu_id not in self.lu_slot_mapping:
            logger.warning(f"LU {lu_id} not found")
            return False
        
        try:
            # 遍历所有约束并隔离
            quarantined_count = 0
            for i in range(100):
                result = self.api_client.quarantine_by_lu_id(f"{lu_id}_c{i}")
                quarantined_count += result.get("count", 0)
                if result.get("count", 0) == 0 and i > 0:
                    break
            
            # 从映射中移除
            del self.lu_slot_mapping[lu_id]
            
            logger.info(f"LU {lu_id} quarantined ({quarantined_count} slots)")
            return True
        except Exception as e:
            logger.error(f"Failed to quarantine LU {lu_id}: {e}")
            return False
    
    def rollback_learning_unit(self, lu_id: str) -> bool:
        """回滚 Learning Unit（等同于隔离）"""
        return self.quarantine_learning_unit(lu_id)
    
    # ==================== 查询接口 ====================
    
    def get_written_units(self) -> List[str]:
        """获取已写入的 LU ID 列表"""
        return list(self.lu_slot_mapping.keys())
    
    def get_lu_status(self, lu_id: str) -> Optional[Dict[str, Any]]:
        """获取 LU 状态"""
        if lu_id not in self.lu_slot_mapping:
            return None
        
        try:
            # 获取第一个约束的槽位信息
            slots = self.api_client.get_slots_by_lu_id(f"{lu_id}_c0")
            if not slots:
                return None
            
            info = self.api_client.get_slot_info(slots[0])
            
            return {
                'lu_id': lu_id,
                'slot_idx': info.get('slot_idx'),
                'lifecycle': info.get('lifecycle_state'),
                'reliability': info.get('reliability'),
                'hit_count': info.get('hit_count'),
                'condition': info.get('condition'),
                'decision': info.get('decision'),
            }
        except Exception as e:
            logger.error(f"Failed to get LU status: {e}")
            return None
    
    def get_write_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取写入历史"""
        return [
            {
                'lu_id': r.lu_id,
                'writer_id': r.writer_id,
                'approval_id': r.approval_id,
                'result': r.result.value,
                'lifecycle': r.lifecycle_state.value,
                'timestamp': r.timestamp.isoformat(),
                'slot_count': len(r.slot_indices),
                'error': r.error,
            }
            for r in self.write_records[-limit:]
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_count = sum(1 for r in self.write_records if r.result == BridgeWriteResult.SUCCESS)
        
        # 获取远程 AGA 统计
        try:
            aga_stats = self.api_client.get_statistics()
        except Exception as e:
            logger.warning(f"Failed to get AGA stats: {e}")
            aga_stats = {}
        
        return {
            'api_url': self.config.aga_api_url,
            'total_writes': len(self.write_records),
            'successful_writes': success_count,
            'active_lus': len(self.lu_slot_mapping),
            'aga_stats': aga_stats,
            'permission_stats': self.permission_validator.get_statistics(),
        }
    
    def close(self):
        """关闭连接"""
        self.api_client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
