"""
AGA HTTP API

为 AGA 模块提供 REST API 接口，支持分布式部署场景。

使用方式：
    # 在 GPU 服务器上启动 AGA 服务
    python -m aga.api --host 0.0.0.0 --port 8081
    
    # 后端服务通过 HTTP 调用
    client = AGAClient("http://gpu-server:8081")
    client.inject_knowledge(slot_idx=0, key_vector=..., value_vector=...)
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging

try:
    from fastapi import FastAPI, HTTPException, Body
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

import torch

from .core import (
    AuxiliaryGovernedAttention,
    AGAManager,
    LifecycleState,
    KnowledgeSlotInfo,
    AGAConfig,
)

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic 模型（用于 API 请求/响应）
# ============================================================

if HAS_FASTAPI:
    class InjectKnowledgeRequest(BaseModel):
        """知识注入请求"""
        slot_idx: int = Field(..., description="目标槽位索引")
        key_vector: List[float] = Field(..., description="条件编码向量")
        value_vector: List[float] = Field(..., description="修正信号向量")
        lu_id: str = Field(..., description="Learning Unit ID")
        lifecycle_state: str = Field(default="probationary", description="初始生命周期状态")
        condition: Optional[str] = Field(default=None, description="条件文本（可选）")
        decision: Optional[str] = Field(default=None, description="决策文本（可选）")
    
    class BatchInjectRequest(BaseModel):
        """批量注入请求"""
        items: List[InjectKnowledgeRequest]
    
    class UpdateLifecycleRequest(BaseModel):
        """更新生命周期请求"""
        slot_idx: int
        new_state: str
    
    class QuarantineByLuIdRequest(BaseModel):
        """按 LU ID 隔离请求"""
        lu_id: str
    
    class SlotInfoResponse(BaseModel):
        """槽位信息响应"""
        slot_idx: int
        lu_id: Optional[str]
        lifecycle_state: str
        reliability: float
        key_norm: float
        value_norm: float
        condition: Optional[str]
        decision: Optional[str]
        hit_count: int
    
    class StatisticsResponse(BaseModel):
        """统计信息响应"""
        total_slots: int
        active_slots: int
        state_distribution: Dict[str, int]
        avg_key_norm: float
        avg_value_norm: float
        total_hits: int


# ============================================================
# AGA HTTP 服务
# ============================================================

class AGAService:
    """AGA 服务（单例）"""
    
    _instance: Optional["AGAService"] = None
    
    def __init__(self, config: Optional[AGAConfig] = None):
        self.config = config or AGAConfig()
        self.aga: Optional[AuxiliaryGovernedAttention] = None
        self.manager: Optional[AGAManager] = None
        self.is_attached = False
        self._init_standalone_aga()
    
    def _init_standalone_aga(self):
        """初始化独立的 AGA 模块（不挂载到模型）"""
        self.aga = AuxiliaryGovernedAttention(
            hidden_dim=self.config.hidden_dim,
            bottleneck_dim=self.config.bottleneck_dim,
            num_slots=self.config.num_slots,
            num_heads=self.config.num_heads,
            tau_low=self.config.tau_low,
            tau_high=self.config.tau_high,
            config=self.config,
        )
        self.aga.eval()
        logger.info(f"Initialized standalone AGA with {self.config.num_slots} slots")
    
    @classmethod
    def get_instance(cls, config: Optional[AGAConfig] = None) -> "AGAService":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
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
        """注入知识"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        # 转换为 tensor
        key_tensor = torch.tensor(key_vector, dtype=torch.float32)
        value_tensor = torch.tensor(value_vector, dtype=torch.float32)
        
        # 获取生命周期状态
        state = LifecycleState(lifecycle_state)
        
        # 注入
        success = self.aga.inject_knowledge(
            slot_idx=slot_idx,
            key_vector=key_tensor,
            value_vector=value_tensor,
            lu_id=lu_id,
            lifecycle_state=state,
            condition=condition,
            decision=decision,
        )
        
        return {
            "success": success,
            "slot_idx": slot_idx,
            "lu_id": lu_id,
            "lifecycle_state": lifecycle_state,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def batch_inject(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量注入"""
        results = []
        success_count = 0
        
        for item in items:
            try:
                result = self.inject_knowledge(**item)
                results.append(result)
                if result.get("success"):
                    success_count += 1
            except Exception as e:
                results.append({
                    "success": False,
                    "slot_idx": item.get("slot_idx"),
                    "error": str(e),
                })
        
        return {
            "total": len(items),
            "success": success_count,
            "failed": len(items) - success_count,
            "results": results,
        }
    
    def update_lifecycle(self, slot_idx: int, new_state: str) -> Dict[str, Any]:
        """更新生命周期"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        state = LifecycleState(new_state)
        self.aga.update_lifecycle(slot_idx, state)
        
        return {
            "slot_idx": slot_idx,
            "new_state": new_state,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def quarantine_slot(self, slot_idx: int) -> Dict[str, Any]:
        """隔离槽位"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        self.aga.quarantine_slot(slot_idx)
        
        return {
            "slot_idx": slot_idx,
            "action": "quarantined",
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def quarantine_by_lu_id(self, lu_id: str) -> Dict[str, Any]:
        """按 LU ID 隔离"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        quarantined = self.aga.quarantine_by_lu_id(lu_id)
        
        return {
            "lu_id": lu_id,
            "quarantined_slots": quarantined,
            "count": len(quarantined),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def find_free_slot(self) -> Optional[int]:
        """查找空闲槽位"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        return self.aga.find_free_slot()
    
    def get_slot_info(self, slot_idx: int) -> Dict[str, Any]:
        """获取槽位信息"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        info = self.aga.get_slot_info(slot_idx)
        return {
            "slot_idx": info.slot_idx,
            "lu_id": info.lu_id,
            "lifecycle_state": info.lifecycle_state.value,
            "reliability": info.reliability,
            "key_norm": info.key_norm,
            "value_norm": info.value_norm,
            "condition": info.condition,
            "decision": info.decision,
            "hit_count": info.hit_count,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        return self.aga.get_statistics()
    
    def get_slots_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 查找槽位"""
        if self.aga is None:
            raise RuntimeError("AGA not initialized")
        
        return self.aga.get_slot_by_lu_id(lu_id)


# ============================================================
# FastAPI 应用
# ============================================================

def create_aga_api(config: Optional[AGAConfig] = None) -> "FastAPI":
    """创建 AGA API 应用"""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI is required. Install with: pip install fastapi uvicorn")
    
    app = FastAPI(
        title="AGA (Auxiliary Governed Attention) API",
        description="REST API for AGA knowledge injection and management",
        version="2.1.0",
    )
    
    service = AGAService.get_instance(config)
    
    @app.get("/health")
    def health_check():
        """健康检查"""
        return {
            "status": "healthy",
            "aga_initialized": service.aga is not None,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    @app.post("/inject")
    def inject_knowledge(request: InjectKnowledgeRequest):
        """注入知识"""
        try:
            result = service.inject_knowledge(
                slot_idx=request.slot_idx,
                key_vector=request.key_vector,
                value_vector=request.value_vector,
                lu_id=request.lu_id,
                lifecycle_state=request.lifecycle_state,
                condition=request.condition,
                decision=request.decision,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/inject/batch")
    def batch_inject(request: BatchInjectRequest):
        """批量注入"""
        try:
            items = [item.dict() for item in request.items]
            return service.batch_inject(items)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/lifecycle/update")
    def update_lifecycle(request: UpdateLifecycleRequest):
        """更新生命周期"""
        try:
            return service.update_lifecycle(request.slot_idx, request.new_state)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/quarantine/slot/{slot_idx}")
    def quarantine_slot(slot_idx: int):
        """隔离槽位"""
        try:
            return service.quarantine_slot(slot_idx)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/quarantine/lu")
    def quarantine_by_lu_id(request: QuarantineByLuIdRequest):
        """按 LU ID 隔离"""
        try:
            return service.quarantine_by_lu_id(request.lu_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/slot/free")
    def find_free_slot():
        """查找空闲槽位"""
        slot_idx = service.find_free_slot()
        return {"free_slot": slot_idx}
    
    @app.get("/slot/{slot_idx}")
    def get_slot_info(slot_idx: int):
        """获取槽位信息"""
        try:
            return service.get_slot_info(slot_idx)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/slots/by-lu/{lu_id}")
    def get_slots_by_lu_id(lu_id: str):
        """按 LU ID 查找槽位"""
        slots = service.get_slots_by_lu_id(lu_id)
        return {"lu_id": lu_id, "slots": slots}
    
    @app.get("/statistics")
    def get_statistics():
        """获取统计信息"""
        return service.get_statistics()
    
    return app


# ============================================================
# AGA HTTP 客户端
# ============================================================

class AGAClient:
    """
    AGA HTTP 客户端
    
    用于后端服务远程调用 AGA API。
    
    使用示例：
        client = AGAClient("http://gpu-server:8081")
        
        # 注入知识
        result = client.inject_knowledge(
            slot_idx=0,
            key_vector=[...],
            value_vector=[...],
            lu_id="LU_001",
        )
        
        # 获取统计
        stats = client.get_statistics()
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Args:
            base_url: AGA API 基础 URL，如 "http://localhost:8081"
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        
        try:
            import httpx
            self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
            self._async_client = None
        except ImportError:
            import urllib.request
            self._client = None
    
    def _request(self, method: str, path: str, json_data: Optional[Dict] = None) -> Dict:
        """发送请求"""
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
            import urllib.request
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
            
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
    
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
        """注入知识"""
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
        """批量注入"""
        return self._request("POST", "/inject/batch", {"items": items})
    
    def update_lifecycle(self, slot_idx: int, new_state: str) -> Dict[str, Any]:
        """更新生命周期"""
        return self._request("POST", "/lifecycle/update", {
            "slot_idx": slot_idx,
            "new_state": new_state,
        })
    
    def quarantine_slot(self, slot_idx: int) -> Dict[str, Any]:
        """隔离槽位"""
        return self._request("POST", f"/quarantine/slot/{slot_idx}")
    
    def quarantine_by_lu_id(self, lu_id: str) -> Dict[str, Any]:
        """按 LU ID 隔离"""
        return self._request("POST", "/quarantine/lu", {"lu_id": lu_id})
    
    def find_free_slot(self) -> Optional[int]:
        """查找空闲槽位"""
        result = self._request("GET", "/slot/free")
        return result.get("free_slot")
    
    def get_slot_info(self, slot_idx: int) -> Dict[str, Any]:
        """获取槽位信息"""
        return self._request("GET", f"/slot/{slot_idx}")
    
    def get_slots_by_lu_id(self, lu_id: str) -> List[int]:
        """按 LU ID 查找槽位"""
        result = self._request("GET", f"/slots/by-lu/{lu_id}")
        return result.get("slots", [])
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._request("GET", "/statistics")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start AGA HTTP API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8081, help="Port to bind")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--bottleneck-dim", type=int, default=64, help="Bottleneck dimension")
    parser.add_argument("--num-slots", type=int, default=100, help="Number of slots")
    
    args = parser.parse_args()
    
    config = AGAConfig(
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_slots=args.num_slots,
    )
    
    app = create_aga_api(config)
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

