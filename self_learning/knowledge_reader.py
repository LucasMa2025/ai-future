"""
知识读取器模块

实现自学习系统对生产知识库的只读访问。

核心组件：
1. ProductionKnowledgeReader - 查询生产库中已有的知识
2. ApprovedLUReader - 读取已审批的 Learning Unit

设计原则：
- 只读访问：自学习系统只能读取，不能写入生产库
- 查询优化：支持语义搜索、领域过滤、相关性排序
- 审计追踪：所有查询操作都被记录

使用场景：
- 学习起点选择：基于现有知识确定探索方向
- 知识去重：避免重复学习已有知识
- 链式学习：在已有知识基础上继续探索
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
import hashlib
import json

logger = logging.getLogger(__name__)


class KnowledgeType(Enum):
    """知识类型"""
    DECISION_RULE = "decision_rule"
    KNOWLEDGE = "knowledge"
    PATTERN = "pattern"
    CONSTRAINT = "constraint"


@dataclass
class ProductionKnowledge:
    """
    生产知识记录
    
    表示已内化到生产系统的知识单元
    """
    id: str
    domain: str
    knowledge_type: KnowledgeType
    condition: str
    decision: str
    confidence: float
    risk_level: str
    
    # 来源信息
    source_lu_id: str
    source_lu_title: Optional[str] = None
    source_goal: Optional[str] = None
    
    # 状态
    status: str = "active"
    hit_count: int = 0
    
    # AGA 映射
    aga_slot_idx: Optional[int] = None
    aga_layer_idx: Optional[int] = None
    
    # 时间戳
    created_at: Optional[datetime] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "knowledge_type": self.knowledge_type.value,
            "condition": self.condition,
            "decision": self.decision,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "source_lu_id": self.source_lu_id,
            "source_lu_title": self.source_lu_title,
            "source_goal": self.source_goal,
            "status": self.status,
            "hit_count": self.hit_count,
            "aga_slot_idx": self.aga_slot_idx,
            "aga_layer_idx": self.aga_layer_idx,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionKnowledge':
        return cls(
            id=data["id"],
            domain=data["domain"],
            knowledge_type=KnowledgeType(data.get("knowledge_type", "knowledge")),
            condition=data["condition"],
            decision=data["decision"],
            confidence=data.get("confidence", 0.5),
            risk_level=data.get("risk_level", "low"),
            source_lu_id=data.get("source_lu_id", ""),
            source_lu_title=data.get("source_lu_title"),
            source_goal=data.get("source_goal"),
            status=data.get("status", "active"),
            hit_count=data.get("hit_count", 0),
            aga_slot_idx=data.get("aga_slot_idx"),
            aga_layer_idx=data.get("aga_layer_idx"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ApprovedLearningUnit:
    """
    已审批的 Learning Unit
    
    用于链式学习的基础
    """
    id: str
    title: str
    learning_goal: str
    status: str
    
    # 知识内容
    knowledge: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    
    # 来源追溯
    provenance: Dict[str, Any]
    
    # 风险信息
    risk_level: str
    risk_score: float
    
    # 审批信息
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # 链式学习信息
    parent_lu_id: Optional[str] = None
    chain_depth: int = 0
    chain_root_id: Optional[str] = None
    
    # 内化状态
    is_internalized: bool = False
    lifecycle_state: str = "quarantined"
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # 时间戳
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "learning_goal": self.learning_goal,
            "status": self.status,
            "knowledge": self.knowledge,
            "constraints": self.constraints,
            "provenance": self.provenance,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "parent_lu_id": self.parent_lu_id,
            "chain_depth": self.chain_depth,
            "chain_root_id": self.chain_root_id,
            "is_internalized": self.is_internalized,
            "lifecycle_state": self.lifecycle_state,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    def get_context_for_continuation(self) -> Dict[str, Any]:
        """
        获取用于继续学习的上下文
        
        Returns:
            包含继续学习所需信息的上下文字典
        """
        return {
            "parent_lu_id": self.id,
            "parent_title": self.title,
            "parent_goal": self.learning_goal,
            "parent_knowledge": self.knowledge,
            "parent_constraints": self.constraints,
            "chain_depth": self.chain_depth + 1,
            "chain_root_id": self.chain_root_id or self.id,
            "inherited_context": {
                "domain": self.knowledge.get("domain"),
                "key_findings": self.knowledge.get("content", {}).get("key_findings", []),
                "explored_aspects": self.knowledge.get("content", {}).get("explored_aspects", []),
            }
        }


@dataclass
class KnowledgeSearchResult:
    """知识搜索结果"""
    knowledge: ProductionKnowledge
    relevance_score: float
    match_reason: str


class ProductionKnowledgeReader(ABC):
    """
    生产知识读取器抽象基类
    
    提供对生产知识库的只读访问接口。
    
    设计原则：
    - 只读：自学习系统无权写入
    - 审计：所有查询被记录
    - 高效：支持索引和缓存
    """
    
    @abstractmethod
    def search_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[KnowledgeSearchResult]:
        """
        搜索相关知识
        
        Args:
            query: 搜索查询（语义搜索）
            domain: 领域过滤
            knowledge_type: 知识类型过滤
            min_confidence: 最小置信度
            limit: 返回数量限制
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    def get_knowledge_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> List[ProductionKnowledge]:
        """
        按领域获取知识
        
        Args:
            domain: 领域名称
            limit: 返回数量限制
            
        Returns:
            知识列表
        """
        pass
    
    @abstractmethod
    def get_knowledge_by_lu_id(
        self,
        lu_id: str,
    ) -> List[ProductionKnowledge]:
        """
        按来源 LU ID 获取知识
        
        Args:
            lu_id: Learning Unit ID
            
        Returns:
            知识列表
        """
        pass
    
    @abstractmethod
    def get_related_knowledge(
        self,
        knowledge_id: str,
        limit: int = 10,
    ) -> List[ProductionKnowledge]:
        """
        获取相关知识
        
        Args:
            knowledge_id: 知识 ID
            limit: 返回数量限制
            
        Returns:
            相关知识列表
        """
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        
        Returns:
            统计信息字典
        """
        pass
    
    @abstractmethod
    def check_knowledge_exists(
        self,
        condition: str,
        decision: str,
        threshold: float = 0.9,
    ) -> Optional[ProductionKnowledge]:
        """
        检查知识是否已存在（去重）
        
        Args:
            condition: 条件
            decision: 决策
            threshold: 相似度阈值
            
        Returns:
            如果存在相似知识则返回，否则返回 None
        """
        pass


class ApprovedLUReader(ABC):
    """
    已审批 Learning Unit 读取器抽象基类
    
    提供对已审批 LU 的只读访问，支持链式学习。
    """
    
    @abstractmethod
    def get_approved_lu(self, lu_id: str) -> Optional[ApprovedLearningUnit]:
        """
        获取已审批的 Learning Unit
        
        Args:
            lu_id: Learning Unit ID
            
        Returns:
            已审批的 LU，如果不存在或未审批则返回 None
        """
        pass
    
    @abstractmethod
    def list_approved_lus(
        self,
        domain: Optional[str] = None,
        risk_level: Optional[str] = None,
        is_internalized: Optional[bool] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[ApprovedLearningUnit], int]:
        """
        列出已审批的 Learning Units
        
        Args:
            domain: 领域过滤
            risk_level: 风险等级过滤
            is_internalized: 是否已内化过滤
            skip: 跳过数量
            limit: 返回数量限制
            
        Returns:
            (LU 列表, 总数)
        """
        pass
    
    @abstractmethod
    def get_lu_chain(self, lu_id: str) -> List[ApprovedLearningUnit]:
        """
        获取 Learning Unit 的完整学习链
        
        Args:
            lu_id: Learning Unit ID
            
        Returns:
            从根到当前 LU 的完整链
        """
        pass
    
    @abstractmethod
    def get_lu_children(self, lu_id: str) -> List[ApprovedLearningUnit]:
        """
        获取 Learning Unit 的子节点
        
        Args:
            lu_id: Learning Unit ID
            
        Returns:
            子 LU 列表
        """
        pass
    
    @abstractmethod
    def search_lus(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ApprovedLearningUnit]:
        """
        搜索 Learning Units
        
        Args:
            query: 搜索查询
            limit: 返回数量限制
            
        Returns:
            匹配的 LU 列表
        """
        pass
    
    @abstractmethod
    def get_continuable_lus(
        self,
        max_chain_depth: int = 5,
        limit: int = 20,
    ) -> List[ApprovedLearningUnit]:
        """
        获取可继续学习的 Learning Units
        
        返回适合作为继续学习起点的 LU：
        - 已审批且已内化
        - 链深度未超过限制
        - 没有被标记为"完成"
        
        Args:
            max_chain_depth: 最大链深度
            limit: 返回数量限制
            
        Returns:
            可继续学习的 LU 列表
        """
        pass


# ==================== 实现类 ====================


class InMemoryProductionKnowledgeReader(ProductionKnowledgeReader):
    """
    内存实现的生产知识读取器
    
    用于测试和演示，生产环境应使用数据库实现
    """
    
    def __init__(self):
        self._knowledge: Dict[str, ProductionKnowledge] = {}
        self._query_log: List[Dict[str, Any]] = []
    
    def add_knowledge(self, knowledge: ProductionKnowledge):
        """添加知识（仅用于测试）"""
        self._knowledge[knowledge.id] = knowledge
    
    def search_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[KnowledgeSearchResult]:
        """搜索相关知识"""
        self._log_query("search", {"query": query, "domain": domain})
        
        results = []
        query_lower = query.lower()
        
        for k in self._knowledge.values():
            # 过滤条件
            if k.status != "active":
                continue
            if domain and k.domain != domain:
                continue
            if knowledge_type and k.knowledge_type != knowledge_type:
                continue
            if k.confidence < min_confidence:
                continue
            
            # 简单的文本匹配评分
            score = 0.0
            match_reasons = []
            
            if query_lower in k.condition.lower():
                score += 0.5
                match_reasons.append("condition_match")
            if query_lower in k.decision.lower():
                score += 0.3
                match_reasons.append("decision_match")
            if k.source_goal and query_lower in k.source_goal.lower():
                score += 0.2
                match_reasons.append("goal_match")
            
            if score > 0:
                results.append(KnowledgeSearchResult(
                    knowledge=k,
                    relevance_score=score,
                    match_reason=", ".join(match_reasons),
                ))
        
        # 按相关性排序
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
    
    def get_knowledge_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> List[ProductionKnowledge]:
        """按领域获取知识"""
        self._log_query("get_by_domain", {"domain": domain})
        
        results = [
            k for k in self._knowledge.values()
            if k.domain == domain and k.status == "active"
        ]
        return results[:limit]
    
    def get_knowledge_by_lu_id(
        self,
        lu_id: str,
    ) -> List[ProductionKnowledge]:
        """按来源 LU ID 获取知识"""
        self._log_query("get_by_lu_id", {"lu_id": lu_id})
        
        return [
            k for k in self._knowledge.values()
            if k.source_lu_id == lu_id
        ]
    
    def get_related_knowledge(
        self,
        knowledge_id: str,
        limit: int = 10,
    ) -> List[ProductionKnowledge]:
        """获取相关知识"""
        self._log_query("get_related", {"knowledge_id": knowledge_id})
        
        source = self._knowledge.get(knowledge_id)
        if not source:
            return []
        
        # 简单实现：同领域的其他知识
        results = [
            k for k in self._knowledge.values()
            if k.id != knowledge_id 
            and k.domain == source.domain 
            and k.status == "active"
        ]
        return results[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        active = [k for k in self._knowledge.values() if k.status == "active"]
        
        domains = {}
        types = {}
        for k in active:
            domains[k.domain] = domains.get(k.domain, 0) + 1
            types[k.knowledge_type.value] = types.get(k.knowledge_type.value, 0) + 1
        
        return {
            "total_knowledge": len(self._knowledge),
            "active_knowledge": len(active),
            "by_domain": domains,
            "by_type": types,
            "total_queries": len(self._query_log),
        }
    
    def check_knowledge_exists(
        self,
        condition: str,
        decision: str,
        threshold: float = 0.9,
    ) -> Optional[ProductionKnowledge]:
        """检查知识是否已存在"""
        self._log_query("check_exists", {"condition": condition[:50], "decision": decision[:50]})
        
        # 简单的字符串相似度检查
        for k in self._knowledge.values():
            if k.status != "active":
                continue
            
            # 计算简单的相似度
            cond_sim = self._simple_similarity(condition, k.condition)
            dec_sim = self._simple_similarity(decision, k.decision)
            
            if cond_sim >= threshold and dec_sim >= threshold:
                return k
        
        return None
    
    def _simple_similarity(self, s1: str, s2: str) -> float:
        """简单的字符串相似度"""
        if not s1 or not s2:
            return 0.0
        
        s1_lower = s1.lower()
        s2_lower = s2.lower()
        
        if s1_lower == s2_lower:
            return 1.0
        
        # Jaccard 相似度
        words1 = set(s1_lower.split())
        words2 = set(s2_lower.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _log_query(self, query_type: str, params: Dict[str, Any]):
        """记录查询日志"""
        self._query_log.append({
            "type": query_type,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        })


class InMemoryApprovedLUReader(ApprovedLUReader):
    """
    内存实现的已审批 LU 读取器
    
    用于测试和演示，生产环境应使用数据库实现
    """
    
    def __init__(self):
        self._lus: Dict[str, ApprovedLearningUnit] = {}
        self._query_log: List[Dict[str, Any]] = []
    
    def add_lu(self, lu: ApprovedLearningUnit):
        """添加 LU（仅用于测试）"""
        self._lus[lu.id] = lu
    
    def get_approved_lu(self, lu_id: str) -> Optional[ApprovedLearningUnit]:
        """获取已审批的 Learning Unit"""
        self._log_query("get_lu", {"lu_id": lu_id})
        
        lu = self._lus.get(lu_id)
        if lu and lu.status in ["approved", "corrected"]:
            return lu
        return None
    
    def list_approved_lus(
        self,
        domain: Optional[str] = None,
        risk_level: Optional[str] = None,
        is_internalized: Optional[bool] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[ApprovedLearningUnit], int]:
        """列出已审批的 Learning Units"""
        self._log_query("list_lus", {
            "domain": domain, 
            "risk_level": risk_level,
            "is_internalized": is_internalized,
        })
        
        results = []
        for lu in self._lus.values():
            if lu.status not in ["approved", "corrected"]:
                continue
            if domain and lu.knowledge.get("domain") != domain:
                continue
            if risk_level and lu.risk_level != risk_level:
                continue
            if is_internalized is not None and lu.is_internalized != is_internalized:
                continue
            results.append(lu)
        
        total = len(results)
        return results[skip:skip + limit], total
    
    def get_lu_chain(self, lu_id: str) -> List[ApprovedLearningUnit]:
        """获取 Learning Unit 的完整学习链"""
        self._log_query("get_chain", {"lu_id": lu_id})
        
        lu = self._lus.get(lu_id)
        if not lu:
            return []
        
        # 向上追溯到根
        chain = []
        current = lu
        while current:
            chain.insert(0, current)
            if current.parent_lu_id:
                current = self._lus.get(current.parent_lu_id)
            else:
                break
        
        return chain
    
    def get_lu_children(self, lu_id: str) -> List[ApprovedLearningUnit]:
        """获取 Learning Unit 的子节点"""
        self._log_query("get_children", {"lu_id": lu_id})
        
        return [
            lu for lu in self._lus.values()
            if lu.parent_lu_id == lu_id
        ]
    
    def search_lus(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ApprovedLearningUnit]:
        """搜索 Learning Units"""
        self._log_query("search", {"query": query})
        
        query_lower = query.lower()
        results = []
        
        for lu in self._lus.values():
            if lu.status not in ["approved", "corrected"]:
                continue
            
            # 简单的文本匹配
            if (query_lower in lu.title.lower() or 
                query_lower in lu.learning_goal.lower()):
                results.append(lu)
        
        return results[:limit]
    
    def get_continuable_lus(
        self,
        max_chain_depth: int = 5,
        limit: int = 20,
    ) -> List[ApprovedLearningUnit]:
        """获取可继续学习的 Learning Units"""
        self._log_query("get_continuable", {"max_chain_depth": max_chain_depth})
        
        results = []
        for lu in self._lus.values():
            if lu.status not in ["approved", "corrected"]:
                continue
            if not lu.is_internalized:
                continue
            if lu.chain_depth >= max_chain_depth:
                continue
            if lu.metadata.get("learning_completed"):
                continue
            
            results.append(lu)
        
        # 按链深度排序（优先浅层）
        results.sort(key=lambda x: x.chain_depth)
        return results[:limit]
    
    def _log_query(self, query_type: str, params: Dict[str, Any]):
        """记录查询日志"""
        self._query_log.append({
            "type": query_type,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        })


# ==================== 数据库实现 ====================


class DatabaseProductionKnowledgeReader(ProductionKnowledgeReader):
    """
    数据库实现的生产知识读取器
    
    使用 SQLAlchemy 访问 PostgreSQL 数据库
    """
    
    def __init__(self, db_session):
        """
        初始化
        
        Args:
            db_session: SQLAlchemy Session
        """
        self.db = db_session
        self._query_log: List[Dict[str, Any]] = []
    
    def search_knowledge(
        self,
        query: str,
        domain: Optional[str] = None,
        knowledge_type: Optional[KnowledgeType] = None,
        min_confidence: float = 0.0,
        limit: int = 10,
    ) -> List[KnowledgeSearchResult]:
        """搜索相关知识"""
        self._log_query("search", {"query": query, "domain": domain})
        
        # 使用 PostgreSQL 全文搜索
        # 注意：这需要数据库中有 production_knowledge 表
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT 
                    pk.*,
                    ts_rank(
                        to_tsvector('english', pk.condition || ' ' || pk.decision),
                        plainto_tsquery('english', :query)
                    ) as relevance_score
                FROM production_knowledge pk
                WHERE pk.status = 'active'
                  AND pk.confidence >= :min_confidence
                  AND (
                      to_tsvector('english', pk.condition || ' ' || pk.decision) 
                      @@ plainto_tsquery('english', :query)
                  )
                  AND (:domain IS NULL OR pk.domain = :domain)
                  AND (:knowledge_type IS NULL OR pk.knowledge_type = :knowledge_type)
                ORDER BY relevance_score DESC
                LIMIT :limit
            """)
            
            result = self.db.execute(sql, {
                "query": query,
                "domain": domain,
                "knowledge_type": knowledge_type.value if knowledge_type else None,
                "min_confidence": min_confidence,
                "limit": limit,
            })
            
            results = []
            for row in result:
                knowledge = ProductionKnowledge(
                    id=str(row.id),
                    domain=row.domain,
                    knowledge_type=KnowledgeType(row.knowledge_type),
                    condition=row.condition,
                    decision=row.decision,
                    confidence=row.confidence,
                    risk_level=row.risk_level or "low",
                    source_lu_id=str(row.source_lu_id),
                    status=row.status,
                    hit_count=row.hit_count or 0,
                    aga_slot_idx=row.aga_slot_idx,
                    aga_layer_idx=row.aga_layer_idx,
                )
                results.append(KnowledgeSearchResult(
                    knowledge=knowledge,
                    relevance_score=row.relevance_score,
                    match_reason="full_text_search",
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Search knowledge failed: {e}")
            return []
    
    def get_knowledge_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> List[ProductionKnowledge]:
        """按领域获取知识"""
        self._log_query("get_by_domain", {"domain": domain})
        
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT * FROM production_knowledge
                WHERE domain = :domain AND status = 'active'
                ORDER BY confidence DESC
                LIMIT :limit
            """)
            
            result = self.db.execute(sql, {"domain": domain, "limit": limit})
            
            return [self._row_to_knowledge(row) for row in result]
            
        except Exception as e:
            logger.error(f"Get knowledge by domain failed: {e}")
            return []
    
    def get_knowledge_by_lu_id(
        self,
        lu_id: str,
    ) -> List[ProductionKnowledge]:
        """按来源 LU ID 获取知识"""
        self._log_query("get_by_lu_id", {"lu_id": lu_id})
        
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT * FROM production_knowledge
                WHERE source_lu_id = :lu_id
            """)
            
            result = self.db.execute(sql, {"lu_id": lu_id})
            
            return [self._row_to_knowledge(row) for row in result]
            
        except Exception as e:
            logger.error(f"Get knowledge by LU ID failed: {e}")
            return []
    
    def get_related_knowledge(
        self,
        knowledge_id: str,
        limit: int = 10,
    ) -> List[ProductionKnowledge]:
        """获取相关知识"""
        self._log_query("get_related", {"knowledge_id": knowledge_id})
        
        try:
            from sqlalchemy import text
            
            # 先获取源知识的领域
            sql = text("""
                SELECT domain FROM production_knowledge WHERE id = :id
            """)
            result = self.db.execute(sql, {"id": knowledge_id}).fetchone()
            
            if not result:
                return []
            
            domain = result.domain
            
            # 获取同领域的其他知识
            sql = text("""
                SELECT * FROM production_knowledge
                WHERE domain = :domain 
                  AND id != :id 
                  AND status = 'active'
                ORDER BY confidence DESC
                LIMIT :limit
            """)
            
            result = self.db.execute(sql, {
                "domain": domain,
                "id": knowledge_id,
                "limit": limit,
            })
            
            return [self._row_to_knowledge(row) for row in result]
            
        except Exception as e:
            logger.error(f"Get related knowledge failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        try:
            from sqlalchemy import text
            
            stats = {}
            
            # 总数
            sql = text("SELECT COUNT(*) as count FROM production_knowledge")
            stats["total_knowledge"] = self.db.execute(sql).fetchone().count
            
            # 活跃数
            sql = text("SELECT COUNT(*) as count FROM production_knowledge WHERE status = 'active'")
            stats["active_knowledge"] = self.db.execute(sql).fetchone().count
            
            # 按领域统计
            sql = text("""
                SELECT domain, COUNT(*) as count 
                FROM production_knowledge 
                WHERE status = 'active'
                GROUP BY domain
            """)
            stats["by_domain"] = {row.domain: row.count for row in self.db.execute(sql)}
            
            # 按类型统计
            sql = text("""
                SELECT knowledge_type, COUNT(*) as count 
                FROM production_knowledge 
                WHERE status = 'active'
                GROUP BY knowledge_type
            """)
            stats["by_type"] = {row.knowledge_type: row.count for row in self.db.execute(sql)}
            
            return stats
            
        except Exception as e:
            logger.error(f"Get statistics failed: {e}")
            return {}
    
    def check_knowledge_exists(
        self,
        condition: str,
        decision: str,
        threshold: float = 0.9,
    ) -> Optional[ProductionKnowledge]:
        """检查知识是否已存在"""
        self._log_query("check_exists", {"condition": condition[:50], "decision": decision[:50]})
        
        try:
            from sqlalchemy import text
            
            # 使用 PostgreSQL 的 similarity 函数（需要 pg_trgm 扩展）
            sql = text("""
                SELECT *, 
                       similarity(condition, :condition) as cond_sim,
                       similarity(decision, :decision) as dec_sim
                FROM production_knowledge
                WHERE status = 'active'
                  AND similarity(condition, :condition) >= :threshold
                  AND similarity(decision, :decision) >= :threshold
                ORDER BY (similarity(condition, :condition) + similarity(decision, :decision)) DESC
                LIMIT 1
            """)
            
            result = self.db.execute(sql, {
                "condition": condition,
                "decision": decision,
                "threshold": threshold,
            }).fetchone()
            
            if result:
                return self._row_to_knowledge(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Check knowledge exists failed: {e}")
            return None
    
    def _row_to_knowledge(self, row) -> ProductionKnowledge:
        """转换数据库行到 ProductionKnowledge"""
        return ProductionKnowledge(
            id=str(row.id),
            domain=row.domain,
            knowledge_type=KnowledgeType(row.knowledge_type),
            condition=row.condition,
            decision=row.decision,
            confidence=row.confidence or 0.5,
            risk_level=row.risk_level or "low",
            source_lu_id=str(row.source_lu_id),
            status=row.status,
            hit_count=row.hit_count or 0,
            aga_slot_idx=row.aga_slot_idx,
            aga_layer_idx=row.aga_layer_idx,
            created_at=row.created_at,
        )
    
    def _log_query(self, query_type: str, params: Dict[str, Any]):
        """记录查询日志"""
        self._query_log.append({
            "type": query_type,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        })


class DatabaseApprovedLUReader(ApprovedLUReader):
    """
    数据库实现的已审批 LU 读取器
    
    使用 SQLAlchemy 访问 PostgreSQL 数据库
    """
    
    def __init__(self, db_session):
        """
        初始化
        
        Args:
            db_session: SQLAlchemy Session
        """
        self.db = db_session
        self._query_log: List[Dict[str, Any]] = []
    
    def get_approved_lu(self, lu_id: str) -> Optional[ApprovedLearningUnit]:
        """获取已审批的 Learning Unit"""
        self._log_query("get_lu", {"lu_id": lu_id})
        
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT lu.*, 
                       u.username as approver_name
                FROM learning_units lu
                LEFT JOIN users u ON lu.approved_by = u.id
                WHERE lu.id = :lu_id
                  AND lu.status IN ('approved', 'corrected')
            """)
            
            result = self.db.execute(sql, {"lu_id": lu_id}).fetchone()
            
            if result:
                return self._row_to_lu(result)
            
            return None
            
        except Exception as e:
            logger.error(f"Get approved LU failed: {e}")
            return None
    
    def list_approved_lus(
        self,
        domain: Optional[str] = None,
        risk_level: Optional[str] = None,
        is_internalized: Optional[bool] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> Tuple[List[ApprovedLearningUnit], int]:
        """列出已审批的 Learning Units"""
        self._log_query("list_lus", {
            "domain": domain,
            "risk_level": risk_level,
            "is_internalized": is_internalized,
        })
        
        try:
            from sqlalchemy import text
            
            # 构建查询
            where_clauses = ["lu.status IN ('approved', 'corrected')"]
            params = {"skip": skip, "limit": limit}
            
            if domain:
                where_clauses.append("lu.knowledge->>'domain' = :domain")
                params["domain"] = domain
            
            if risk_level:
                where_clauses.append("lu.risk_level = :risk_level")
                params["risk_level"] = risk_level
            
            if is_internalized is not None:
                where_clauses.append("lu.is_internalized = :is_internalized")
                params["is_internalized"] = is_internalized
            
            where_sql = " AND ".join(where_clauses)
            
            # 获取总数
            count_sql = text(f"SELECT COUNT(*) as count FROM learning_units lu WHERE {where_sql}")
            total = self.db.execute(count_sql, params).fetchone().count
            
            # 获取数据
            sql = text(f"""
                SELECT lu.*, u.username as approver_name
                FROM learning_units lu
                LEFT JOIN users u ON lu.approved_by = u.id
                WHERE {where_sql}
                ORDER BY lu.created_at DESC
                OFFSET :skip LIMIT :limit
            """)
            
            result = self.db.execute(sql, params)
            lus = [self._row_to_lu(row) for row in result]
            
            return lus, total
            
        except Exception as e:
            logger.error(f"List approved LUs failed: {e}")
            return [], 0
    
    def get_lu_chain(self, lu_id: str) -> List[ApprovedLearningUnit]:
        """获取 Learning Unit 的完整学习链"""
        self._log_query("get_chain", {"lu_id": lu_id})
        
        try:
            from sqlalchemy import text
            
            # 使用递归 CTE 获取完整链
            sql = text("""
                WITH RECURSIVE chain AS (
                    -- 基础情况：目标 LU
                    SELECT lu.*, 0 as depth
                    FROM learning_units lu
                    WHERE lu.id = :lu_id
                    
                    UNION ALL
                    
                    -- 递归：向上追溯父节点
                    SELECT lu.*, c.depth - 1
                    FROM learning_units lu
                    JOIN chain c ON lu.id = c.parent_lu_id
                )
                SELECT * FROM chain
                ORDER BY depth
            """)
            
            result = self.db.execute(sql, {"lu_id": lu_id})
            return [self._row_to_lu(row) for row in result]
            
        except Exception as e:
            logger.error(f"Get LU chain failed: {e}")
            return []
    
    def get_lu_children(self, lu_id: str) -> List[ApprovedLearningUnit]:
        """获取 Learning Unit 的子节点"""
        self._log_query("get_children", {"lu_id": lu_id})
        
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT lu.*, u.username as approver_name
                FROM learning_units lu
                LEFT JOIN users u ON lu.approved_by = u.id
                WHERE lu.parent_lu_id = :lu_id
                ORDER BY lu.created_at
            """)
            
            result = self.db.execute(sql, {"lu_id": lu_id})
            return [self._row_to_lu(row) for row in result]
            
        except Exception as e:
            logger.error(f"Get LU children failed: {e}")
            return []
    
    def search_lus(
        self,
        query: str,
        limit: int = 10,
    ) -> List[ApprovedLearningUnit]:
        """搜索 Learning Units"""
        self._log_query("search", {"query": query})
        
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT lu.*, u.username as approver_name,
                       ts_rank(
                           to_tsvector('english', lu.title || ' ' || COALESCE(lu.learning_goal, '')),
                           plainto_tsquery('english', :query)
                       ) as relevance
                FROM learning_units lu
                LEFT JOIN users u ON lu.approved_by = u.id
                WHERE lu.status IN ('approved', 'corrected')
                  AND to_tsvector('english', lu.title || ' ' || COALESCE(lu.learning_goal, ''))
                      @@ plainto_tsquery('english', :query)
                ORDER BY relevance DESC
                LIMIT :limit
            """)
            
            result = self.db.execute(sql, {"query": query, "limit": limit})
            return [self._row_to_lu(row) for row in result]
            
        except Exception as e:
            logger.error(f"Search LUs failed: {e}")
            return []
    
    def get_continuable_lus(
        self,
        max_chain_depth: int = 5,
        limit: int = 20,
    ) -> List[ApprovedLearningUnit]:
        """获取可继续学习的 Learning Units"""
        self._log_query("get_continuable", {"max_chain_depth": max_chain_depth})
        
        try:
            from sqlalchemy import text
            
            sql = text("""
                SELECT lu.*, u.username as approver_name
                FROM learning_units lu
                LEFT JOIN users u ON lu.approved_by = u.id
                WHERE lu.status IN ('approved', 'corrected')
                  AND lu.is_internalized = true
                  AND COALESCE(lu.chain_depth, 0) < :max_chain_depth
                  AND COALESCE(lu.metadata->>'learning_completed', 'false') != 'true'
                ORDER BY lu.chain_depth, lu.created_at DESC
                LIMIT :limit
            """)
            
            result = self.db.execute(sql, {
                "max_chain_depth": max_chain_depth,
                "limit": limit,
            })
            return [self._row_to_lu(row) for row in result]
            
        except Exception as e:
            logger.error(f"Get continuable LUs failed: {e}")
            return []
    
    def _row_to_lu(self, row) -> ApprovedLearningUnit:
        """转换数据库行到 ApprovedLearningUnit"""
        # 获取约束
        constraints = []
        try:
            from sqlalchemy import text
            sql = text("""
                SELECT * FROM lu_constraints WHERE learning_unit_id = :lu_id
            """)
            result = self.db.execute(sql, {"lu_id": str(row.id)})
            for c in result:
                constraints.append({
                    "id": str(c.id),
                    "condition": c.condition,
                    "decision": c.decision,
                    "confidence": c.confidence,
                })
        except Exception:
            pass
        
        return ApprovedLearningUnit(
            id=str(row.id),
            title=row.title,
            learning_goal=row.learning_goal or "",
            status=row.status,
            knowledge=row.knowledge or {},
            constraints=constraints,
            provenance=row.provenance or {},
            risk_level=row.risk_level or "low",
            risk_score=row.risk_score or 0.0,
            approved_by=getattr(row, 'approver_name', None),
            approved_at=row.approved_at,
            parent_lu_id=str(row.parent_lu_id) if row.parent_lu_id else None,
            chain_depth=row.chain_depth or 0,
            chain_root_id=str(row.chain_root_id) if row.chain_root_id else None,
            is_internalized=row.is_internalized or False,
            lifecycle_state=row.lifecycle_state or "quarantined",
            metadata=row.metadata or {},
            tags=row.tags or [],
            created_at=row.created_at,
        )
    
    def _log_query(self, query_type: str, params: Dict[str, Any]):
        """记录查询日志"""
        self._query_log.append({
            "type": query_type,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        })

