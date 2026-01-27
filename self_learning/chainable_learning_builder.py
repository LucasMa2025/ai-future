"""
链式学习构建器

支持在已有 Learning Unit 基础上继续学习，形成知识链。

核心功能：
1. continue_learning() - 在已审批 LU 基础上继续探索
2. 知识图谱追踪 - 追踪 LU 之间的关联关系
3. 学习起点选择 - 基于现有知识库选择最佳起点

设计原则：
- 链式学习必须基于已审批的 LU
- 链深度受治理系统控制
- 所有关联关系都被记录用于审计
"""
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging

from .knowledge_reader import (
    ProductionKnowledgeReader,
    ApprovedLUReader,
    ApprovedLearningUnit,
    ProductionKnowledge,
    KnowledgeSearchResult,
    InMemoryProductionKnowledgeReader,
    InMemoryApprovedLUReader,
)
from .nl_core import (
    NLLevel,
    LearningScope,
    ContextFlowSegment,
    NestedLearningKernel,
    LLMBasedNLKernel,
)

logger = logging.getLogger(__name__)


class LURelationType(Enum):
    """Learning Unit 关系类型"""
    CONTINUES = "continues"       # 继续学习
    REFINES = "refines"          # 细化
    CONTRADICTS = "contradicts"  # 矛盾
    SUPPORTS = "supports"        # 支持
    DEPENDS_ON = "depends_on"    # 依赖
    SUPERSEDES = "supersedes"    # 取代


@dataclass
class LURelation:
    """Learning Unit 关系"""
    source_lu_id: str
    target_lu_id: str
    relation_type: LURelationType
    strength: float = 1.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_lu_id": self.source_lu_id,
            "target_lu_id": self.target_lu_id,
            "relation_type": self.relation_type.value,
            "strength": self.strength,
            "description": self.description,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class LearningChain:
    """学习链"""
    chain_id: str
    root_lu_id: str
    head_lu_id: str
    total_depth: int = 1
    total_units: int = 1
    initial_goal: str = ""
    current_goal: str = ""
    status: str = "active"
    approved_count: int = 0
    rejected_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "root_lu_id": self.root_lu_id,
            "head_lu_id": self.head_lu_id,
            "total_depth": self.total_depth,
            "total_units": self.total_units,
            "initial_goal": self.initial_goal,
            "current_goal": self.current_goal,
            "status": self.status,
            "approved_count": self.approved_count,
            "rejected_count": self.rejected_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ContinueLearningContext:
    """继续学习上下文"""
    parent_lu: ApprovedLearningUnit
    inherited_knowledge: Dict[str, Any]
    inherited_constraints: List[Dict[str, Any]]
    chain_depth: int
    chain_root_id: str
    exploration_direction: str = ""
    focus_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_lu_id": self.parent_lu.id,
            "parent_title": self.parent_lu.title,
            "parent_goal": self.parent_lu.learning_goal,
            "inherited_knowledge": self.inherited_knowledge,
            "inherited_constraints": self.inherited_constraints,
            "chain_depth": self.chain_depth,
            "chain_root_id": self.chain_root_id,
            "exploration_direction": self.exploration_direction,
            "focus_areas": self.focus_areas,
        }


@dataclass
class ChainableLearningUnit:
    """
    可链式学习的 Learning Unit
    
    扩展标准 LearningUnit，添加链式学习支持
    """
    id: str
    title: str
    learning_goal: str
    
    # 知识内容
    knowledge: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    
    # 来源追溯
    provenance: Dict[str, Any]
    
    # 链式学习信息
    parent_lu_id: Optional[str] = None
    chain_depth: int = 0
    chain_root_id: Optional[str] = None
    continue_from_context: Optional[Dict[str, Any]] = None
    
    # 关系
    relations: List[LURelation] = field(default_factory=list)
    
    # 状态
    status: str = "pending"
    risk_level: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "learning_goal": self.learning_goal,
            "knowledge": self.knowledge,
            "constraints": self.constraints,
            "provenance": self.provenance,
            "parent_lu_id": self.parent_lu_id,
            "chain_depth": self.chain_depth,
            "chain_root_id": self.chain_root_id,
            "continue_from_context": self.continue_from_context,
            "relations": [r.to_dict() for r in self.relations],
            "status": self.status,
            "risk_level": self.risk_level,
            "metadata": self.metadata,
            "signals": self.signals,
            "created_at": self.created_at.isoformat(),
        }


class ChainableLearningUnitBuilder:
    """
    链式学习构建器
    
    支持：
    1. 在已审批 LU 基础上继续学习
    2. 知识图谱追踪
    3. 学习起点智能选择
    4. 链深度控制
    
    核心治理原则：
    - 链式学习必须基于已审批的 LU
    - 链深度受治理系统控制（通过 max_chain_depth）
    - 所有关联关系都被记录用于审计
    """
    
    def __init__(
        self,
        knowledge_reader: Optional[ProductionKnowledgeReader] = None,
        lu_reader: Optional[ApprovedLUReader] = None,
        nl_kernel: Optional[NestedLearningKernel] = None,
        llm_adapter = None,
        submit_callback: Optional[Callable[[ChainableLearningUnit], None]] = None,
        max_chain_depth: int = 5,
    ):
        """
        初始化链式学习构建器
        
        Args:
            knowledge_reader: 生产知识读取器
            lu_reader: 已审批 LU 读取器
            nl_kernel: NL 内核
            llm_adapter: LLM 适配器
            submit_callback: 提交回调
            max_chain_depth: 最大链深度（治理控制）
        """
        self.knowledge_reader = knowledge_reader or InMemoryProductionKnowledgeReader()
        self.lu_reader = lu_reader or InMemoryApprovedLUReader()
        self.nl_kernel = nl_kernel or LLMBasedNLKernel(llm_adapter=llm_adapter)
        self.submit_callback = submit_callback
        self.max_chain_depth = max_chain_depth
        
        # 初始化 NL 内核
        if not self.nl_kernel._initialized:
            self.nl_kernel.initialize({})
        
        # 学习链追踪
        self.learning_chains: Dict[str, LearningChain] = {}
        self.lu_relations: List[LURelation] = []
        
        # 构建历史
        self.built_units: List[ChainableLearningUnit] = []
        
        # 当前会话
        self.current_scope: Optional[LearningScope] = None
        self.current_session_id: Optional[str] = None
    
    # ==================== 学习起点选择 ====================
    
    def select_learning_starting_point(
        self,
        goal: str,
        domain: Optional[str] = None,
        strategy: str = "relevance",
    ) -> Tuple[Optional[ApprovedLearningUnit], Dict[str, Any]]:
        """
        选择学习起点
        
        基于现有知识库选择最佳的学习起点。
        
        Args:
            goal: 学习目标
            domain: 领域过滤
            strategy: 选择策略
                - "relevance": 选择最相关的 LU
                - "shallow": 选择链深度最浅的 LU
                - "recent": 选择最近的 LU
                - "high_confidence": 选择置信度最高的 LU
                
        Returns:
            (选中的 LU, 选择原因)
        """
        print(f"\n[ChainBuilder] 选择学习起点...")
        print(f"  目标: {goal}")
        print(f"  策略: {strategy}")
        
        # 1. 搜索相关知识
        related_knowledge = self.knowledge_reader.search_knowledge(
            query=goal,
            domain=domain,
            limit=20,
        )
        
        # 2. 获取可继续学习的 LU
        continuable_lus = self.lu_reader.get_continuable_lus(
            max_chain_depth=self.max_chain_depth,
            limit=20,
        )
        
        if not continuable_lus:
            print("  [ChainBuilder] 没有可继续学习的 LU，将从零开始")
            return None, {
                "reason": "no_continuable_lus",
                "related_knowledge_count": len(related_knowledge),
            }
        
        # 3. 根据策略选择
        selected_lu = None
        selection_reason = {}
        
        if strategy == "relevance":
            # 基于相关性选择
            selected_lu, selection_reason = self._select_by_relevance(
                goal, continuable_lus, related_knowledge
            )
        elif strategy == "shallow":
            # 选择链深度最浅的
            selected_lu = min(continuable_lus, key=lambda x: x.chain_depth)
            selection_reason = {
                "strategy": "shallow",
                "chain_depth": selected_lu.chain_depth,
            }
        elif strategy == "recent":
            # 选择最近的
            selected_lu = max(continuable_lus, key=lambda x: x.created_at or datetime.min)
            selection_reason = {
                "strategy": "recent",
                "created_at": selected_lu.created_at.isoformat() if selected_lu.created_at else None,
            }
        elif strategy == "high_confidence":
            # 选择置信度最高的
            def get_confidence(lu):
                return lu.knowledge.get("confidence", 0) if lu.knowledge else 0
            selected_lu = max(continuable_lus, key=get_confidence)
            selection_reason = {
                "strategy": "high_confidence",
                "confidence": get_confidence(selected_lu),
            }
        else:
            # 默认：第一个
            selected_lu = continuable_lus[0]
            selection_reason = {"strategy": "default"}
        
        if selected_lu:
            print(f"  [ChainBuilder] 选中 LU: {selected_lu.id}")
            print(f"  标题: {selected_lu.title}")
            print(f"  链深度: {selected_lu.chain_depth}")
            print(f"  选择原因: {selection_reason}")
        
        return selected_lu, selection_reason
    
    def _select_by_relevance(
        self,
        goal: str,
        lus: List[ApprovedLearningUnit],
        related_knowledge: List[KnowledgeSearchResult],
    ) -> Tuple[Optional[ApprovedLearningUnit], Dict[str, Any]]:
        """基于相关性选择 LU"""
        # 构建知识来源 LU 的相关性映射
        lu_relevance: Dict[str, float] = {}
        
        for result in related_knowledge:
            lu_id = result.knowledge.source_lu_id
            if lu_id not in lu_relevance:
                lu_relevance[lu_id] = 0.0
            lu_relevance[lu_id] += result.relevance_score
        
        # 计算每个可继续 LU 的得分
        lu_scores = []
        for lu in lus:
            score = lu_relevance.get(lu.id, 0.0)
            
            # 加入目标文本匹配
            goal_lower = goal.lower()
            if goal_lower in lu.learning_goal.lower():
                score += 0.5
            if goal_lower in lu.title.lower():
                score += 0.3
            
            # 惩罚深链
            depth_penalty = lu.chain_depth * 0.1
            score -= depth_penalty
            
            lu_scores.append((lu, score))
        
        # 选择得分最高的
        lu_scores.sort(key=lambda x: x[1], reverse=True)
        
        if lu_scores:
            selected, score = lu_scores[0]
            return selected, {
                "strategy": "relevance",
                "relevance_score": score,
                "knowledge_matches": len([r for r in related_knowledge if r.knowledge.source_lu_id == selected.id]),
            }
        
        return None, {"strategy": "relevance", "reason": "no_matches"}
    
    # ==================== 链式学习 ====================
    
    def continue_learning(
        self,
        parent_lu_id: str,
        new_goal: str,
        exploration_direction: Optional[str] = None,
        focus_areas: Optional[List[str]] = None,
        scope: Optional[LearningScope] = None,
    ) -> Optional[ChainableLearningUnit]:
        """
        在已审批 LU 基础上继续学习
        
        这是链式学习的核心方法。
        
        Args:
            parent_lu_id: 父 LU ID（必须是已审批的）
            new_goal: 新的学习目标
            exploration_direction: 探索方向（可选）
            focus_areas: 重点关注领域（可选）
            scope: 学习范围（由治理系统提供）
            
        Returns:
            新构建的 Learning Unit
        """
        print(f"\n{'='*60}")
        print(f"[ChainBuilder] 继续学习")
        print(f"  父 LU: {parent_lu_id}")
        print(f"  新目标: {new_goal}")
        print(f"{'='*60}")
        
        # 1. 获取父 LU
        parent_lu = self.lu_reader.get_approved_lu(parent_lu_id)
        if not parent_lu:
            logger.error(f"Parent LU not found or not approved: {parent_lu_id}")
            print(f"  [Error] 父 LU 不存在或未审批: {parent_lu_id}")
            return None
        
        # 2. 检查链深度
        new_depth = parent_lu.chain_depth + 1
        if new_depth > self.max_chain_depth:
            logger.warning(f"Chain depth exceeded: {new_depth} > {self.max_chain_depth}")
            print(f"  [Warning] 链深度超限: {new_depth} > {self.max_chain_depth}")
            return None
        
        # 3. 构建继续学习上下文
        continue_context = ContinueLearningContext(
            parent_lu=parent_lu,
            inherited_knowledge=parent_lu.knowledge,
            inherited_constraints=parent_lu.constraints,
            chain_depth=new_depth,
            chain_root_id=parent_lu.chain_root_id or parent_lu.id,
            exploration_direction=exploration_direction or "",
            focus_areas=focus_areas or [],
        )
        
        print(f"  [ChainBuilder] 继续学习上下文:")
        print(f"    父 LU 标题: {parent_lu.title}")
        print(f"    父 LU 目标: {parent_lu.learning_goal}")
        print(f"    链深度: {new_depth}")
        print(f"    链根: {continue_context.chain_root_id}")
        
        # 4. 设置 Scope
        if scope:
            self.current_scope = scope
        else:
            self.current_scope = self._create_default_scope()
        
        # 5. 执行学习
        unit = self._execute_continued_learning(
            new_goal=new_goal,
            continue_context=continue_context,
        )
        
        if unit:
            # 6. 创建关系
            relation = LURelation(
                source_lu_id=unit.id,
                target_lu_id=parent_lu_id,
                relation_type=LURelationType.CONTINUES,
                strength=1.0,
                description=f"Continues learning from: {parent_lu.title}",
                metadata={
                    "new_goal": new_goal,
                    "exploration_direction": exploration_direction,
                },
            )
            unit.relations.append(relation)
            self.lu_relations.append(relation)
            
            # 7. 更新学习链
            self._update_learning_chain(unit, continue_context)
            
            # 8. 记录
            self.built_units.append(unit)
            
            # 9. 提交
            if self.submit_callback:
                print(f"\n[ChainBuilder] 提交 Learning Unit 到审计系统: {unit.id}")
                self.submit_callback(unit)
            
            self._print_build_summary(unit)
        
        return unit
    
    def _execute_continued_learning(
        self,
        new_goal: str,
        continue_context: ContinueLearningContext,
    ) -> Optional[ChainableLearningUnit]:
        """执行继续学习"""
        # 解冻 NL 内核
        self.nl_kernel.unfreeze()
        
        try:
            # 1. 构建增强的学习上下文
            enhanced_context = self._build_enhanced_context(new_goal, continue_context)
            
            # 2. 执行 NL 学习步骤
            segments = self._execute_nl_learning_with_context(
                goal=new_goal,
                context=enhanced_context,
            )
            
            # 3. 生成知识（基于继承的知识）
            knowledge, constraints = self._generate_knowledge_with_inheritance(
                goal=new_goal,
                segments=segments,
                continue_context=continue_context,
            )
            
            # 4. 构建 Learning Unit
            unit_id = f"lu_{uuid.uuid4().hex[:12]}"
            
            unit = ChainableLearningUnit(
                id=unit_id,
                title=f"继续学习: {new_goal[:50]}",
                learning_goal=new_goal,
                knowledge=knowledge,
                constraints=constraints,
                provenance={
                    "learning_goal": new_goal,
                    "parent_lu_id": continue_context.parent_lu.id,
                    "parent_goal": continue_context.parent_lu.learning_goal,
                    "chain_depth": continue_context.chain_depth,
                    "exploration_direction": continue_context.exploration_direction,
                    "focus_areas": continue_context.focus_areas,
                    "segments_count": len(segments),
                },
                parent_lu_id=continue_context.parent_lu.id,
                chain_depth=continue_context.chain_depth,
                chain_root_id=continue_context.chain_root_id,
                continue_from_context=continue_context.to_dict(),
                status="pending",
                metadata={
                    "builder": "chainable_learning_builder",
                    "scope_id": self.current_scope.scope_id if self.current_scope else None,
                    "scope_risk_level": self.current_scope.get_risk_level() if self.current_scope else "low",
                },
                signals=[
                    {
                        "signal_type": "chain_learning",
                        "content": {
                            "parent_lu_id": continue_context.parent_lu.id,
                            "chain_depth": continue_context.chain_depth,
                            "chain_root_id": continue_context.chain_root_id,
                        }
                    },
                    {
                        "signal_type": "nl_segments",
                        "content": {
                            "count": len(segments),
                            "affected_levels": list(set(
                                level.name
                                for seg in segments
                                for level in seg.get_affected_levels()
                            )),
                        }
                    },
                ],
            )
            
            return unit
            
        except Exception as e:
            logger.error(f"Continue learning failed: {e}")
            print(f"  [Error] 继续学习失败: {e}")
            return None
        finally:
            # 冻结 NL 内核
            self.nl_kernel.freeze()
    
    def _build_enhanced_context(
        self,
        goal: str,
        continue_context: ContinueLearningContext,
    ) -> Dict[str, Any]:
        """构建增强的学习上下文"""
        # 从父 LU 继承的知识
        inherited = continue_context.inherited_knowledge or {}
        
        # 搜索相关的生产知识
        related_knowledge = self.knowledge_reader.search_knowledge(
            query=goal,
            domain=inherited.get("domain"),
            limit=5,
        )
        
        return {
            "goal": goal,
            "parent_goal": continue_context.parent_lu.learning_goal,
            "parent_knowledge": inherited,
            "parent_constraints": continue_context.inherited_constraints,
            "chain_depth": continue_context.chain_depth,
            "exploration_direction": continue_context.exploration_direction,
            "focus_areas": continue_context.focus_areas,
            "related_production_knowledge": [
                {
                    "condition": r.knowledge.condition,
                    "decision": r.knowledge.decision,
                    "confidence": r.knowledge.confidence,
                    "relevance": r.relevance_score,
                }
                for r in related_knowledge
            ],
            # 指示 LLM 这是继续学习
            "learning_mode": "continuation",
            "instruction": f"""
你正在进行链式学习，基于已有的知识继续探索。

父学习单元信息：
- 目标: {continue_context.parent_lu.learning_goal}
- 领域: {inherited.get('domain', 'unknown')}
- 已发现的知识: {inherited.get('content', {})}

当前学习目标: {goal}
探索方向: {continue_context.exploration_direction or '深入探索'}
重点关注: {', '.join(continue_context.focus_areas) if continue_context.focus_areas else '无特定限制'}

请基于已有知识，进一步探索和发现新的知识。
""",
        }
    
    def _execute_nl_learning_with_context(
        self,
        goal: str,
        context: Dict[str, Any],
    ) -> List[ContextFlowSegment]:
        """执行带上下文的 NL 学习"""
        segments = []
        
        # 执行多个学习步骤
        for step in range(3):  # 默认 3 步
            step_context = {
                **context,
                "step": step,
                "previous_segments": [s.to_dict() for s in segments],
            }
            
            try:
                segment = self.nl_kernel.execute_learning_step(
                    context=step_context,
                    scope=self.current_scope,
                )
                segments.append(segment)
            except Exception as e:
                logger.warning(f"NL learning step {step} failed: {e}")
        
        return segments
    
    def _generate_knowledge_with_inheritance(
        self,
        goal: str,
        segments: List[ContextFlowSegment],
        continue_context: ContinueLearningContext,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """生成知识（带继承）"""
        # 从 segments 提取学习结果
        learning_results = []
        for seg in segments:
            if seg.level_deltas:
                for level, delta in seg.level_deltas.items():
                    learning_results.append({
                        "level": level.name,
                        "delta_type": delta.delta_type,
                        "content": delta.delta_content,
                    })
        
        # 继承父 LU 的领域
        parent_knowledge = continue_context.inherited_knowledge or {}
        domain = parent_knowledge.get("domain", "general")
        
        # 构建知识
        knowledge = {
            "domain": domain,
            "type": "knowledge",
            "content": {
                "goal": goal,
                "inherited_from": continue_context.parent_lu.id,
                "learning_results": learning_results,
                "key_findings": [
                    f"基于 '{continue_context.parent_lu.title}' 继续学习",
                    f"探索方向: {continue_context.exploration_direction or '深入探索'}",
                ],
                "explored_aspects": continue_context.focus_areas,
            },
            "confidence": 0.7,  # 继续学习的默认置信度
            "rationale": f"Chain learning from {continue_context.parent_lu.id}",
        }
        
        # 继承并扩展约束
        constraints = []
        for c in continue_context.inherited_constraints:
            # 标记为继承的约束
            inherited_constraint = {
                **c,
                "inherited": True,
                "inherited_from": continue_context.parent_lu.id,
            }
            constraints.append(inherited_constraint)
        
        # 添加新的约束（基于学习结果）
        if learning_results:
            new_constraint = {
                "condition": f"当处理与 '{goal}' 相关的情况时",
                "decision": f"应用链式学习发现的知识",
                "confidence": 0.6,
                "source_type": "chain_learning",
                "inherited": False,
            }
            constraints.append(new_constraint)
        
        return knowledge, constraints
    
    def _update_learning_chain(
        self,
        unit: ChainableLearningUnit,
        continue_context: ContinueLearningContext,
    ):
        """更新学习链"""
        chain_root_id = continue_context.chain_root_id
        
        if chain_root_id in self.learning_chains:
            # 更新现有链
            chain = self.learning_chains[chain_root_id]
            chain.head_lu_id = unit.id
            chain.total_depth = max(chain.total_depth, continue_context.chain_depth)
            chain.total_units += 1
            chain.current_goal = unit.learning_goal
            chain.updated_at = datetime.now()
        else:
            # 创建新链
            chain = LearningChain(
                chain_id=f"chain_{uuid.uuid4().hex[:8]}",
                root_lu_id=chain_root_id,
                head_lu_id=unit.id,
                total_depth=continue_context.chain_depth,
                total_units=2,  # 父 + 当前
                initial_goal=continue_context.parent_lu.learning_goal,
                current_goal=unit.learning_goal,
            )
            self.learning_chains[chain_root_id] = chain
    
    # ==================== 从零开始学习 ====================
    
    def build_from_scratch(
        self,
        goal: str,
        domain: Optional[str] = None,
        scope: Optional[LearningScope] = None,
    ) -> Optional[ChainableLearningUnit]:
        """
        从零开始构建 Learning Unit
        
        当没有合适的继续学习起点时使用。
        
        Args:
            goal: 学习目标
            domain: 领域
            scope: 学习范围
            
        Returns:
            新构建的 Learning Unit
        """
        print(f"\n{'='*60}")
        print(f"[ChainBuilder] 从零开始学习")
        print(f"  目标: {goal}")
        print(f"  领域: {domain or 'general'}")
        print(f"{'='*60}")
        
        # 设置 Scope
        if scope:
            self.current_scope = scope
        else:
            self.current_scope = self._create_default_scope()
        
        # 解冻 NL 内核
        self.nl_kernel.unfreeze()
        
        try:
            # 1. 搜索相关的现有知识作为参考
            related_knowledge = self.knowledge_reader.search_knowledge(
                query=goal,
                domain=domain,
                limit=5,
            )
            
            # 2. 构建学习上下文
            context = {
                "goal": goal,
                "domain": domain or "general",
                "learning_mode": "from_scratch",
                "related_knowledge": [
                    {
                        "condition": r.knowledge.condition,
                        "decision": r.knowledge.decision,
                        "confidence": r.knowledge.confidence,
                    }
                    for r in related_knowledge
                ],
                "instruction": f"""
你正在进行全新的学习探索。

学习目标: {goal}
领域: {domain or 'general'}

相关的现有知识（仅供参考）:
{chr(10).join([f"- {r.knowledge.condition}: {r.knowledge.decision}" for r in related_knowledge]) if related_knowledge else "无相关知识"}

请进行探索并发现新的知识。
""",
            }
            
            # 3. 执行 NL 学习
            segments = self._execute_nl_learning_with_context(goal, context)
            
            # 4. 生成知识
            knowledge = {
                "domain": domain or "general",
                "type": "knowledge",
                "content": {
                    "goal": goal,
                    "key_findings": [],
                    "explored_aspects": [],
                },
                "confidence": 0.5,
                "rationale": "New learning from scratch",
            }
            
            constraints = []
            
            # 5. 构建 Learning Unit
            unit_id = f"lu_{uuid.uuid4().hex[:12]}"
            
            unit = ChainableLearningUnit(
                id=unit_id,
                title=f"新学习: {goal[:50]}",
                learning_goal=goal,
                knowledge=knowledge,
                constraints=constraints,
                provenance={
                    "learning_goal": goal,
                    "domain": domain,
                    "segments_count": len(segments),
                    "related_knowledge_count": len(related_knowledge),
                },
                parent_lu_id=None,
                chain_depth=0,
                chain_root_id=unit_id,  # 自己是根
                status="pending",
                metadata={
                    "builder": "chainable_learning_builder",
                    "learning_mode": "from_scratch",
                    "scope_id": self.current_scope.scope_id if self.current_scope else None,
                },
                signals=[
                    {
                        "signal_type": "new_learning",
                        "content": {
                            "domain": domain,
                            "related_knowledge_count": len(related_knowledge),
                        }
                    },
                ],
            )
            
            # 6. 创建学习链
            chain = LearningChain(
                chain_id=f"chain_{uuid.uuid4().hex[:8]}",
                root_lu_id=unit_id,
                head_lu_id=unit_id,
                total_depth=0,
                total_units=1,
                initial_goal=goal,
                current_goal=goal,
            )
            self.learning_chains[unit_id] = chain
            
            # 7. 记录
            self.built_units.append(unit)
            
            # 8. 提交
            if self.submit_callback:
                print(f"\n[ChainBuilder] 提交 Learning Unit 到审计系统: {unit.id}")
                self.submit_callback(unit)
            
            self._print_build_summary(unit)
            
            return unit
            
        except Exception as e:
            logger.error(f"Build from scratch failed: {e}")
            print(f"  [Error] 从零开始学习失败: {e}")
            return None
        finally:
            self.nl_kernel.freeze()
    
    # ==================== 智能学习 ====================
    
    def smart_learn(
        self,
        goal: str,
        domain: Optional[str] = None,
        prefer_continuation: bool = True,
        scope: Optional[LearningScope] = None,
    ) -> Optional[ChainableLearningUnit]:
        """
        智能学习
        
        自动选择最佳的学习方式：
        - 如果有合适的继续学习起点，则继续学习
        - 否则从零开始
        
        Args:
            goal: 学习目标
            domain: 领域
            prefer_continuation: 是否优先继续学习
            scope: 学习范围
            
        Returns:
            新构建的 Learning Unit
        """
        print(f"\n{'='*60}")
        print(f"[ChainBuilder] 智能学习")
        print(f"  目标: {goal}")
        print(f"  优先继续学习: {prefer_continuation}")
        print(f"{'='*60}")
        
        if prefer_continuation:
            # 尝试选择学习起点
            parent_lu, selection_reason = self.select_learning_starting_point(
                goal=goal,
                domain=domain,
                strategy="relevance",
            )
            
            if parent_lu:
                print(f"  [ChainBuilder] 选择继续学习模式")
                return self.continue_learning(
                    parent_lu_id=parent_lu.id,
                    new_goal=goal,
                    scope=scope,
                )
        
        # 从零开始
        print(f"  [ChainBuilder] 选择从零开始模式")
        return self.build_from_scratch(
            goal=goal,
            domain=domain,
            scope=scope,
        )
    
    # ==================== 辅助方法 ====================
    
    def _create_default_scope(self) -> LearningScope:
        """创建默认 Scope"""
        return LearningScope(
            scope_id=f"scope_{uuid.uuid4().hex[:8]}",
            max_level=NLLevel.MEMORY,
            allowed_levels=[NLLevel.PARAMETER, NLLevel.MEMORY],
            created_by="chainable_builder",
        )
    
    def _print_build_summary(self, unit: ChainableLearningUnit):
        """打印构建摘要"""
        print(f"\n{'='*60}")
        print(f"[ChainBuilder] Learning Unit 构建完成")
        print(f"  ID: {unit.id}")
        print(f"  标题: {unit.title}")
        print(f"  目标: {unit.learning_goal}")
        print(f"  链深度: {unit.chain_depth}")
        print(f"  父 LU: {unit.parent_lu_id or 'None (root)'}")
        print(f"  链根: {unit.chain_root_id}")
        print(f"  约束数: {len(unit.constraints)}")
        print(f"  关系数: {len(unit.relations)}")
        print(f"  状态: {unit.status}")
        print(f"{'='*60}")
    
    # ==================== 查询方法 ====================
    
    def get_learning_chain(self, chain_root_id: str) -> Optional[LearningChain]:
        """获取学习链"""
        return self.learning_chains.get(chain_root_id)
    
    def get_lu_relations(self, lu_id: str) -> List[LURelation]:
        """获取 LU 的所有关系"""
        return [
            r for r in self.lu_relations
            if r.source_lu_id == lu_id or r.target_lu_id == lu_id
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_built": len(self.built_units),
            "total_chains": len(self.learning_chains),
            "total_relations": len(self.lu_relations),
            "by_chain_depth": self._count_by_chain_depth(),
            "by_status": self._count_by_status(),
        }
    
    def _count_by_chain_depth(self) -> Dict[int, int]:
        """按链深度统计"""
        counts = {}
        for unit in self.built_units:
            depth = unit.chain_depth
            counts[depth] = counts.get(depth, 0) + 1
        return counts
    
    def _count_by_status(self) -> Dict[str, int]:
        """按状态统计"""
        counts = {}
        for unit in self.built_units:
            status = unit.status
            counts[status] = counts.get(status, 0) + 1
        return counts

