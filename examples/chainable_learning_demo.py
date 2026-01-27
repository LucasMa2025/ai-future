"""
链式学习演示

演示如何使用 ChainableLearningUnitBuilder 进行链式学习：
1. 从零开始学习
2. 在已有知识基础上继续学习
3. 智能选择学习起点
4. 知识图谱追踪

运行方式：
    python examples/chainable_learning_demo.py
"""
import sys
import os
from datetime import datetime
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_learning import (
    ChainableLearningUnitBuilder,
    ChainableLearningUnit,
    LearningChain,
    ProductionKnowledge,
    ApprovedLearningUnit,
    KnowledgeType,
    InMemoryProductionKnowledgeReader,
    InMemoryApprovedLUReader,
    NLLevel,
    LearningScope,
)


def create_mock_data():
    """创建模拟数据"""
    # 创建知识读取器
    knowledge_reader = InMemoryProductionKnowledgeReader()
    lu_reader = InMemoryApprovedLUReader()
    
    # 添加一些模拟的生产知识
    knowledge_reader.add_knowledge(ProductionKnowledge(
        id="pk_001",
        domain="financial",
        knowledge_type=KnowledgeType.DECISION_RULE,
        condition="当用户请求大额转账时",
        decision="需要进行额外的身份验证",
        confidence=0.9,
        risk_level="high",
        source_lu_id="lu_base_001",
        source_lu_title="大额交易安全规则",
        source_goal="学习金融交易安全规则",
        hit_count=150,
    ))
    
    knowledge_reader.add_knowledge(ProductionKnowledge(
        id="pk_002",
        domain="financial",
        knowledge_type=KnowledgeType.KNOWLEDGE,
        condition="当检测到异常交易模式时",
        decision="触发风险预警并暂停交易",
        confidence=0.85,
        risk_level="high",
        source_lu_id="lu_base_001",
        source_lu_title="大额交易安全规则",
        source_goal="学习金融交易安全规则",
        hit_count=80,
    ))
    
    knowledge_reader.add_knowledge(ProductionKnowledge(
        id="pk_003",
        domain="customer_service",
        knowledge_type=KnowledgeType.PATTERN,
        condition="当用户表达不满情绪时",
        decision="优先转接人工客服并记录问题",
        confidence=0.8,
        risk_level="medium",
        source_lu_id="lu_base_002",
        source_lu_title="客户情绪识别规则",
        source_goal="学习客户服务最佳实践",
        hit_count=200,
    ))
    
    # 添加一些已审批的 Learning Unit
    lu_reader.add_lu(ApprovedLearningUnit(
        id="lu_base_001",
        title="大额交易安全规则",
        learning_goal="学习金融交易安全规则",
        status="approved",
        knowledge={
            "domain": "financial",
            "type": "decision_rule",
            "content": {
                "key_findings": [
                    "大额交易需要多重验证",
                    "异常模式需要实时监控",
                ],
                "explored_aspects": ["身份验证", "风险评估", "交易监控"],
            },
            "confidence": 0.9,
        },
        constraints=[
            {
                "condition": "大额转账",
                "decision": "额外身份验证",
                "confidence": 0.9,
            },
            {
                "condition": "异常交易模式",
                "decision": "触发风险预警",
                "confidence": 0.85,
            },
        ],
        provenance={
            "learning_goal": "学习金融交易安全规则",
            "exploration_steps": 5,
        },
        risk_level="high",
        risk_score=75.0,
        approved_by="admin",
        approved_at=datetime.now(),
        is_internalized=True,
        lifecycle_state="confirmed",
        chain_depth=0,
        chain_root_id="lu_base_001",
        tags=["financial", "security", "transaction"],
    ))
    
    lu_reader.add_lu(ApprovedLearningUnit(
        id="lu_base_002",
        title="客户情绪识别规则",
        learning_goal="学习客户服务最佳实践",
        status="approved",
        knowledge={
            "domain": "customer_service",
            "type": "pattern",
            "content": {
                "key_findings": [
                    "负面情绪需要及时响应",
                    "人工介入可以提高满意度",
                ],
                "explored_aspects": ["情绪识别", "响应策略", "满意度提升"],
            },
            "confidence": 0.8,
        },
        constraints=[
            {
                "condition": "用户表达不满",
                "decision": "转接人工客服",
                "confidence": 0.8,
            },
        ],
        provenance={
            "learning_goal": "学习客户服务最佳实践",
            "exploration_steps": 4,
        },
        risk_level="medium",
        risk_score=45.0,
        approved_by="admin",
        approved_at=datetime.now(),
        is_internalized=True,
        lifecycle_state="confirmed",
        chain_depth=0,
        chain_root_id="lu_base_002",
        tags=["customer_service", "emotion", "support"],
    ))
    
    return knowledge_reader, lu_reader


def demo_from_scratch():
    """演示从零开始学习"""
    print("\n" + "="*80)
    print("演示 1: 从零开始学习")
    print("="*80)
    
    knowledge_reader, lu_reader = create_mock_data()
    
    # 创建构建器
    builder = ChainableLearningUnitBuilder(
        knowledge_reader=knowledge_reader,
        lu_reader=lu_reader,
        max_chain_depth=5,
    )
    
    # 从零开始学习
    unit = builder.build_from_scratch(
        goal="学习如何处理用户投诉",
        domain="customer_service",
    )
    
    if unit:
        print("\n构建结果:")
        print(f"  ID: {unit.id}")
        print(f"  标题: {unit.title}")
        print(f"  链深度: {unit.chain_depth}")
        print(f"  链根: {unit.chain_root_id}")
        
        # 查看学习链
        chain = builder.get_learning_chain(unit.chain_root_id)
        if chain:
            print(f"\n学习链信息:")
            print(f"  链 ID: {chain.chain_id}")
            print(f"  总深度: {chain.total_depth}")
            print(f"  总单元数: {chain.total_units}")
    
    return builder, unit


def demo_continue_learning(builder: ChainableLearningUnitBuilder, parent_lu_id: str):
    """演示继续学习"""
    print("\n" + "="*80)
    print("演示 2: 继续学习")
    print("="*80)
    
    # 在已有 LU 基础上继续学习
    unit = builder.continue_learning(
        parent_lu_id=parent_lu_id,
        new_goal="深入学习大额交易的风险评估方法",
        exploration_direction="风险评估",
        focus_areas=["风险模型", "阈值设定", "实时监控"],
    )
    
    if unit:
        print("\n构建结果:")
        print(f"  ID: {unit.id}")
        print(f"  标题: {unit.title}")
        print(f"  父 LU: {unit.parent_lu_id}")
        print(f"  链深度: {unit.chain_depth}")
        print(f"  链根: {unit.chain_root_id}")
        
        # 查看关系
        relations = builder.get_lu_relations(unit.id)
        if relations:
            print(f"\n关系:")
            for r in relations:
                print(f"  - {r.relation_type.value}: {r.source_lu_id} -> {r.target_lu_id}")
    
    return unit


def demo_smart_learning():
    """演示智能学习"""
    print("\n" + "="*80)
    print("演示 3: 智能学习")
    print("="*80)
    
    knowledge_reader, lu_reader = create_mock_data()
    
    # 创建构建器
    builder = ChainableLearningUnitBuilder(
        knowledge_reader=knowledge_reader,
        lu_reader=lu_reader,
        max_chain_depth=5,
    )
    
    # 智能学习 - 会自动选择最佳方式
    unit = builder.smart_learn(
        goal="学习金融交易中的欺诈检测",
        domain="financial",
        prefer_continuation=True,
    )
    
    if unit:
        print("\n构建结果:")
        print(f"  ID: {unit.id}")
        print(f"  标题: {unit.title}")
        print(f"  父 LU: {unit.parent_lu_id or 'None (从零开始)'}")
        print(f"  链深度: {unit.chain_depth}")
    
    return builder, unit


def demo_learning_chain():
    """演示学习链追踪"""
    print("\n" + "="*80)
    print("演示 4: 学习链追踪")
    print("="*80)
    
    knowledge_reader, lu_reader = create_mock_data()
    
    # 创建构建器
    builder = ChainableLearningUnitBuilder(
        knowledge_reader=knowledge_reader,
        lu_reader=lu_reader,
        max_chain_depth=5,
    )
    
    # 构建一个学习链
    print("\n步骤 1: 从零开始学习")
    unit1 = builder.build_from_scratch(
        goal="学习基础的风险管理概念",
        domain="financial",
    )
    
    if not unit1:
        print("构建失败")
        return
    
    # 模拟审批通过
    # 在实际系统中，这会由审计系统完成
    approved_lu = ApprovedLearningUnit(
        id=unit1.id,
        title=unit1.title,
        learning_goal=unit1.learning_goal,
        status="approved",
        knowledge=unit1.knowledge,
        constraints=unit1.constraints,
        provenance=unit1.provenance,
        risk_level="medium",
        risk_score=50.0,
        approved_by="admin",
        approved_at=datetime.now(),
        is_internalized=True,
        lifecycle_state="confirmed",
        chain_depth=unit1.chain_depth,
        chain_root_id=unit1.chain_root_id,
    )
    lu_reader.add_lu(approved_lu)
    
    print("\n步骤 2: 继续学习 (第一次)")
    unit2 = builder.continue_learning(
        parent_lu_id=unit1.id,
        new_goal="深入学习信用风险评估",
        exploration_direction="信用风险",
    )
    
    if unit2:
        # 模拟审批
        approved_lu2 = ApprovedLearningUnit(
            id=unit2.id,
            title=unit2.title,
            learning_goal=unit2.learning_goal,
            status="approved",
            knowledge=unit2.knowledge,
            constraints=unit2.constraints,
            provenance=unit2.provenance,
            risk_level="medium",
            risk_score=55.0,
            approved_by="admin",
            approved_at=datetime.now(),
            is_internalized=True,
            lifecycle_state="confirmed",
            chain_depth=unit2.chain_depth,
            chain_root_id=unit2.chain_root_id,
            parent_lu_id=unit2.parent_lu_id,
        )
        lu_reader.add_lu(approved_lu2)
        
        print("\n步骤 3: 继续学习 (第二次)")
        unit3 = builder.continue_learning(
            parent_lu_id=unit2.id,
            new_goal="学习信用评分模型的构建",
            exploration_direction="评分模型",
        )
    
    # 打印学习链信息
    print("\n" + "-"*40)
    print("学习链统计:")
    stats = builder.get_statistics()
    print(f"  总构建数: {stats['total_built']}")
    print(f"  总链数: {stats['total_chains']}")
    print(f"  总关系数: {stats['total_relations']}")
    print(f"  按链深度: {stats['by_chain_depth']}")
    
    # 打印所有学习链
    print("\n学习链详情:")
    for chain_id, chain in builder.learning_chains.items():
        print(f"\n  链 ID: {chain.chain_id}")
        print(f"    根 LU: {chain.root_lu_id}")
        print(f"    头 LU: {chain.head_lu_id}")
        print(f"    总深度: {chain.total_depth}")
        print(f"    总单元数: {chain.total_units}")
        print(f"    初始目标: {chain.initial_goal}")
        print(f"    当前目标: {chain.current_goal}")


def demo_knowledge_query():
    """演示知识查询"""
    print("\n" + "="*80)
    print("演示 5: 知识查询")
    print("="*80)
    
    knowledge_reader, lu_reader = create_mock_data()
    
    # 搜索知识
    print("\n搜索 '交易' 相关知识:")
    results = knowledge_reader.search_knowledge(
        query="交易",
        domain="financial",
        limit=5,
    )
    
    for r in results:
        print(f"\n  知识 ID: {r.knowledge.id}")
        print(f"    条件: {r.knowledge.condition}")
        print(f"    决策: {r.knowledge.decision}")
        print(f"    置信度: {r.knowledge.confidence}")
        print(f"    相关性: {r.relevance_score:.2f}")
        print(f"    来源 LU: {r.knowledge.source_lu_id}")
    
    # 获取可继续学习的 LU
    print("\n可继续学习的 LU:")
    continuable = lu_reader.get_continuable_lus(max_chain_depth=5, limit=10)
    
    for lu in continuable:
        print(f"\n  LU ID: {lu.id}")
        print(f"    标题: {lu.title}")
        print(f"    目标: {lu.learning_goal}")
        print(f"    链深度: {lu.chain_depth}")
        print(f"    已内化: {lu.is_internalized}")
    
    # 获取知识库统计
    print("\n知识库统计:")
    stats = knowledge_reader.get_statistics()
    print(f"  总知识数: {stats.get('total_knowledge', 0)}")
    print(f"  活跃知识数: {stats.get('active_knowledge', 0)}")
    print(f"  按领域: {stats.get('by_domain', {})}")
    print(f"  按类型: {stats.get('by_type', {})}")


def main():
    """主函数"""
    print("="*80)
    print("链式学习演示")
    print("="*80)
    print("""
本演示展示了链式学习的核心功能：

1. 从零开始学习 - 当没有相关知识时
2. 继续学习 - 在已审批的 LU 基础上深入探索
3. 智能学习 - 自动选择最佳学习方式
4. 学习链追踪 - 追踪知识的演进路径
5. 知识查询 - 查询生产库中的已有知识

核心设计原则：
- 学习的起点是 LLM 的现有知识库
- 链式学习必须基于已审批的 LU
- 所有关联关系都被记录用于审计
- 链深度受治理系统控制
""")
    
    # 运行演示
    demo_knowledge_query()
    
    builder, unit = demo_from_scratch()
    
    if unit:
        # 模拟审批后继续学习
        # 注意：在实际系统中，需要等待审计系统审批
        print("\n[注意] 在实际系统中，需要等待审计系统审批后才能继续学习")
    
    demo_smart_learning()
    
    demo_learning_chain()
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


if __name__ == "__main__":
    main()

