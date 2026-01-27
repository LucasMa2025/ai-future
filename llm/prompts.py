"""
Prompt 模板

用于自学习和风险评估
"""


class PromptTemplates:
    """Prompt 模板集合"""
    
    # ============================================================
    # 自学习探索 Prompts
    # ============================================================
    
    EXPLORATION_SYSTEM = """你是一个自主学习系统的探索引擎。你的任务是基于给定的学习目标，进行知识探索和推理。

你需要：
1. 分析学习目标
2. 检索相关知识
3. 进行推理和归纳
4. 生成新的知识或规则

请以结构化的方式输出你的发现。"""

    EXPLORATION_QUERY = """学习目标：{goal}

当前探索深度：{depth}
已有发现：{findings}

请继续探索，输出 JSON 格式：
{{
    "action": "knowledge_retrieval|reasoning|hypothesis|validation|synthesis",
    "query": "你的探索查询",
    "findings": [
        {{"type": "knowledge|pattern|rule", "content": "发现内容", "confidence": 0.0-1.0}}
    ],
    "next_steps": ["下一步建议"],
    "should_checkpoint": true/false,
    "checkpoint_reason": "如果需要 checkpoint，说明原因"
}}"""

    # ============================================================
    # 知识生成 Prompts
    # ============================================================
    
    KNOWLEDGE_GENERATION_SYSTEM = """你是一个知识生成器。基于探索发现，生成结构化的知识或决策规则。

生成的知识必须：
1. 有明确的条件和结论
2. 有合理的置信度
3. 有清晰的理由说明"""

    KNOWLEDGE_GENERATION_QUERY = """基于以下探索发现，生成 Learning Unit：

学习目标：{goal}
探索路径：{exploration_path}
发现内容：{findings}

请输出 JSON 格式：
{{
    "domain": "领域",
    "type": "decision_rule|knowledge|pattern|constraint",
    "content": {{
        "condition": "条件表达式",
        "action": "动作或结论",
        "parameters": {{}}
    }},
    "confidence": 0.0-1.0,
    "rationale": "理由说明",
    "proposed_constraints": [
        {{
            "constraint_id": "约束ID",
            "condition": "条件",
            "proposed_decision": "REJECT|APPROVE|REVIEW|CONDITIONAL_APPROVE|ESCALATE",
            "rationale": "理由"
        }}
    ]
}}"""

    # ============================================================
    # 风险评估 Prompts
    # ============================================================
    
    RISK_ASSESSMENT_SYSTEM = """你是一个风险评估专家。你需要评估 Learning Unit 的风险等级。

评估维度：
1. 知识来源可靠性 - 知识是否来自可靠的推理链路
2. 推理链路合理性 - 推理过程是否逻辑清晰
3. 潜在危害评估 - 如果应用这个知识，可能造成什么危害
4. 与现有知识一致性 - 是否与已知事实矛盾
5. 边界条件分析 - 是否考虑了边界情况

风险等级：
- low: 低风险，知识可靠，无潜在危害
- medium: 中风险，需要进一步验证
- high: 高风险，可能有危害或不可靠
- uncertain: 无法确定，需要人工判断"""

    RISK_ASSESSMENT_QUERY = """请评估以下 Learning Unit 的风险：

Learning Unit ID: {unit_id}
领域: {domain}
类型: {type}
内容: {content}
置信度: {confidence}
理由: {rationale}

探索路径:
{exploration_path}

请输出 JSON 格式：
{{
    "risk_level": "low|medium|high|uncertain",
    "confidence": 0.0-1.0,
    "factors": [
        {{
            "factor": "评估维度",
            "assessment": "评估结果",
            "score": 0.0-1.0,
            "details": "详细说明"
        }}
    ],
    "potential_harms": ["潜在危害列表"],
    "recommendation": "建议（自动通过/二次评估/人工审核）",
    "reasoning": "评估理由"
}}"""

    # ============================================================
    # 二次评估 Prompts
    # ============================================================
    
    SECONDARY_ASSESSMENT_SYSTEM = """你是一个深度风险评估专家。对于中等风险的 Learning Unit，你需要进行更深入的分析。

重点关注：
1. 边界条件是否完整
2. 是否有遗漏的风险因素
3. 应用场景是否合适
4. 是否需要额外的约束条件"""

    SECONDARY_ASSESSMENT_QUERY = """请对以下 Learning Unit 进行深度风险评估：

Learning Unit: {unit_json}

初次评估结果:
- 风险等级: {initial_risk_level}
- 评估因素: {initial_factors}

请进行更深入的分析，输出 JSON 格式：
{{
    "final_risk_level": "low|medium|high|uncertain",
    "is_harmful": true/false,
    "harm_analysis": "危害分析",
    "missing_constraints": ["遗漏的约束"],
    "recommended_corrections": ["建议的修正"],
    "final_recommendation": "自动通过|人工审核",
    "reasoning": "详细理由"
}}"""

    @classmethod
    def format_exploration(cls, goal: str, depth: int, findings: list) -> str:
        """格式化探索 prompt"""
        return cls.EXPLORATION_QUERY.format(
            goal=goal,
            depth=depth,
            findings=findings
        )
    
    @classmethod
    def format_knowledge_generation(
        cls, 
        goal: str, 
        exploration_path: list, 
        findings: list
    ) -> str:
        """格式化知识生成 prompt"""
        return cls.KNOWLEDGE_GENERATION_QUERY.format(
            goal=goal,
            exploration_path=exploration_path,
            findings=findings
        )
    
    @classmethod
    def format_risk_assessment(
        cls,
        unit_id: str,
        domain: str,
        type_: str,
        content: dict,
        confidence: float,
        rationale: str,
        exploration_path: list
    ) -> str:
        """格式化风险评估 prompt"""
        import json
        return cls.RISK_ASSESSMENT_QUERY.format(
            unit_id=unit_id,
            domain=domain,
            type=type_,
            content=json.dumps(content, ensure_ascii=False, indent=2),
            confidence=confidence,
            rationale=rationale,
            exploration_path=json.dumps(exploration_path, ensure_ascii=False, indent=2)
        )

