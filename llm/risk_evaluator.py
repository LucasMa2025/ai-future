"""
LLM 风险评估器

利用 LLM 进行 Learning Unit 的风险预评估
"""
from typing import Dict, Any, Optional
import json

from core.types import LearningUnit, RiskAssessment
from core.enums import RiskLevel
from .client import DeepSeekClient, MockDeepSeekClient
from .prompts import PromptTemplates


class LLMRiskEvaluator:
    """
    LLM 风险评估器
    
    特性：
    - 利用 LLM 进行风险预评估
    - 支持二次评估
    - 输出结构化的风险评估结果
    """
    
    def __init__(
        self,
        llm_client: Optional[DeepSeekClient] = None,
        risk_thresholds: Optional[Dict[str, float]] = None
    ):
        self.llm = llm_client or MockDeepSeekClient()
        self.risk_thresholds = risk_thresholds or {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
        }
    
    def assess(self, unit: LearningUnit) -> RiskAssessment:
        """
        评估 Learning Unit 的风险
        
        Args:
            unit: Learning Unit
            
        Returns:
            风险评估结果
        """
        # 准备评估数据
        exploration_path = []
        if unit.provenance:
            exploration_path = [s.to_dict() for s in unit.provenance.exploration_path]
        
        knowledge_content = {}
        if unit.knowledge:
            knowledge_content = unit.knowledge.content
        
        # 构建 prompt
        messages = [
            {"role": "system", "content": PromptTemplates.RISK_ASSESSMENT_SYSTEM},
            {"role": "user", "content": PromptTemplates.format_risk_assessment(
                unit_id=unit.id,
                domain=unit.knowledge.domain if unit.knowledge else "unknown",
                type_=unit.knowledge.type.value if unit.knowledge else "unknown",
                content=knowledge_content,
                confidence=unit.knowledge.confidence if unit.knowledge else 0.5,
                rationale=unit.knowledge.rationale if unit.knowledge else "",
                exploration_path=exploration_path
            )}
        ]
        
        # 调用 LLM
        try:
            response = self.llm.chat_json(messages, temperature=0.3)
        except Exception as e:
            # LLM 调用失败，返回 uncertain
            return RiskAssessment(
                risk_level=RiskLevel.UNCERTAIN,
                confidence=0.0,
                factors=[{"factor": "LLM 评估失败", "assessment": str(e), "score": 0.0}],
                recommendation="需要人工审核（LLM 评估失败）"
            )
        
        # 解析响应
        risk_level_str = response.get('risk_level', 'uncertain')
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.UNCERTAIN
        
        return RiskAssessment(
            risk_level=risk_level,
            confidence=response.get('confidence', 0.5),
            factors=response.get('factors', []),
            recommendation=response.get('recommendation', '需要人工审核')
        )
    
    def secondary_assess(
        self,
        unit: LearningUnit,
        initial_assessment: RiskAssessment
    ) -> RiskAssessment:
        """
        二次评估（用于中等风险的 Learning Unit）
        
        Args:
            unit: Learning Unit
            initial_assessment: 初次评估结果
            
        Returns:
            二次评估结果
        """
        messages = [
            {"role": "system", "content": PromptTemplates.SECONDARY_ASSESSMENT_SYSTEM},
            {"role": "user", "content": PromptTemplates.SECONDARY_ASSESSMENT_QUERY.format(
                unit_json=json.dumps(unit.to_dict(), ensure_ascii=False, indent=2),
                initial_risk_level=initial_assessment.risk_level.value,
                initial_factors=json.dumps(initial_assessment.factors, ensure_ascii=False)
            )}
        ]
        
        try:
            response = self.llm.chat_json(messages, temperature=0.3)
        except Exception as e:
            return RiskAssessment(
                risk_level=RiskLevel.UNCERTAIN,
                confidence=0.0,
                factors=[{"factor": "二次评估失败", "assessment": str(e), "score": 0.0}],
                recommendation="需要人工审核（二次评估失败）"
            )
        
        risk_level_str = response.get('final_risk_level', 'uncertain')
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.UNCERTAIN
        
        # 合并因素
        factors = initial_assessment.factors.copy()
        factors.append({
            "factor": "二次评估",
            "assessment": response.get('final_recommendation', ''),
            "score": response.get('confidence', 0.5),
            "is_harmful": response.get('is_harmful', False),
            "harm_analysis": response.get('harm_analysis', ''),
        })
        
        return RiskAssessment(
            risk_level=risk_level,
            confidence=response.get('confidence', 0.5),
            factors=factors,
            recommendation=response.get('final_recommendation', '需要人工审核')
        )
    
    def is_low_risk_and_harmless(self, assessment: RiskAssessment) -> bool:
        """
        判断是否为低风险且无害
        
        注意：第一阶段所有都走人类审核，此方法仅作为参考
        """
        if assessment.risk_level != RiskLevel.LOW:
            return False
        
        if assessment.confidence < 0.8:
            return False
        
        # 检查是否有危害因素
        for factor in assessment.factors:
            if factor.get('is_harmful', False):
                return False
            if factor.get('score', 1.0) < 0.5:
                return False
        
        return True
    
    def get_classification(self, assessment: RiskAssessment) -> str:
        """
        获取分类结果
        
        Returns:
            "auto_approve" | "secondary_review" | "human_review"
        """
        # 第一阶段：所有都走人类审核
        # 但仍然返回分类结果作为参考
        
        if assessment.risk_level == RiskLevel.LOW:
            if self.is_low_risk_and_harmless(assessment):
                return "auto_approve"  # 参考值，实际仍走人类审核
            return "secondary_review"
        
        if assessment.risk_level == RiskLevel.MEDIUM:
            return "secondary_review"
        
        # HIGH 或 UNCERTAIN
        return "human_review"

