"""
内化引擎

将 Learning Unit 的约束内化到 Decision Head
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.types import LearningUnit, ProposedConstraint
from core.enums import DecisionType
from core.exceptions import InternalizationFailed


class DecisionHead(nn.Module):
    """
    Decision Head
    
    将隐藏状态映射到决策空间
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_decisions: int = 5,
        intermediate_dim: int = 128
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_decisions = num_decisions
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, num_decisions)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Linear(intermediate_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        
        decision_logits = self.projection(hidden_states)
        decision_probs = F.softmax(decision_logits, dim=-1)
        confidence = self.confidence_head(hidden_states)
        
        return {
            'decision_logits': decision_logits,
            'decision_probs': decision_probs,
            'confidence': confidence,
        }


class InternalizationEngine:
    """
    内化引擎
    
    特性：
    - 将 Learning Unit 的约束内化到 Decision Head
    - 支持增量更新
    - 版本管理
    """
    
    def __init__(
        self,
        decision_head: Optional[DecisionHead] = None,
        hidden_dim: int = 256,
        learning_rate: float = 0.0001,
        epochs: int = 10,
        batch_size: int = 32
    ):
        self.decision_head = decision_head or DecisionHead(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.optimizer = torch.optim.AdamW(
            self.decision_head.parameters(),
            lr=learning_rate
        )
        
        # 已内化的约束
        self.internalized_constraints: List[Dict[str, Any]] = []
        
        # 版本历史
        self.version_history: List[Dict[str, Any]] = []
        self.current_version = 0
    
    def internalize(self, unit: LearningUnit) -> Dict[str, Any]:
        """
        内化 Learning Unit
        
        Args:
            unit: Learning Unit
            
        Returns:
            内化结果
        """
        print(f"\n[内化引擎] 开始内化: {unit.id}")
        
        if not unit.proposed_constraints:
            print(f"  无约束需要内化")
            return {
                'status': 'skipped',
                'reason': 'No constraints to internalize',
            }
        
        # 生成训练样本
        samples = self._generate_samples(unit.proposed_constraints)
        
        if samples['input_features'].size(0) == 0:
            return {
                'status': 'skipped',
                'reason': 'No samples generated',
            }
        
        print(f"  生成样本: {samples['input_features'].size(0)} 个")
        
        # 训练
        self.decision_head.train()
        
        for epoch in range(self.epochs):
            epoch_loss = self._train_epoch(samples)
            
            if epoch % 3 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}: loss={epoch_loss:.4f}")
        
        # 验证
        validation_result = self._validate(samples)
        
        # 保存版本
        self.current_version += 1
        self.version_history.append({
            'version': self.current_version,
            'unit_id': unit.id,
            'constraints_count': len(unit.proposed_constraints),
            'validation_accuracy': validation_result['accuracy'],
            'timestamp': datetime.now().isoformat(),
        })
        
        # 记录已内化的约束
        for constraint in unit.proposed_constraints:
            self.internalized_constraints.append({
                'constraint_id': constraint.constraint_id,
                'unit_id': unit.id,
                'version': self.current_version,
            })
        
        print(f"  内化完成: 准确率 {validation_result['accuracy']:.2%}")
        
        return {
            'status': 'success',
            'version': self.current_version,
            'validation': validation_result,
        }
    
    def _generate_samples(
        self,
        constraints: List[ProposedConstraint]
    ) -> Dict[str, torch.Tensor]:
        """生成训练样本"""
        input_features = []
        target_decisions = []
        
        decision_to_idx = {d: i for i, d in enumerate(DecisionType)}
        
        for constraint in constraints:
            # 为每个约束生成多个样本
            for _ in range(10):
                features = torch.randn(self.hidden_dim)
                target = decision_to_idx[constraint.proposed_decision]
                
                input_features.append(features)
                target_decisions.append(target)
        
        if not input_features:
            return {
                'input_features': torch.empty(0, self.hidden_dim),
                'target_decisions': torch.empty(0, dtype=torch.long),
            }
        
        return {
            'input_features': torch.stack(input_features),
            'target_decisions': torch.tensor(target_decisions),
        }
    
    def _train_epoch(self, samples: Dict[str, torch.Tensor]) -> float:
        """训练一个 epoch"""
        total_loss = 0.0
        num_batches = 0
        
        num_samples = samples['input_features'].size(0)
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            
            input_features = samples['input_features'][batch_indices]
            targets = samples['target_decisions'][batch_indices]
            
            output = self.decision_head(input_features)
            loss = F.cross_entropy(output['decision_logits'], targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _validate(self, samples: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """验证内化结果"""
        self.decision_head.eval()
        
        with torch.no_grad():
            output = self.decision_head(samples['input_features'])
            predictions = torch.argmax(output['decision_probs'], dim=-1)
            
            correct = (predictions == samples['target_decisions']).float()
            accuracy = correct.mean().item()
        
        return {
            'accuracy': accuracy,
            'total_samples': samples['input_features'].size(0),
        }
    
    def get_decision(self, hidden_states: torch.Tensor) -> Dict[str, Any]:
        """获取决策"""
        self.decision_head.eval()
        
        with torch.no_grad():
            output = self.decision_head(hidden_states)
            decision_idx = torch.argmax(output['decision_probs'], dim=-1).item()
            confidence = output['confidence'].item()
        
        decision = list(DecisionType)[decision_idx]
        
        return {
            'decision': decision.name,
            'confidence': confidence,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_version': self.current_version,
            'internalized_constraints': len(self.internalized_constraints),
            'version_history': self.version_history[-10:],
        }

