"""
权限验证器

验证写入权限和审计批准

支持多种输入格式：
1. core.types.AuditApproval (原始格式)
2. BridgeAuditApproval (后端服务格式)
3. 任何具有 approval_id, decision, verify() 的对象
"""
from typing import Set, Dict, Any, Union, Protocol, Optional
from enum import Enum

# 尝试导入 core 模块，如果失败则使用本地定义
try:
    from core.types import AuditApproval as CoreAuditApproval
    from core.enums import AuditDecision
    from core.exceptions import UnauthorizedWriter, InvalidAuditApproval
    HAS_CORE = True
except ImportError:
    HAS_CORE = False
    
    class AuditDecision(str, Enum):
        APPROVE = "approve"
        REJECT = "reject"
        CORRECT = "correct"
    
    class UnauthorizedWriter(Exception):
        """未授权的写入者"""
        pass
    
    class InvalidAuditApproval(Exception):
        """无效的审计批准"""
        pass


class AuditApprovalProtocol(Protocol):
    """审计批准协议"""
    approval_id: str
    decision: Union[str, AuditDecision]
    
    def verify(self) -> bool:
        ...


class PermissionValidator:
    """
    权限验证器
    
    特性：
    - 验证写入者权限
    - 验证审计批准证明
    - 记录验证历史
    """
    
    def __init__(self, authorized_writers: Set[str] = None):
        # 授权的写入者列表
        self.authorized_writers = authorized_writers or {"audit_system"}
        
        # 验证历史
        self.validation_history: list = []
    
    def validate_writer(self, writer_id: str) -> bool:
        """
        验证写入者权限
        
        Args:
            writer_id: 写入者 ID
            
        Returns:
            是否有权限
            
        Raises:
            UnauthorizedWriter: 如果没有权限
        """
        if writer_id not in self.authorized_writers:
            self.validation_history.append({
                'type': 'writer_validation',
                'writer_id': writer_id,
                'result': 'denied',
                'reason': 'Not in authorized writers list',
            })
            raise UnauthorizedWriter(
                f"Writer '{writer_id}' is not authorized to write to production bridge. "
                f"Only {self.authorized_writers} can write."
            )
        
        self.validation_history.append({
            'type': 'writer_validation',
            'writer_id': writer_id,
            'result': 'allowed',
        })
        
        return True
    
    def validate_approval(self, approval: Union[AuditApprovalProtocol, Any]) -> bool:
        """
        验证审计批准证明
        
        支持多种输入格式，自动适配。
        
        Args:
            approval: 审计批准证明（支持多种格式）
            
        Returns:
            是否有效
            
        Raises:
            InvalidAuditApproval: 如果无效
        """
        # 获取 approval_id（兼容不同格式）
        approval_id = getattr(approval, 'approval_id', str(approval))
        
        # 验证签名（如果有 verify 方法）
        if hasattr(approval, 'verify'):
            try:
                if not approval.verify():
                    self.validation_history.append({
                        'type': 'approval_validation',
                        'approval_id': approval_id,
                        'result': 'invalid',
                        'reason': 'Signature verification failed',
                    })
                    raise InvalidAuditApproval("Audit approval signature verification failed")
            except Exception as e:
                if isinstance(e, InvalidAuditApproval):
                    raise
                # verify() 方法可能抛出其他异常，视为验证失败
                self.validation_history.append({
                    'type': 'approval_validation',
                    'approval_id': approval_id,
                    'result': 'invalid',
                    'reason': f'Verification error: {e}',
                })
                raise InvalidAuditApproval(f"Audit approval verification error: {e}")
        
        # 获取决策（兼容不同格式）
        decision = getattr(approval, 'decision', None)
        if decision is not None:
            # 转换为字符串进行比较
            decision_str = decision.value if hasattr(decision, 'value') else str(decision)
            
            # 验证决策类型
            valid_decisions = ['approve', 'correct']
            if decision_str.lower() not in valid_decisions:
                self.validation_history.append({
                    'type': 'approval_validation',
                    'approval_id': approval_id,
                    'result': 'invalid',
                    'reason': f'Invalid decision type: {decision_str}',
                })
                raise InvalidAuditApproval(
                    f"Invalid approval decision: {decision_str}. "
                    f"Only APPROVE or CORRECT decisions can be written to production."
                )
        
        self.validation_history.append({
            'type': 'approval_validation',
            'approval_id': approval_id,
            'result': 'valid',
        })
        
        return True
    
    def validate_all(
        self,
        writer_id: str,
        approval: Union[AuditApprovalProtocol, Any]
    ) -> bool:
        """
        完整验证
        
        Args:
            writer_id: 写入者 ID
            approval: 审计批准证明
            
        Returns:
            是否通过所有验证
        """
        self.validate_writer(writer_id)
        self.validate_approval(approval)
        return True
    
    def add_authorized_writer(self, writer_id: str):
        """添加授权写入者"""
        self.authorized_writers.add(writer_id)
    
    def remove_authorized_writer(self, writer_id: str):
        """移除授权写入者"""
        self.authorized_writers.discard(writer_id)
    
    def get_validation_history(self, limit: int = 50) -> list:
        """获取验证历史"""
        return self.validation_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        allowed = sum(1 for v in self.validation_history if v['result'] in ['allowed', 'valid'])
        denied = sum(1 for v in self.validation_history if v['result'] in ['denied', 'invalid'])
        
        return {
            'total_validations': len(self.validation_history),
            'allowed': allowed,
            'denied': denied,
            'authorized_writers': list(self.authorized_writers),
        }

