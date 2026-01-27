"""
生产桥接器

被动接收审计系统写入的 Learning Unit
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.types import LearningUnit, AuditApproval
from core.enums import BridgeWriteResult, AuditDecision
from core.exceptions import UnauthorizedWriter, InvalidAuditApproval, InternalizationFailed

from .permission import PermissionValidator
from .internalization import InternalizationEngine


class ProductionBridge:
    """
    生产桥接器
    
    特性：
    - 被动接收：只接受审计系统的写入
    - 权限验证：验证写入者和审计批准
    - 内化执行：将约束内化到 Decision Head
    - 版本管理：支持回滚
    
    注意：
    - 自学习系统无权直接写入
    - 只有审计系统有权写入
    """
    
    def __init__(
        self,
        authorized_writers: Optional[set] = None,
        hidden_dim: int = 256
    ):
        # 权限验证器
        self.permission_validator = PermissionValidator(
            authorized_writers=authorized_writers or {"audit_system"}
        )
        
        # 内化引擎
        self.internalization_engine = InternalizationEngine(
            hidden_dim=hidden_dim
        )
        
        # 已写入的 Learning Units
        self.written_units: List[Dict[str, Any]] = []
        
        # 写入历史
        self.write_history: List[Dict[str, Any]] = []
    
    def write_learning_unit(
        self,
        learning_unit: LearningUnit,
        writer_id: str,
        audit_approval: AuditApproval
    ) -> BridgeWriteResult:
        """
        写入 Learning Unit
        
        Args:
            learning_unit: Learning Unit
            writer_id: 写入者 ID
            audit_approval: 审计批准证明
            
        Returns:
            写入结果
        """
        print(f"\n[生产桥接器] 接收写入请求")
        print(f"  Learning Unit: {learning_unit.id}")
        print(f"  写入者: {writer_id}")
        print(f"  审批 ID: {audit_approval.approval_id}")
        
        # 1. 验证权限
        try:
            self.permission_validator.validate_all(writer_id, audit_approval)
        except UnauthorizedWriter as e:
            print(f"  ❌ 权限拒绝: {e}")
            self._record_write(learning_unit.id, writer_id, BridgeWriteResult.PERMISSION_DENIED, str(e))
            return BridgeWriteResult.PERMISSION_DENIED
        except InvalidAuditApproval as e:
            print(f"  ❌ 审批无效: {e}")
            self._record_write(learning_unit.id, writer_id, BridgeWriteResult.INVALID_APPROVAL, str(e))
            return BridgeWriteResult.INVALID_APPROVAL
        
        print(f"  ✓ 权限验证通过")
        
        # 2. 执行内化
        try:
            internalization_result = self.internalization_engine.internalize(learning_unit)
            
            if internalization_result['status'] == 'failed':
                print(f"  ❌ 内化失败")
                self._record_write(learning_unit.id, writer_id, BridgeWriteResult.INTERNALIZATION_FAILED)
                return BridgeWriteResult.INTERNALIZATION_FAILED
        
        except Exception as e:
            print(f"  ❌ 内化异常: {e}")
            self._record_write(learning_unit.id, writer_id, BridgeWriteResult.INTERNALIZATION_FAILED, str(e))
            return BridgeWriteResult.INTERNALIZATION_FAILED
        
        # 3. 记录写入
        self.written_units.append({
            'unit_id': learning_unit.id,
            'writer_id': writer_id,
            'approval_id': audit_approval.approval_id,
            'version': internalization_result.get('version'),
            'timestamp': datetime.now().isoformat(),
        })
        
        self._record_write(learning_unit.id, writer_id, BridgeWriteResult.SUCCESS)
        
        print(f"  ✓ 写入成功 (版本: {internalization_result.get('version')})")
        
        return BridgeWriteResult.SUCCESS
    
    def _record_write(
        self,
        unit_id: str,
        writer_id: str,
        result: BridgeWriteResult,
        error: str = ""
    ):
        """记录写入历史"""
        self.write_history.append({
            'unit_id': unit_id,
            'writer_id': writer_id,
            'result': result.value,
            'error': error,
            'timestamp': datetime.now().isoformat(),
        })
    
    def get_decision(self, hidden_states) -> Dict[str, Any]:
        """获取决策"""
        return self.internalization_engine.get_decision(hidden_states)
    
    def get_written_units(self) -> List[Dict[str, Any]]:
        """获取已写入的 Learning Units"""
        return self.written_units.copy()
    
    def get_write_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取写入历史"""
        return self.write_history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_count = sum(1 for w in self.write_history if w['result'] == 'success')
        denied_count = sum(1 for w in self.write_history if w['result'] == 'permission_denied')
        
        return {
            'total_writes': len(self.write_history),
            'successful_writes': success_count,
            'denied_writes': denied_count,
            'written_units_count': len(self.written_units),
            'permission_stats': self.permission_validator.get_statistics(),
            'internalization_stats': self.internalization_engine.get_statistics(),
        }

