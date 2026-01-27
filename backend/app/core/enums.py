"""
NLGSM 枚举定义
"""
from enum import Enum


class NLGSMState(str, Enum):
    """NLGSM 八种核心状态"""
    LEARNING = "learning"
    VALIDATION = "validation"
    FROZEN = "frozen"
    RELEASE = "release"
    ROLLBACK = "rollback"
    SAFE_HALT = "safe_halt"
    DIAGNOSIS = "diagnosis"
    RECOVERY_PLAN = "recovery_plan"


class EventType(str, Enum):
    """状态转换事件类型"""
    REACH_STEPS = "reach_steps"
    AUDIT_SIGNAL = "audit_signal"
    PERIODIC = "periodic"
    PASS_VALIDATION = "pass_validation"
    FAIL_BUT_FIXABLE = "fail_but_fixable"
    HUMAN_APPROVE = "human_approve"
    RELEASE_COMPLETE = "release_complete"
    ANOMALY = "anomaly"
    RECOVER = "recover"
    START_DIAGNOSIS = "start_diagnosis"
    DIAGNOSIS_COMPLETE = "diagnosis_complete"
    PLAN_APPROVED = "plan_approved"
    PLAN_RESTORE = "plan_restore"
    PLAN_REJECTED = "plan_rejected"


class Decision(str, Enum):
    """治理决策"""
    ALLOW = "allow"
    DENY = "deny"
    ROLLBACK = "rollback"
    HALT = "halt"
    DIAGNOSE = "diagnose"


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditLevel(str, Enum):
    """审计级别"""
    NONE = "none"          # 不审计
    INFO = "info"          # 信息级别
    NORMAL = "normal"      # 普通审计
    IMPORTANT = "important"  # 重要审计
    CRITICAL = "critical"  # 关键审计


class NotificationType(str, Enum):
    """通知类型"""
    APPROVAL_REQUIRED = "approval_required"
    STATE_CHANGED = "state_changed"
    ANOMALY_DETECTED = "anomaly_detected"
    SAFE_HALT_TRIGGERED = "safe_halt_triggered"
    APPROVAL_COMPLETED = "approval_completed"
    SYSTEM_ALERT = "system_alert"


class NotificationChannel(str, Enum):
    """通知渠道"""
    WEBSOCKET = "websocket"
    EMAIL = "email"
    BOTH = "both"


class ApprovalStatus(str, Enum):
    """审批状态"""
    PENDING = "pending"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalDecision(str, Enum):
    """审批决策"""
    APPROVE = "approve"
    REJECT = "reject"
    CORRECT = "correct"
    TERMINATE = "terminate"


class LearningUnitStatus(str, Enum):
    """Learning Unit 状态"""
    PENDING = "pending"
    AUTO_CLASSIFIED = "auto_classified"
    HUMAN_REVIEW = "human_review"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    TERMINATED = "terminated"


class AnomalySeverity(str, Enum):
    """异常严重性"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyStatus(str, Enum):
    """异常状态"""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    IGNORED = "ignored"


class TransactionStatus(str, Enum):
    """事务状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMMITTED = "committed"
    ABORTED = "aborted"


class TransactionType(str, Enum):
    """事务类型"""
    ROLLBACK = "rollback"
    RELEASE = "release"
    UPDATE = "update"

