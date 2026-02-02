"""
NLGSM 枚举定义
"""
from enum import Enum


class NLGSMState(str, Enum):
    """NLGSM 核心状态（扩展为 9 种）"""
    LEARNING = "learning"           # 学习态
    VALIDATION = "validation"       # 验证态
    FROZEN = "frozen"               # 冻结态
    RELEASE = "release"             # 发布态
    ROLLBACK = "rollback"           # 回滚态
    SAFE_HALT = "safe_halt"         # 安全停机
    DIAGNOSIS = "diagnosis"         # 诊断态
    RECOVERY_PLAN = "recovery_plan" # 恢复计划
    PAUSED = "paused"               # 暂停态（v4.0 新增，可恢复的学习暂停）


class EventType(str, Enum):
    """状态转换事件类型"""
    # ==================== 原有事件 ====================
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
    
    # ==================== 学习控制事件 (v4.0 新增) ====================
    PAUSE_LEARNING = "pause_learning"        # 暂停学习（可恢复）
    RESUME_LEARNING = "resume_learning"      # 恢复学习
    STOP_LEARNING = "stop_learning"          # 停止学习（终止当前会话）
    REDIRECT_LEARNING = "redirect_learning"  # 调整学习方向
    CHECKPOINT_REQUEST = "checkpoint_request"  # 请求创建检查点
    ROLLBACK_TO_CHECKPOINT = "rollback_to_checkpoint"  # 回滚到检查点


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
    # 学习控制通知 (v4.0 新增)
    LEARNING_PAUSED = "learning_paused"
    LEARNING_RESUMED = "learning_resumed"
    LEARNING_STOPPED = "learning_stopped"
    LEARNING_REDIRECTED = "learning_redirected"
    CHECKPOINT_CREATED = "checkpoint_created"


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

