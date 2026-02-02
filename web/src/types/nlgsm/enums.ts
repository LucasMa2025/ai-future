/**
 * NLGSM 枚举定义
 */

/** NLGSM 核心状态（9种） */
export enum NLGSMState {
  LEARNING = 'learning',
  VALIDATION = 'validation',
  FROZEN = 'frozen',
  RELEASE = 'release',
  ROLLBACK = 'rollback',
  SAFE_HALT = 'safe_halt',
  DIAGNOSIS = 'diagnosis',
  RECOVERY_PLAN = 'recovery_plan',
  PAUSED = 'paused'
}

/** 风险等级 */
export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

/** 事件类型 */
export enum EventType {
  // 原有事件
  REACH_STEPS = 'reach_steps',
  AUDIT_SIGNAL = 'audit_signal',
  PERIODIC = 'periodic',
  PASS_VALIDATION = 'pass_validation',
  FAIL_BUT_FIXABLE = 'fail_but_fixable',
  HUMAN_APPROVE = 'human_approve',
  RELEASE_COMPLETE = 'release_complete',
  ANOMALY = 'anomaly',
  RECOVER = 'recover',
  START_DIAGNOSIS = 'start_diagnosis',
  DIAGNOSIS_COMPLETE = 'diagnosis_complete',
  PLAN_APPROVED = 'plan_approved',
  PLAN_RESTORE = 'plan_restore',
  PLAN_REJECTED = 'plan_rejected',
  // 学习控制事件 (v4.0)
  PAUSE_LEARNING = 'pause_learning',
  RESUME_LEARNING = 'resume_learning',
  STOP_LEARNING = 'stop_learning',
  REDIRECT_LEARNING = 'redirect_learning',
  CHECKPOINT_REQUEST = 'checkpoint_request',
  ROLLBACK_TO_CHECKPOINT = 'rollback_to_checkpoint'
}

/** 治理决策 */
export enum Decision {
  ALLOW = 'allow',
  DENY = 'deny',
  ROLLBACK = 'rollback',
  HALT = 'halt',
  DIAGNOSE = 'diagnose'
}

/** Learning Unit 状态 */
export enum LearningUnitStatus {
  PENDING = 'pending',
  AUTO_CLASSIFIED = 'auto_classified',
  HUMAN_REVIEW = 'human_review',
  APPROVED = 'approved',
  CORRECTED = 'corrected',
  REJECTED = 'rejected',
  TERMINATED = 'terminated'
}

/** 审批状态 */
export enum ApprovalStatus {
  PENDING = 'pending',
  COMPLETED = 'completed',
  REJECTED = 'rejected',
  EXPIRED = 'expired'
}

/** 审批决策 */
export enum ApprovalDecision {
  APPROVE = 'approve',
  REJECT = 'reject',
  CORRECT = 'correct',
  TERMINATE = 'terminate'
}

/** 异常严重性 */
export enum AnomalySeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

/** 异常状态 */
export enum AnomalyStatus {
  OPEN = 'open',
  INVESTIGATING = 'investigating',
  RESOLVED = 'resolved',
  IGNORED = 'ignored'
}

/** 通知类型 */
export enum NotificationType {
  APPROVAL_REQUIRED = 'approval_required',
  STATE_CHANGED = 'state_changed',
  ANOMALY_DETECTED = 'anomaly_detected',
  SAFE_HALT_TRIGGERED = 'safe_halt_triggered',
  APPROVAL_COMPLETED = 'approval_completed',
  SYSTEM_ALERT = 'system_alert',
  LEARNING_PAUSED = 'learning_paused',
  LEARNING_RESUMED = 'learning_resumed',
  LEARNING_STOPPED = 'learning_stopped',
  LEARNING_REDIRECTED = 'learning_redirected',
  CHECKPOINT_CREATED = 'checkpoint_created'
}

/** 学习会话状态 */
export enum LearningSessionState {
  LEARNING = 'learning',
  PAUSED = 'paused',
  FROZEN = 'frozen',
  VALIDATION = 'validation'
}

/** 学习控制操作 */
export enum LearningControlAction {
  START = 'start',
  PAUSE = 'pause',
  RESUME = 'resume',
  STOP = 'stop',
  REDIRECT = 'redirect',
  CHECKPOINT = 'checkpoint',
  ROLLBACK = 'rollback'
}

/** 状态标签映射 */
export const stateLabels: Record<NLGSMState, string> = {
  [NLGSMState.LEARNING]: '学习中',
  [NLGSMState.VALIDATION]: '验证中',
  [NLGSMState.FROZEN]: '已冻结',
  [NLGSMState.RELEASE]: '发布中',
  [NLGSMState.ROLLBACK]: '回滚中',
  [NLGSMState.SAFE_HALT]: '安全停机',
  [NLGSMState.DIAGNOSIS]: '诊断中',
  [NLGSMState.RECOVERY_PLAN]: '恢复计划',
  [NLGSMState.PAUSED]: '已暂停'
}

/** 状态颜色映射 */
export const stateColors: Record<NLGSMState, string> = {
  [NLGSMState.LEARNING]: '#ff9800',
  [NLGSMState.VALIDATION]: '#2196f3',
  [NLGSMState.FROZEN]: '#9c27b0',
  [NLGSMState.RELEASE]: '#4caf50',
  [NLGSMState.ROLLBACK]: '#f44336',
  [NLGSMState.SAFE_HALT]: '#b71c1c',
  [NLGSMState.DIAGNOSIS]: '#607d8b',
  [NLGSMState.RECOVERY_PLAN]: '#795548',
  [NLGSMState.PAUSED]: '#9e9e9e'
}

/** 风险等级标签映射 */
export const riskLabels: Record<RiskLevel, string> = {
  [RiskLevel.LOW]: '低风险',
  [RiskLevel.MEDIUM]: '中风险',
  [RiskLevel.HIGH]: '高风险',
  [RiskLevel.CRITICAL]: '严重'
}

/** 风险等级颜色映射 */
export const riskColors: Record<RiskLevel, string> = {
  [RiskLevel.LOW]: '#4caf50',
  [RiskLevel.MEDIUM]: '#ff9800',
  [RiskLevel.HIGH]: '#f44336',
  [RiskLevel.CRITICAL]: '#b71c1c'
}
