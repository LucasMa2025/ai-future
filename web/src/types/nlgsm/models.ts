/**
 * NLGSM 核心模型接口定义
 */

import type {
  NLGSMState,
  RiskLevel,
  EventType,
  Decision,
  LearningUnitStatus,
  ApprovalStatus,
  ApprovalDecision,
  AnomalySeverity,
  AnomalyStatus
} from './enums'

// ==================== 用户与角色 ====================

/** 用户信息 */
export interface User {
  id: string
  username: string
  email: string
  roles: string[]
  isActive: boolean
  avatar?: string
  createdAt?: string
}

/** 角色信息 */
export interface Role {
  id: string
  name: string
  displayName: string
  permissions: string[]
  riskLevelLimit: RiskLevel
}

// ==================== 状态机 ====================

/** 系统状态 */
export interface SystemState {
  state: NLGSMState
  enteredAt: string
  triggerEvent: string
  triggerSource?: string
  iterationCount: number
  metadata?: Record<string, any>
}

/** 状态转换 */
export interface StateTransition {
  id: string
  fromState: NLGSMState
  toState: NLGSMState
  triggerEvent: EventType
  triggerSource: string
  decision: Decision
  decisionReason?: string
  decisionEvidence?: Record<string, any>
  actionsExecuted: string[]
  success: boolean
  errorMessage?: string
  durationMs: number
  createdAt: string
}

/** 可用的状态转换 */
export interface AvailableTransition {
  eventType: EventType
  targetState: NLGSMState
}

// ==================== Learning Unit ====================

/** 约束条件 */
export interface Constraint {
  condition: string
  decision: string
}

/** 探索步骤 */
export interface ExplorationStep {
  stepId: string
  action: string
  result: string
  timestamp: string
  depth: number
}

/** 来源追溯 */
export interface Provenance {
  origin: string
  timestamp: string
  parentLuId?: string
  chainDepth: number
}

/** Learning Unit */
export interface LearningUnit {
  id: string
  version: number
  source: string
  learningGoal: string
  knowledge: string
  provenance: Provenance
  status: LearningUnitStatus
  riskLevel?: RiskLevel
  constraints: Constraint[]
  explorationSteps: ExplorationStep[]
  createdAt: string
  updatedAt: string
  // 审批相关
  approvalId?: string
  approvers?: string[]
  approvedAt?: string
  rejectedReason?: string
}

// ==================== 工件 ====================

/** 工件快照 */
export interface ArtifactSnapshot {
  parameters: Record<string, any>
  metrics: Record<string, number>
  timestamp: string
}

/** 工件 */
export interface Artifact {
  id: string
  version: number
  snapshot: ArtifactSnapshot
  metrics: Record<string, number>
  riskScore: number
  nlState: NLGSMState
  isApproved: boolean
  integrityHash: string
  approvers: string[]
  createdAt: string
  updatedAt: string
}

/** 版本差异 */
export interface VersionDiff {
  field: string
  oldValue: any
  newValue: any
  changeType: 'added' | 'removed' | 'modified'
}

// ==================== 审批 ====================

/** 审批记录 */
export interface Approval {
  id: string
  targetType: 'learning_unit' | 'artifact' | 'state_transition'
  targetId: string
  status: ApprovalStatus
  riskLevel: RiskLevel
  requiredApprovers: number
  currentApprovers: string[]
  decisions: ApprovalDecisionRecord[]
  createdAt: string
  updatedAt: string
  expiresAt?: string
}

/** 审批决策记录 */
export interface ApprovalDecisionRecord {
  userId: string
  username: string
  decision: ApprovalDecision
  comment?: string
  createdAt: string
}

// ==================== 审计 ====================

/** 审计日志 */
export interface AuditLog {
  id: string
  action: string
  actor: string
  targetType: string
  targetId: string
  details: Record<string, any>
  ipAddress?: string
  userAgent?: string
  timestamp: string
  integrityHash: string
}

// ==================== 异常监控 ====================

/** 异常信号 */
export interface AnomalySignal {
  detectorId: string
  detectorName: string
  score: number
  threshold: number
  details: Record<string, any>
}

/** 异常事件 */
export interface AnomalyEvent {
  id: string
  severity: AnomalySeverity
  compositeScore: number
  detectedBy: string[]
  recommendation: string
  status: AnomalyStatus
  signals: AnomalySignal[]
  createdAt: string
  resolvedAt?: string
  resolvedBy?: string
}

/** 检测器状态 */
export interface DetectorStatus {
  id: string
  name: string
  enabled: boolean
  threshold: number
  lastCheckAt: string
  lastScore: number
  alertCount: number
}

// ==================== 仪表盘 ====================

/** 仪表盘统计 */
export interface DashboardStats {
  currentState: NLGSMState
  pendingApprovals: number
  todayTransitions: number
  activeArtifacts: number
  openAnomalies: number
  systemHealth: number
}

/** 时间线事件 */
export interface TimelineEvent {
  id: string
  type: 'state_change' | 'approval' | 'anomaly' | 'learning' | 'checkpoint'
  title: string
  description: string
  timestamp: string
  severity?: 'info' | 'warning' | 'error' | 'success'
  metadata?: Record<string, any>
}

// ==================== 通知 ====================

/** 通知 */
export interface Notification {
  id: string
  type: string
  title: string
  message: string
  read: boolean
  createdAt: string
  metadata?: Record<string, any>
}
