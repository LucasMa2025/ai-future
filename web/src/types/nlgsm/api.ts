/**
 * NLGSM API 请求/响应类型定义
 */

import type { LearningScope } from './learning-control'
import type { RiskLevel, LearningUnitStatus, ApprovalStatus, AnomalyStatus } from './enums'

// ==================== 通用 ====================

/** 分页参数 */
export interface PaginationParams {
  page?: number
  pageSize?: number
}

/** 分页响应 */
export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  pageSize: number
  totalPages: number
}

// ==================== 状态机 API ====================

/** 触发事件请求 */
export interface TriggerEventRequest {
  eventType: string
  source?: string
  metadata?: Record<string, any>
}

/** 触发事件响应 */
export interface TriggerEventResponse {
  success: boolean
  fromState: string
  toState: string
  decision: string
  reason: string
  actionsExecuted: string[]
  durationMs: number
  error?: string
}

/** 强制状态请求 */
export interface ForceStateRequest {
  targetState: string
  reason: string
}

// ==================== Learning Unit API ====================

/** LU 列表筛选参数 */
export interface LUListParams extends PaginationParams {
  status?: LearningUnitStatus
  riskLevel?: RiskLevel
  search?: string
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}

/** LU 审批请求 */
export interface LUApprovalRequest {
  decision: 'approve' | 'reject' | 'correct'
  comment?: string
  corrections?: Record<string, any>
}

// ==================== 工件 API ====================

/** 工件列表筛选参数 */
export interface ArtifactListParams extends PaginationParams {
  search?: string
  isApproved?: boolean
  sortBy?: string
  sortOrder?: 'asc' | 'desc'
}

/** 回滚请求 */
export interface RollbackRequest {
  targetVersion: number
  reason: string
}

// ==================== 审批 API ====================

/** 审批列表筛选参数 */
export interface ApprovalListParams extends PaginationParams {
  status?: ApprovalStatus
  targetType?: string
  assignedToMe?: boolean
}

/** 提交审批决策请求 */
export interface SubmitApprovalRequest {
  decision: string
  comment?: string
}

// ==================== 审计 API ====================

/** 审计日志筛选参数 */
export interface AuditLogParams extends PaginationParams {
  action?: string
  actor?: string
  targetType?: string
  targetId?: string
  startDate?: string
  endDate?: string
}

// ==================== 异常监控 API ====================

/** 异常列表筛选参数 */
export interface AnomalyListParams extends PaginationParams {
  status?: AnomalyStatus
  severity?: string
  startDate?: string
  endDate?: string
}

/** 更新检测器配置请求 */
export interface UpdateDetectorRequest {
  enabled?: boolean
  threshold?: number
}

/** 解决异常请求 */
export interface ResolveAnomalyRequest {
  resolution: string
  comment?: string
}

// ==================== 学习控制 API ====================

/** 启动学习请求 */
export interface StartLearningRequest {
  goal: string
  scope?: LearningScope
}

/** 暂停学习请求 */
export interface PauseLearningRequest {
  reason: string
}

/** 停止学习请求 */
export interface StopLearningRequest {
  reason: string
  saveProgress?: boolean
}

/** 调整方向请求 */
export interface RedirectLearningRequest {
  newDirection: string
  reason: string
  newScope?: LearningScope
}

/** 创建检查点请求 */
export interface CreateCheckpointRequest {
  reason?: string
  metadata?: Record<string, any>
}

/** 回滚到检查点请求 */
export interface RollbackToCheckpointRequest {
  checkpointId: string
  reason: string
}

/** 更新进度请求 */
export interface UpdateProgressRequest {
  completedSteps?: number
  totalSteps?: number
  currentDepth?: number
}

/** 学习控制响应 */
export interface LearningControlResponse {
  success: boolean
  message: string
  data?: Record<string, any>
}
