/**
 * 学习控制相关类型定义 (v4.0)
 */

import type { LearningSessionState } from './enums'

// ==================== 学习会话 ====================

/** 学习范围 */
export interface LearningScope {
  maxLevel: string
  allowedLevels: string[]
  createdBy: string
}

/** 学习进度 */
export interface LearningProgress {
  totalSteps: number
  completedSteps: number
  currentDepth: number
  progressPercent: number
}

/** 方向变更记录 */
export interface DirectionChange {
  timestamp: string
  oldDirection: string
  newDirection: string
  reason: string
  actor: string
  checkpointId: string | null
}

/** 学习会话 */
export interface LearningSession {
  sessionId: string
  startedAt: string
  state: LearningSessionState
  goal: string | null
  scope: LearningScope | null
  progress: LearningProgress
  checkpoints: string[]
  latestCheckpointId: string | null
  directionChanges: DirectionChange[]
  isPaused: boolean
  pausedAt: string | null
  pauseReason: string | null
}

// ==================== 检查点 ====================

/** 检查点 */
export interface Checkpoint {
  checkpointId: string
  sessionId: string
  createdAt: string
  reason: string
  stateSnapshot: Record<string, any>
  progressSnapshot: LearningProgress
  metadata: Record<string, any>
}

// ==================== 可视化数据 ====================

/** 状态流转项 */
export interface StateFlowItem {
  fromState: string
  toState: string
  event: string
  timestamp: string
  success: boolean
}

/** 进度曲线点 */
export interface ProgressCurvePoint {
  timestamp: string
  progressPercent: number
  depth: number
}

/** 检查点标记 */
export interface CheckpointMarker {
  checkpointId: string
  timestamp: string
  reason: string
  progress: LearningProgress
}

/** 学习统计 */
export interface LearningStatistics {
  durationSeconds: number | null
  totalSteps: number
  completedSteps: number
  progressPercent: number
  checkpointsCount: number
  directionChangesCount: number
  currentDepth: number
  isPaused: boolean
}

/** 学习可视化数据 */
export interface LearningVisualizationData {
  sessionId: string
  stateFlow: StateFlowItem[]
  timelineEvents: TimelineEventItem[]
  progressCurve: ProgressCurvePoint[]
  checkpointMarkers: CheckpointMarker[]
  directionChangeMarkers: DirectionChange[]
  statistics: LearningStatistics
}

/** 时间线事件项 */
export interface TimelineEventItem {
  type: string
  timestamp: string
  description: string
  checkpointId?: string
  reason?: string
}

// ==================== 学习状态 ====================

/** 学习状态 */
export interface LearningStatus {
  hasActiveSession: boolean
  session: LearningSession | null
  progress: LearningProgress | null
  availableActions: string[]
}
