/**
 * 学习控制 API (v4.0)
 */
import request from '@/utils/http'
import type {
  LearningSession,
  Checkpoint,
  LearningVisualizationData,
  LearningStatus
} from '@/types/nlgsm'
import type {
  StartLearningRequest,
  PauseLearningRequest,
  StopLearningRequest,
  RedirectLearningRequest,
  CreateCheckpointRequest,
  RollbackToCheckpointRequest,
  UpdateProgressRequest,
  LearningControlResponse
} from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/learning'

export const learningControlApi = {
  // ==================== 会话管理 ====================

  /**
   * 启动学习会话
   */
  startSession(data: StartLearningRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/session/start`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 获取当前会话
   */
  getCurrentSession() {
    return request.get<{ success: boolean; data: LearningSession | null }>({
      url: `${BASE_URL}/session/current`
    })
  },

  /**
   * 获取会话历史
   */
  getSessionHistory(limit = 20) {
    return request.get<{ success: boolean; data: LearningSession[]; count: number }>({
      url: `${BASE_URL}/session/history`,
      params: { limit }
    })
  },

  // ==================== 学习控制 ====================

  /**
   * 暂停学习
   */
  pause(data: PauseLearningRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/pause`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 恢复学习
   */
  resume() {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/resume`,
      showSuccessMessage: true
    })
  },

  /**
   * 停止学习
   */
  stop(data: StopLearningRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/stop`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 调整学习方向
   */
  redirect(data: RedirectLearningRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/redirect`,
      data,
      showSuccessMessage: true
    })
  },

  // ==================== 检查点管理 ====================

  /**
   * 创建检查点
   */
  createCheckpoint(data: CreateCheckpointRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/checkpoint`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 获取检查点列表
   */
  getCheckpoints(sessionId?: string, limit = 50) {
    return request.get<{ success: boolean; data: Checkpoint[]; count: number }>({
      url: `${BASE_URL}/checkpoints`,
      params: { session_id: sessionId, limit }
    })
  },

  /**
   * 回滚到检查点
   */
  rollback(data: RollbackToCheckpointRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/rollback`,
      data,
      showSuccessMessage: true
    })
  },

  // ==================== 进度与可视化 ====================

  /**
   * 获取学习进度
   */
  getProgress() {
    return request.get<{ success: boolean; data: any }>({
      url: `${BASE_URL}/progress`
    })
  },

  /**
   * 更新学习进度
   */
  updateProgress(data: UpdateProgressRequest) {
    return request.post<LearningControlResponse>({
      url: `${BASE_URL}/progress`,
      data
    })
  },

  /**
   * 获取可视化数据
   */
  getVisualizationData(sessionId?: string) {
    return request.get<{ success: boolean; data: LearningVisualizationData }>({
      url: `${BASE_URL}/visualization`,
      params: { session_id: sessionId }
    })
  },

  // ==================== 状态查询 ====================

  /**
   * 获取完整学习状态
   */
  getStatus() {
    return request.get<{ success: boolean; data: LearningStatus }>({
      url: `${BASE_URL}/status`
    })
  }
}
