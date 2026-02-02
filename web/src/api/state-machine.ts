/**
 * 状态机 API
 */
import request from '@/utils/http'
import type {
  SystemState,
  StateTransition,
  AvailableTransition,
  DashboardStats,
  TimelineEvent
} from '@/types/nlgsm'
import type {
  TriggerEventRequest,
  TriggerEventResponse,
  ForceStateRequest,
  PaginatedResponse
} from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/state-machine'

export const stateMachineApi = {
  /**
   * 获取当前系统状态
   */
  getCurrentState() {
    return request.get<SystemState>({
      url: `${BASE_URL}/current`
    })
  },

  /**
   * 获取可用的状态转换
   */
  getAvailableTransitions() {
    return request.get<AvailableTransition[]>({
      url: `${BASE_URL}/transitions/available`
    })
  },

  /**
   * 触发状态转换事件
   */
  triggerEvent(data: TriggerEventRequest) {
    return request.post<TriggerEventResponse>({
      url: `${BASE_URL}/event`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 强制设置状态（管理员操作）
   */
  forceState(data: ForceStateRequest) {
    return request.post<TriggerEventResponse>({
      url: `${BASE_URL}/force`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 获取状态转换历史
   */
  getHistory(params?: { page?: number; pageSize?: number }) {
    return request.get<PaginatedResponse<StateTransition>>({
      url: `${BASE_URL}/history`,
      params
    })
  },

  /**
   * 获取仪表盘统计数据
   */
  getDashboardStats() {
    return request.get<DashboardStats>({
      url: `${BASE_URL}/dashboard/stats`
    })
  },

  /**
   * 获取时间线事件
   */
  getTimelineEvents(params?: { limit?: number }) {
    return request.get<TimelineEvent[]>({
      url: `${BASE_URL}/timeline`,
      params
    })
  },

  /**
   * 获取状态机图数据
   */
  getDiagramData() {
    return request.get<{
      states: Array<{ id: string; label: string; color: string }>
      transitions: Array<{ from: string; to: string; event: string }>
      currentState: string
    }>({
      url: `${BASE_URL}/diagram`
    })
  }
}
