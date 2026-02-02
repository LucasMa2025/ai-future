/**
 * 异常监控 API
 */
import request from '@/utils/http'
import type { AnomalyEvent, DetectorStatus } from '@/types/nlgsm'
import type {
  AnomalyListParams,
  UpdateDetectorRequest,
  ResolveAnomalyRequest,
  PaginatedResponse
} from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/anomaly'

export const anomalyApi = {
  /**
   * 获取异常事件列表
   */
  getList(params?: AnomalyListParams) {
    return request.get<PaginatedResponse<AnomalyEvent>>({
      url: `${BASE_URL}/events`,
      params
    })
  },

  /**
   * 获取异常事件详情
   */
  getById(id: string) {
    return request.get<AnomalyEvent>({
      url: `${BASE_URL}/events/${id}`
    })
  },

  /**
   * 解决异常
   */
  resolve(id: string, data: ResolveAnomalyRequest) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/events/${id}/resolve`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 忽略异常
   */
  ignore(id: string, reason: string) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/events/${id}/ignore`,
      data: { reason },
      showSuccessMessage: true
    })
  },

  /**
   * 获取检测器列表
   */
  getDetectors() {
    return request.get<DetectorStatus[]>({
      url: `${BASE_URL}/detectors`
    })
  },

  /**
   * 更新检测器配置
   */
  updateDetector(id: string, data: UpdateDetectorRequest) {
    return request.put<{ success: boolean; message: string }>({
      url: `${BASE_URL}/detectors/${id}`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 手动触发检测
   */
  triggerDetection(detectorId?: string) {
    return request.post<{
      triggered: string[]
      results: Array<{
        detectorId: string
        score: number
        anomalyDetected: boolean
      }>
    }>({
      url: `${BASE_URL}/detect`,
      data: { detectorId }
    })
  },

  /**
   * 获取异常统计
   */
  getStats() {
    return request.get<{
      open: number
      investigating: number
      resolved: number
      ignored: number
      bySeverity: Record<string, number>
      recentTrend: Array<{ date: string; count: number }>
    }>({
      url: `${BASE_URL}/stats`
    })
  },

  /**
   * 获取实时告警
   */
  getActiveAlerts() {
    return request.get<AnomalyEvent[]>({
      url: `${BASE_URL}/alerts/active`
    })
  }
}
