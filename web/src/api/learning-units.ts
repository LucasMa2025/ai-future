/**
 * Learning Units API
 */
import request from '@/utils/http'
import type { LearningUnit } from '@/types/nlgsm'
import type { LUListParams, LUApprovalRequest, PaginatedResponse } from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/learning-units'

export const learningUnitsApi = {
  /**
   * 获取 LU 列表
   */
  getList(params?: LUListParams) {
    return request.get<PaginatedResponse<LearningUnit>>({
      url: BASE_URL,
      params
    })
  },

  /**
   * 获取 LU 详情
   */
  getById(id: string) {
    return request.get<LearningUnit>({
      url: `${BASE_URL}/${id}`
    })
  },

  /**
   * 提交审批决策
   */
  submitApproval(id: string, data: LUApprovalRequest) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/${id}/approval`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 审批通过
   */
  approve(id: string, comment?: string) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/${id}/approve`,
      data: { comment },
      showSuccessMessage: true
    })
  },

  /**
   * 审批拒绝
   */
  reject(id: string, reason: string) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/${id}/reject`,
      data: { reason },
      showSuccessMessage: true
    })
  },

  /**
   * 修正 LU
   */
  correct(id: string, corrections: Record<string, any>, comment?: string) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/${id}/correct`,
      data: { corrections, comment },
      showSuccessMessage: true
    })
  },

  /**
   * 终止 LU
   */
  terminate(id: string, reason: string) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/${id}/terminate`,
      data: { reason },
      showSuccessMessage: true
    })
  },

  /**
   * 获取 LU 探索路径
   */
  getExplorationPath(id: string) {
    return request.get<{
      steps: Array<{
        stepId: string
        action: string
        result: string
        timestamp: string
        depth: number
      }>
    }>({
      url: `${BASE_URL}/${id}/exploration`
    })
  },

  /**
   * 获取 LU 统计信息
   */
  getStats() {
    return request.get<{
      total: number
      byStatus: Record<string, number>
      byRiskLevel: Record<string, number>
      pendingApproval: number
    }>({
      url: `${BASE_URL}/stats`
    })
  }
}
