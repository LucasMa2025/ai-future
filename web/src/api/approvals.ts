/**
 * 审批中心 API
 */
import request from '@/utils/http'
import type { Approval } from '@/types/nlgsm'
import type {
  ApprovalListParams,
  SubmitApprovalRequest,
  PaginatedResponse
} from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/approvals'

export const approvalsApi = {
  /**
   * 获取审批列表
   */
  getList(params?: ApprovalListParams) {
    return request.get<PaginatedResponse<Approval>>({
      url: BASE_URL,
      params
    })
  },

  /**
   * 获取待审批列表（分配给我的）
   */
  getPending() {
    return request.get<Approval[]>({
      url: `${BASE_URL}/pending`
    })
  },

  /**
   * 获取我的审批记录
   */
  getMyApprovals(params?: { page?: number; pageSize?: number }) {
    return request.get<PaginatedResponse<Approval>>({
      url: `${BASE_URL}/my`,
      params
    })
  },

  /**
   * 获取审批详情
   */
  getById(id: string) {
    return request.get<Approval>({
      url: `${BASE_URL}/${id}`
    })
  },

  /**
   * 提交审批决策
   */
  submitDecision(id: string, data: SubmitApprovalRequest) {
    return request.post<{ success: boolean; message: string }>({
      url: `${BASE_URL}/${id}/decision`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 批量审批
   */
  batchApprove(ids: string[], comment?: string) {
    return request.post<{ success: boolean; message: string; processed: number }>({
      url: `${BASE_URL}/batch/approve`,
      data: { ids, comment },
      showSuccessMessage: true
    })
  },

  /**
   * 获取审批统计
   */
  getStats() {
    return request.get<{
      pending: number
      completed: number
      rejected: number
      expired: number
      myPending: number
    }>({
      url: `${BASE_URL}/stats`
    })
  },

  /**
   * 获取审批矩阵配置
   */
  getApprovalMatrix() {
    return request.get<{
      matrix: Array<{
        riskLevel: string
        requiredApprovers: number
        allowedRoles: string[]
      }>
    }>({
      url: `${BASE_URL}/matrix`
    })
  }
}
