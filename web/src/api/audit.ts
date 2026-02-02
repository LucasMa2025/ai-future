/**
 * 审计日志 API
 */
import request from '@/utils/http'
import type { AuditLog } from '@/types/nlgsm'
import type { AuditLogParams, PaginatedResponse } from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/audit'

export const auditApi = {
  /**
   * 获取审计日志列表
   */
  getList(params?: AuditLogParams) {
    return request.get<PaginatedResponse<AuditLog>>({
      url: BASE_URL,
      params
    })
  },

  /**
   * 获取审计日志详情
   */
  getById(id: string) {
    return request.get<AuditLog>({
      url: `${BASE_URL}/${id}`
    })
  },

  /**
   * 验证日志完整性
   */
  verifyIntegrity(id: string) {
    return request.get<{
      valid: boolean
      computedHash: string
      storedHash: string
    }>({
      url: `${BASE_URL}/${id}/verify`
    })
  },

  /**
   * 批量验证完整性
   */
  batchVerify(ids: string[]) {
    return request.post<{
      results: Array<{
        id: string
        valid: boolean
      }>
      allValid: boolean
    }>({
      url: `${BASE_URL}/verify/batch`,
      data: { ids }
    })
  },

  /**
   * 导出审计日志
   */
  exportLogs(params?: AuditLogParams) {
    return request.get<Blob>({
      url: `${BASE_URL}/export`,
      params,
      responseType: 'blob'
    } as any)
  },

  /**
   * 获取操作类型列表
   */
  getActionTypes() {
    return request.get<string[]>({
      url: `${BASE_URL}/action-types`
    })
  },

  /**
   * 获取目标类型列表
   */
  getTargetTypes() {
    return request.get<string[]>({
      url: `${BASE_URL}/target-types`
    })
  },

  /**
   * 获取审计统计
   */
  getStats(params?: { startDate?: string; endDate?: string }) {
    return request.get<{
      totalLogs: number
      byAction: Record<string, number>
      byActor: Record<string, number>
      byDay: Array<{ date: string; count: number }>
    }>({
      url: `${BASE_URL}/stats`,
      params
    })
  }
}
