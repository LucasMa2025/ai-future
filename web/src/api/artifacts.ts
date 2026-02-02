/**
 * 工件管理 API
 */
import request from '@/utils/http'
import type { Artifact, VersionDiff } from '@/types/nlgsm'
import type { ArtifactListParams, RollbackRequest, PaginatedResponse } from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/artifacts'

export const artifactsApi = {
  /**
   * 获取工件列表
   */
  getList(params?: ArtifactListParams) {
    return request.get<PaginatedResponse<Artifact>>({
      url: BASE_URL,
      params
    })
  },

  /**
   * 获取工件详情
   */
  getById(id: string) {
    return request.get<Artifact>({
      url: `${BASE_URL}/${id}`
    })
  },

  /**
   * 获取工件版本历史
   */
  getVersionHistory(id: string) {
    return request.get<{
      versions: Array<{
        version: number
        createdAt: string
        createdBy: string
        changeDescription: string
      }>
    }>({
      url: `${BASE_URL}/${id}/versions`
    })
  },

  /**
   * 获取版本差异
   */
  getDiff(id: string, fromVersion: number, toVersion: number) {
    return request.get<VersionDiff[]>({
      url: `${BASE_URL}/${id}/diff`,
      params: { fromVersion, toVersion }
    })
  },

  /**
   * 回滚工件
   */
  rollback(id: string, data: RollbackRequest) {
    return request.post<{ success: boolean; message: string; newVersion: number }>({
      url: `${BASE_URL}/${id}/rollback`,
      data,
      showSuccessMessage: true
    })
  },

  /**
   * 验证工件完整性
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
   * 获取工件统计信息
   */
  getStats() {
    return request.get<{
      total: number
      approved: number
      pending: number
      byState: Record<string, number>
    }>({
      url: `${BASE_URL}/stats`
    })
  }
}
