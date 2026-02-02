/**
 * 数据备份 API 客户端
 */

import http from '@/utils/http'
import type {
  DataBackup,
  BackupCreateRequest,
  BackupRestoreRequest,
  BackupStatistics,
  TableInfo
} from '@/types/system'
import type { PaginatedResponse } from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/backups'

/**
 * 获取备份列表
 */
export function getList(params?: {
  page?: number
  page_size?: number
  status?: string
  backup_type?: string
}) {
  return http.get<{ data: PaginatedResponse<DataBackup> }>(`${BASE_URL}/`, { params })
}

/**
 * 获取备份详情
 */
export function getById(id: number) {
  return http.get<{ data: DataBackup }>(`${BASE_URL}/${id}`)
}

/**
 * 获取备份统计
 */
export function getStats() {
  return http.get<{ data: BackupStatistics }>(`${BASE_URL}/stats`)
}

/**
 * 获取可备份表信息
 */
export function getTableInfo() {
  return http.get<{ data: TableInfo[] }>(`${BASE_URL}/tables`)
}

/**
 * 创建备份
 */
export function create(data: BackupCreateRequest) {
  return http.post<{ data: { id: number; status: string } }>(`${BASE_URL}/`, data)
}

/**
 * 恢复备份
 */
export function restore(id: number, data?: BackupRestoreRequest) {
  return http.post<{
    data: {
      backup_id: number
      results: Record<string, { success: boolean; records?: number; error?: string }>
      restored_at: string
    }
  }>(`${BASE_URL}/${id}/restore`, data || {})
}

/**
 * 删除备份
 */
export function remove(id: number) {
  return http.delete(`${BASE_URL}/${id}`)
}

export const backupsApi = {
  getList,
  getById,
  getStats,
  getTableInfo,
  create,
  restore,
  remove
}

export default backupsApi
