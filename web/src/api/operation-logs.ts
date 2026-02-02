/**
 * 操作日志 API 客户端
 */

import http from '@/utils/http'
import type { OperationLog, OperationLogStatistics, OperationLogQuery } from '@/types/system'
import type { PaginatedResponse } from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/operation-logs'

/**
 * 获取操作日志列表
 */
export function getList(params?: OperationLogQuery) {
  return http.get<{ data: PaginatedResponse<OperationLog> }>(`${BASE_URL}/`, { params })
}

/**
 * 获取日志详情
 */
export function getById(id: number) {
  return http.get<{ data: OperationLog }>(`${BASE_URL}/${id}`)
}

/**
 * 获取日志统计
 */
export function getStatistics(startDate?: string, endDate?: string) {
  return http.get<{ data: OperationLogStatistics }>(`${BASE_URL}/statistics`, {
    params: { start_date: startDate, end_date: endDate }
  })
}

/**
 * 导出日志
 */
export function exportLogs(params?: {
  start_date?: string
  end_date?: string
  username?: string
  method?: string
  is_success?: boolean
}) {
  return http.get(`${BASE_URL}/export`, {
    params,
    responseType: 'blob'
  })
}

/**
 * 清理旧日志
 */
export function cleanup(days: number) {
  return http.delete<{ data: { deleted_count: number } }>(`${BASE_URL}/cleanup`, {
    params: { days }
  })
}

export const operationLogsApi = {
  getList,
  getById,
  getStatistics,
  exportLogs,
  cleanup
}

export default operationLogsApi
