/**
 * 系统配置 API 客户端
 */

import http from '@/utils/http'
import type {
  SystemConfig,
  ConfigGroup,
  ConfigCreateRequest,
  ConfigUpdateRequest,
  BatchConfigUpdateRequest
} from '@/types/system'

const BASE_URL = '/api/v1/system-configs'

/**
 * 获取所有配置
 */
export function getList(group?: string) {
  return http.get<{ data: SystemConfig[] }>(`${BASE_URL}/`, {
    params: { group }
  })
}

/**
 * 获取配置分组
 */
export function getGroups() {
  return http.get<{ data: ConfigGroup[] }>(`${BASE_URL}/groups`)
}

/**
 * 获取分组配置
 */
export function getGroupConfigs(group: string) {
  return http.get<{ data: Record<string, any> }>(`${BASE_URL}/group/${group}`)
}

/**
 * 获取单个配置
 */
export function getByKey(key: string) {
  return http.get<{ data: SystemConfig }>(`${BASE_URL}/${key}`)
}

/**
 * 更新配置
 */
export function update(key: string, data: ConfigUpdateRequest) {
  return http.put<{ data: SystemConfig }>(`${BASE_URL}/${key}`, data)
}

/**
 * 批量更新配置
 */
export function batchUpdate(data: BatchConfigUpdateRequest) {
  return http.put(`${BASE_URL}/`, data)
}

/**
 * 创建配置
 */
export function create(data: ConfigCreateRequest) {
  return http.post<{ data: { id: number } }>(`${BASE_URL}/`, data)
}

/**
 * 初始化默认配置
 */
export function initDefault() {
  return http.post<{ data: { count: number } }>(`${BASE_URL}/init`)
}

export const systemConfigsApi = {
  getList,
  getGroups,
  getGroupConfigs,
  getByKey,
  update,
  batchUpdate,
  create,
  initDefault
}

export default systemConfigsApi
