/**
 * 系统功能 API 客户端
 */

import http from '@/utils/http'
import type {
  SystemFunction,
  SystemFunctionCreateRequest,
  SystemFunctionUpdateRequest
} from '@/types/system'

const BASE_URL = '/api/v1/system-functions'

/**
 * 获取系统功能列表
 */
export function getList(includeDisabled = false) {
  return http.get<{ data: SystemFunction[] }>(`${BASE_URL}/`, {
    params: { include_disabled: includeDisabled }
  })
}

/**
 * 获取系统功能树
 */
export function getTree() {
  return http.get<{ data: SystemFunction[] }>(`${BASE_URL}/tree`)
}

/**
 * 获取功能详情
 */
export function getById(id: number) {
  return http.get<{ data: SystemFunction }>(`${BASE_URL}/${id}`)
}

/**
 * 创建系统功能
 */
export function create(data: SystemFunctionCreateRequest) {
  return http.post<{ data: { id: number } }>(`${BASE_URL}/`, data)
}

/**
 * 更新系统功能
 */
export function update(id: number, data: SystemFunctionUpdateRequest) {
  return http.put(`${BASE_URL}/${id}`, data)
}

/**
 * 删除系统功能
 */
export function remove(id: number) {
  return http.delete(`${BASE_URL}/${id}`)
}

/**
 * 初始化默认功能
 */
export function initDefault() {
  return http.post<{ data: { count: number } }>(`${BASE_URL}/init`)
}

export const systemFunctionsApi = {
  getList,
  getTree,
  getById,
  create,
  update,
  remove,
  initDefault
}

export default systemFunctionsApi
