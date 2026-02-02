/**
 * 用户管理 API 客户端
 */

import http from '@/utils/http'
import type {
  User,
  UserCreateRequest,
  UserUpdateRequest,
  ResetPasswordRequest
} from '@/types/system'
import type { PaginatedResponse } from '@/types/nlgsm/api'

const BASE_URL = '/api/v1/users'

/**
 * 获取用户列表
 */
export function getList(params?: {
  page?: number
  page_size?: number
  username?: string
  email?: string
  is_active?: boolean
}) {
  return http.get<{ data: PaginatedResponse<User> }>(`${BASE_URL}/`, { params })
}

/**
 * 获取用户详情
 */
export function getById(id: string) {
  return http.get<{ data: User }>(`${BASE_URL}/${id}`)
}

/**
 * 创建用户
 */
export function create(data: UserCreateRequest) {
  return http.post<{ data: { id: string } }>(`${BASE_URL}/`, data)
}

/**
 * 更新用户
 */
export function update(id: string, data: UserUpdateRequest) {
  return http.put(`${BASE_URL}/${id}`, data)
}

/**
 * 删除用户
 */
export function remove(id: string) {
  return http.delete(`${BASE_URL}/${id}`)
}

/**
 * 重置密码
 */
export function resetPassword(id: string, data: ResetPasswordRequest) {
  return http.post(`${BASE_URL}/${id}/reset-password`, data)
}

/**
 * 分配角色
 */
export function assignRoles(id: string, roleIds: number[]) {
  return http.post(`${BASE_URL}/${id}/roles`, { role_ids: roleIds })
}

/**
 * 启用/禁用用户
 */
export function toggleActive(id: string, isActive: boolean) {
  return http.put(`${BASE_URL}/${id}`, { is_active: isActive })
}

export const usersApi = {
  getList,
  getById,
  create,
  update,
  remove,
  resetPassword,
  assignRoles,
  toggleActive
}

export default usersApi
