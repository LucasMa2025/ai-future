/**
 * 角色管理 API 客户端
 */

import http from '@/utils/http'
import type { Role, RoleCreateRequest, RoleUpdateRequest } from '@/types/system'

const BASE_URL = '/api/v1/roles'

/**
 * 获取角色列表
 */
export function getList(params?: { page?: number; page_size?: number; name?: string }) {
  return http.get<{ data: Role[] }>(`${BASE_URL}/`, { params })
}

/**
 * 获取角色详情
 */
export function getById(id: number) {
  return http.get<{ data: Role }>(`${BASE_URL}/${id}`)
}

/**
 * 创建角色
 */
export function create(data: RoleCreateRequest) {
  return http.post<{ data: { id: number } }>(`${BASE_URL}/`, data)
}

/**
 * 更新角色
 */
export function update(id: number, data: RoleUpdateRequest) {
  return http.put(`${BASE_URL}/${id}`, data)
}

/**
 * 删除角色
 */
export function remove(id: number) {
  return http.delete(`${BASE_URL}/${id}`)
}

/**
 * 获取角色的用户列表
 */
export function getUsers(id: number) {
  return http.get<{ data: { id: string; username: string; full_name?: string }[] }>(
    `${BASE_URL}/${id}/users`
  )
}

export const rolesApi = {
  getList,
  getById,
  create,
  update,
  remove,
  getUsers
}

export default rolesApi
