/**
 * 权限管理 API 客户端
 */

import http from '@/utils/http'
import type {
  Permission,
  PermissionCreateRequest,
  UserPermission,
  EffectivePermissions,
  SystemFunction
} from '@/types/system'

const BASE_URL = '/api/v1/permissions'

/**
 * 获取所有权限
 */
export function getList(includeDisabled = false) {
  return http.get<{ data: Permission[] }>(`${BASE_URL}/`, {
    params: { include_disabled: includeDisabled }
  })
}

/**
 * 获取权限树（带授权状态）
 */
export function getTree(roleId?: number, userId?: string) {
  return http.get<{ data: SystemFunction[] }>(`${BASE_URL}/tree`, {
    params: { role_id: roleId, user_id: userId }
  })
}

/**
 * 创建权限
 */
export function create(data: PermissionCreateRequest) {
  return http.post<{ data: { id: number } }>(`${BASE_URL}/`, data)
}

// ==================== 角色权限 ====================

/**
 * 获取角色权限
 */
export function getRolePermissions(roleId: number) {
  return http.get<{ data: Permission[] }>(`${BASE_URL}/role/${roleId}`)
}

/**
 * 设置角色权限
 */
export function setRolePermissions(roleId: number, permissionIds: number[]) {
  return http.put(`${BASE_URL}/role/${roleId}`, { permission_ids: permissionIds })
}

// ==================== 用户权限 ====================

/**
 * 获取用户直接权限
 */
export function getUserPermissions(userId: string) {
  return http.get<{ data: UserPermission[] }>(`${BASE_URL}/user/${userId}`)
}

/**
 * 获取用户有效权限
 */
export function getUserEffectivePermissions(userId: string) {
  return http.get<{ data: EffectivePermissions }>(`${BASE_URL}/user/${userId}/effective`)
}

/**
 * 设置用户权限
 */
export function setUserPermissions(
  userId: string,
  permissions: Array<{
    permission_id: number
    grant_type?: string
  }>
) {
  return http.put(`${BASE_URL}/user/${userId}`, { permissions })
}

// ==================== 权限检查 ====================

/**
 * 检查权限
 */
export function checkPermission(permissionCode: string) {
  return http.post<{ data: { permission_code: string; allowed: boolean } }>(`${BASE_URL}/check`, {
    permission_code: permissionCode
  })
}

/**
 * 获取我的权限
 */
export function getMyPermissions() {
  return http.get<{ data: EffectivePermissions }>(`${BASE_URL}/my`)
}

export const permissionsApi = {
  getList,
  getTree,
  create,
  getRolePermissions,
  setRolePermissions,
  getUserPermissions,
  getUserEffectivePermissions,
  setUserPermissions,
  checkPermission,
  getMyPermissions
}

export default permissionsApi
