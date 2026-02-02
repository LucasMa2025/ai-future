/**
 * 系统管理模块数据模型
 */

import type {
  PermissionAction,
  GrantType,
  AuditLevel,
  HttpMethod,
  BackupStatus,
  BackupType,
  ConfigValueType
} from './enums'

// ==================== 用户相关 ====================

export interface User {
  id: string
  username: string
  email: string
  full_name?: string
  phone?: string
  avatar?: string
  is_active: boolean
  is_superuser: boolean
  last_login?: string
  login_count: number
  created_at: string
  updated_at: string
  roles: Role[]
}

export interface UserCreateRequest {
  username: string
  email: string
  password: string
  full_name?: string
  phone?: string
  is_active?: boolean
  is_superuser?: boolean
  role_ids?: number[]
}

export interface UserUpdateRequest {
  email?: string
  full_name?: string
  phone?: string
  avatar?: string
  is_active?: boolean
  role_ids?: number[]
}

export interface ResetPasswordRequest {
  new_password: string
}

// ==================== 角色相关 ====================

export interface Role {
  id: number
  name: string
  display_name: string
  description?: string
  risk_level_limit?: string
  is_system: boolean
  sort_order: number
  created_at: string
  permissions?: Permission[]
}

export interface RoleCreateRequest {
  name: string
  display_name: string
  description?: string
  risk_level_limit?: string
  sort_order?: number
  permission_ids?: number[]
}

export interface RoleUpdateRequest {
  display_name?: string
  description?: string
  risk_level_limit?: string
  sort_order?: number
}

// ==================== 权限相关 ====================

export interface Permission {
  id: number
  code: string
  name: string
  description?: string
  function_id?: number
  action: PermissionAction
  resource_type?: string
  resource_scope: string
  is_enabled: boolean
}

export interface PermissionCreateRequest {
  code: string
  name: string
  function_id?: number
  action?: string
  description?: string
  resource_type?: string
  resource_scope?: string
}

export interface UserPermission {
  id: number
  permission_id: number
  permission_code: string
  permission_name: string
  grant_type: GrantType
  expires_at?: string
  constraints?: Record<string, any>
}

export interface EffectivePermissions {
  is_superuser: boolean
  permissions: Record<
    string,
    {
      allowed: boolean
      source: string
    }
  >
}

// ==================== 系统功能相关 ====================

export interface SystemFunction {
  id: number
  code: string
  name: string
  description?: string
  level: number
  parent_id?: number
  module?: string
  method?: HttpMethod
  api_path?: string
  icon?: string
  is_visible: boolean
  is_audited: boolean
  audit_level: AuditLevel
  sort_order: number
  is_enabled: boolean
  children?: SystemFunction[]
  permissions?: PermissionItem[]
}

export interface PermissionItem {
  id: number
  code: string
  name: string
  action: string
  checked: boolean
}

export interface SystemFunctionCreateRequest {
  code: string
  name: string
  level?: number
  parent_id?: number
  description?: string
  module?: string
  method?: string
  api_path?: string
  icon?: string
  is_visible?: boolean
  is_audited?: boolean
  audit_level?: string
  sort_order?: number
  is_enabled?: boolean
}

export interface SystemFunctionUpdateRequest {
  name?: string
  description?: string
  icon?: string
  is_visible?: boolean
  is_audited?: boolean
  audit_level?: string
  sort_order?: number
  is_enabled?: boolean
}

// ==================== 操作日志相关 ====================

export interface OperationLog {
  id: number
  user_id?: string
  username?: string
  request_id?: string
  method: string
  path: string
  query_params?: Record<string, any>
  ip_address?: string
  user_agent?: string
  status_code?: number
  response_time_ms?: number
  function_code?: string
  is_success: boolean
  error_message?: string
  created_at: string
}

export interface OperationLogStatistics {
  total_requests: number
  success_count: number
  failure_count: number
  success_rate: number
  avg_response_time_ms: number
  by_method: Record<string, number>
  by_user: Array<{ username: string; count: number }>
  by_day: Array<{ date: string; count: number }>
}

export interface OperationLogQuery {
  page?: number
  page_size?: number
  user_id?: string
  username?: string
  method?: string
  path?: string
  function_code?: string
  is_success?: boolean
  start_date?: string
  end_date?: string
  ip_address?: string
}

// ==================== 数据备份相关 ====================

export interface DataBackup {
  id: number
  backup_name: string
  backup_type: BackupType
  tables: string[]
  file_path?: string
  file_size: number
  file_size_display: string
  compressed: boolean
  status: BackupStatus
  progress: number
  started_at?: string
  completed_at?: string
  duration_seconds?: number
  created_by?: string
  error_message?: string
  record_counts?: Record<string, number>
  description?: string
  created_at: string
}

export interface BackupCreateRequest {
  backup_name: string
  backup_type?: BackupType
  tables?: string[]
  compress?: boolean
  description?: string
}

export interface BackupRestoreRequest {
  tables?: string[]
}

export interface BackupStatistics {
  total_backups: number
  completed_backups: number
  failed_backups: number
  total_storage_size: number
  total_storage_display: string
  latest_backup?: DataBackup
  available_tables: string[]
}

export interface TableInfo {
  table_name: string
  record_count: number
  error?: string
}

// ==================== 系统配置相关 ====================

export interface SystemConfig {
  id: number
  config_key: string
  config_value: string
  value_type: ConfigValueType
  config_group?: string
  display_name?: string
  description?: string
  is_readonly: boolean
  is_secret: boolean
  default_value?: string
  updated_at?: string
  updated_by?: string
}

export interface ConfigGroup {
  key: string
  name: string
  icon: string
  config_count: number
}

export interface ConfigCreateRequest {
  config_key: string
  config_value: string
  value_type?: string
  config_group?: string
  display_name?: string
  description?: string
  is_readonly?: boolean
  is_secret?: boolean
  default_value?: string
}

export interface ConfigUpdateRequest {
  value: any
}

export interface BatchConfigUpdateRequest {
  configs: Record<string, any>
}
