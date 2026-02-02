/**
 * 系统管理模块枚举定义
 */

// 权限操作类型
export enum PermissionAction {
  ACCESS = 'access',
  CREATE = 'create',
  READ = 'read',
  UPDATE = 'update',
  DELETE = 'delete',
  APPROVE = 'approve',
  EXPORT = 'export'
}

export const PermissionActionLabels: Record<PermissionAction, string> = {
  [PermissionAction.ACCESS]: '访问',
  [PermissionAction.CREATE]: '创建',
  [PermissionAction.READ]: '查看',
  [PermissionAction.UPDATE]: '修改',
  [PermissionAction.DELETE]: '删除',
  [PermissionAction.APPROVE]: '审批',
  [PermissionAction.EXPORT]: '导出'
}

// 权限授权类型
export enum GrantType {
  ALLOW = 'allow',
  DENY = 'deny'
}

export const GrantTypeLabels: Record<GrantType, string> = {
  [GrantType.ALLOW]: '允许',
  [GrantType.DENY]: '拒绝'
}

// 功能层级
export enum FunctionLevel {
  MODULE = 1,
  FEATURE_GROUP = 2,
  API_ENDPOINT = 3
}

export const FunctionLevelLabels: Record<FunctionLevel, string> = {
  [FunctionLevel.MODULE]: '模块',
  [FunctionLevel.FEATURE_GROUP]: '功能组',
  [FunctionLevel.API_ENDPOINT]: 'API端点'
}

// 审计级别
export enum AuditLevel {
  NONE = 'none',
  INFO = 'info',
  NORMAL = 'normal',
  IMPORTANT = 'important',
  CRITICAL = 'critical'
}

export const AuditLevelLabels: Record<AuditLevel, string> = {
  [AuditLevel.NONE]: '不审计',
  [AuditLevel.INFO]: '信息',
  [AuditLevel.NORMAL]: '普通',
  [AuditLevel.IMPORTANT]: '重要',
  [AuditLevel.CRITICAL]: '关键'
}

// HTTP 方法
export enum HttpMethod {
  GET = 'GET',
  POST = 'POST',
  PUT = 'PUT',
  DELETE = 'DELETE',
  PATCH = 'PATCH'
}

export const HttpMethodColors: Record<HttpMethod, string> = {
  [HttpMethod.GET]: '#67c23a',
  [HttpMethod.POST]: '#409eff',
  [HttpMethod.PUT]: '#e6a23c',
  [HttpMethod.DELETE]: '#f56c6c',
  [HttpMethod.PATCH]: '#909399'
}

// 备份状态
export enum BackupStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export const BackupStatusLabels: Record<BackupStatus, string> = {
  [BackupStatus.PENDING]: '等待中',
  [BackupStatus.RUNNING]: '执行中',
  [BackupStatus.COMPLETED]: '已完成',
  [BackupStatus.FAILED]: '失败'
}

export const BackupStatusColors: Record<BackupStatus, string> = {
  [BackupStatus.PENDING]: 'info',
  [BackupStatus.RUNNING]: 'warning',
  [BackupStatus.COMPLETED]: 'success',
  [BackupStatus.FAILED]: 'danger'
}

// 备份类型
export enum BackupType {
  FULL = 'full',
  TABLES = 'tables'
}

export const BackupTypeLabels: Record<BackupType, string> = {
  [BackupType.FULL]: '完整备份',
  [BackupType.TABLES]: '指定表备份'
}

// 配置值类型
export enum ConfigValueType {
  STRING = 'string',
  NUMBER = 'number',
  BOOLEAN = 'boolean',
  JSON = 'json'
}

export const ConfigValueTypeLabels: Record<ConfigValueType, string> = {
  [ConfigValueType.STRING]: '字符串',
  [ConfigValueType.NUMBER]: '数字',
  [ConfigValueType.BOOLEAN]: '布尔值',
  [ConfigValueType.JSON]: 'JSON'
}
