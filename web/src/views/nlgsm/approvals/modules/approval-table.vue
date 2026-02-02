<!-- 审批表格 -->
<template>
  <ElTable :data="data" :loading="loading" stripe style="width: 100%">
    <ElTableColumn prop="id" label="ID" width="120">
      <template #default="{ row }">
        <ElButton link type="primary" @click="$emit('view', row)">
          {{ row.id.slice(0, 8) }}...
        </ElButton>
      </template>
    </ElTableColumn>

    <ElTableColumn label="目标类型" width="130">
      <template #default="{ row }">
        <ElTag size="small">{{ getTargetTypeLabel(row.targetType) }}</ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn prop="targetId" label="目标 ID" width="150">
      <template #default="{ row }">
        <ElTooltip :content="row.targetId" placement="top">
          <span class="target-id">{{ row.targetId }}</span>
        </ElTooltip>
      </template>
    </ElTableColumn>

    <ElTableColumn label="风险等级" width="100">
      <template #default="{ row }">
        <ElTag :type="getRiskTagType(row.riskLevel)" size="small">
          {{ getRiskLabel(row.riskLevel) }}
        </ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn label="状态" width="100">
      <template #default="{ row }">
        <ElTag :type="getStatusTagType(row.status)" size="small">
          {{ getStatusLabel(row.status) }}
        </ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn label="审批进度" width="140">
      <template #default="{ row }">
        <div class="progress-cell">
          <ElProgress
            :percentage="getProgressPercentage(row)"
            :stroke-width="6"
            :show-text="false"
          />
          <span class="progress-text">
            {{ row.currentApprovers?.length || 0 }} / {{ row.requiredApprovers }}
          </span>
        </div>
      </template>
    </ElTableColumn>

    <ElTableColumn label="创建时间" width="160">
      <template #default="{ row }">
        {{ formatTime(row.createdAt) }}
      </template>
    </ElTableColumn>

    <ElTableColumn label="过期时间" width="160">
      <template #default="{ row }">
        <span :class="{ expired: isExpired(row) }">
          {{ row.expiresAt ? formatTime(row.expiresAt) : '-' }}
        </span>
      </template>
    </ElTableColumn>

    <ElTableColumn label="操作" width="150" fixed="right">
      <template #default="{ row }">
        <ElButton link type="primary" size="small" @click="$emit('view', row)">
          查看
        </ElButton>
        <template v-if="row.status === 'pending'">
          <ElButton link type="success" size="small" @click="$emit('approve', row)">
            通过
          </ElButton>
          <ElButton link type="danger" size="small" @click="$emit('reject', row)">
            拒绝
          </ElButton>
        </template>
      </template>
    </ElTableColumn>
  </ElTable>
</template>

<script setup lang="ts">
import type { Approval } from '@/types/nlgsm'
import { RiskLevel, riskLabels, ApprovalStatus } from '@/types/nlgsm/enums'

interface Props {
  data: Approval[]
  loading: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'view', row: Approval): void
  (e: 'approve', row: Approval): void
  (e: 'reject', row: Approval): void
}>()

const targetTypeLabels: Record<string, string> = {
  learning_unit: 'Learning Unit',
  artifact: '工件',
  state_transition: '状态转换'
}

const statusLabels: Record<string, string> = {
  pending: '待审批',
  completed: '已完成',
  rejected: '已拒绝',
  expired: '已过期'
}

const getTargetTypeLabel = (type: string) => targetTypeLabels[type] || type
const getStatusLabel = (status: ApprovalStatus) => statusLabels[status] || status
const getRiskLabel = (risk: RiskLevel) => riskLabels[risk] || risk

const getRiskTagType = (risk: RiskLevel) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    low: 'success',
    medium: 'warning',
    high: 'danger',
    critical: 'danger'
  }
  return typeMap[risk] || 'info'
}

const getStatusTagType = (status: ApprovalStatus) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    pending: 'warning',
    completed: 'success',
    rejected: 'danger',
    expired: 'info'
  }
  return typeMap[status] || 'info'
}

const getProgressPercentage = (row: Approval) => {
  if (!row.requiredApprovers) return 0
  return Math.round(((row.currentApprovers?.length || 0) / row.requiredApprovers) * 100)
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const isExpired = (row: Approval) => {
  if (!row.expiresAt) return false
  return new Date(row.expiresAt) < new Date()
}
</script>

<style scoped lang="scss">
.target-id {
  max-width: 130px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: inline-block;
}

.progress-cell {
  display: flex;
  align-items: center;
  gap: 8px;

  .el-progress {
    flex: 1;
  }

  .progress-text {
    font-size: 12px;
    color: var(--el-text-color-secondary);
    white-space: nowrap;
  }
}

.expired {
  color: var(--el-color-danger);
}
</style>
