<!-- LU 表格 -->
<template>
  <ElTable :data="data" :loading="loading" stripe style="width: 100%">
    <ElTableColumn prop="id" label="ID" width="150">
      <template #default="{ row }">
        <ElButton link type="primary" @click="$emit('view', row)">
          {{ row.id }}
        </ElButton>
      </template>
    </ElTableColumn>

    <ElTableColumn label="学习目标" min-width="200">
      <template #default="{ row }">
        <ElTooltip :content="row.learningGoal" placement="top">
          <span class="goal-text">{{ row.learningGoal }}</span>
        </ElTooltip>
      </template>
    </ElTableColumn>

    <ElTableColumn label="风险等级" width="100">
      <template #default="{ row }">
        <ElTag v-if="row.riskLevel" :type="getRiskTagType(row.riskLevel)" size="small">
          {{ getRiskLabel(row.riskLevel) }}
        </ElTag>
        <span v-else class="text-secondary">-</span>
      </template>
    </ElTableColumn>

    <ElTableColumn label="状态" width="100">
      <template #default="{ row }">
        <ElTag :type="getStatusTagType(row.status)" size="small">
          {{ getStatusLabel(row.status) }}
        </ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn prop="version" label="版本" width="70" />

    <ElTableColumn label="来源" width="100">
      <template #default="{ row }">
        {{ row.source || '-' }}
      </template>
    </ElTableColumn>

    <ElTableColumn label="链深度" width="80">
      <template #default="{ row }">
        {{ row.provenance?.chainDepth || 0 }}
      </template>
    </ElTableColumn>

    <ElTableColumn label="创建时间" width="160">
      <template #default="{ row }">
        {{ formatTime(row.createdAt) }}
      </template>
    </ElTableColumn>

    <ElTableColumn label="操作" width="160" fixed="right">
      <template #default="{ row }">
        <ElButton link type="primary" size="small" @click="$emit('view', row)">
          查看
        </ElButton>
        <template v-if="canApprove(row)">
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
import type { LearningUnit } from '@/types/nlgsm'
import { LearningUnitStatus, RiskLevel, riskLabels } from '@/types/nlgsm/enums'

interface Props {
  data: LearningUnit[]
  loading: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'view', row: LearningUnit): void
  (e: 'approve', row: LearningUnit): void
  (e: 'reject', row: LearningUnit): void
}>()

const statusLabels: Record<string, string> = {
  [LearningUnitStatus.PENDING]: '待处理',
  [LearningUnitStatus.AUTO_CLASSIFIED]: '已分类',
  [LearningUnitStatus.HUMAN_REVIEW]: '人工审核',
  [LearningUnitStatus.APPROVED]: '已通过',
  [LearningUnitStatus.CORRECTED]: '已修正',
  [LearningUnitStatus.REJECTED]: '已拒绝',
  [LearningUnitStatus.TERMINATED]: '已终止'
}

const getRiskLabel = (risk: RiskLevel) => riskLabels[risk] || risk

const getStatusLabel = (status: LearningUnitStatus) => statusLabels[status] || status

const getRiskTagType = (risk: RiskLevel) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    [RiskLevel.LOW]: 'success',
    [RiskLevel.MEDIUM]: 'warning',
    [RiskLevel.HIGH]: 'danger',
    [RiskLevel.CRITICAL]: 'danger'
  }
  return typeMap[risk] || 'info'
}

const getStatusTagType = (status: LearningUnitStatus) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    [LearningUnitStatus.PENDING]: 'info',
    [LearningUnitStatus.AUTO_CLASSIFIED]: '',
    [LearningUnitStatus.HUMAN_REVIEW]: 'warning',
    [LearningUnitStatus.APPROVED]: 'success',
    [LearningUnitStatus.CORRECTED]: 'success',
    [LearningUnitStatus.REJECTED]: 'danger',
    [LearningUnitStatus.TERMINATED]: 'info'
  }
  return typeMap[status] || 'info'
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const canApprove = (row: LearningUnit) => {
  return [LearningUnitStatus.AUTO_CLASSIFIED, LearningUnitStatus.HUMAN_REVIEW].includes(row.status)
}
</script>

<style scoped lang="scss">
.goal-text {
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}

.text-secondary {
  color: var(--el-text-color-secondary);
}
</style>
