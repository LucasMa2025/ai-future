<!-- LU 卡片列表 -->
<template>
  <div v-loading="loading" class="lu-card-list">
    <ElRow :gutter="16">
      <ElCol
        v-for="item in data"
        :key="item.id"
        :xs="24"
        :sm="12"
        :md="8"
        :lg="6"
      >
        <ElCard shadow="hover" class="lu-card" @click="$emit('view', item)">
          <div class="card-header">
            <div class="card-id">{{ item.id }}</div>
            <div class="card-badges">
              <ElTag
                v-if="item.riskLevel"
                :type="getRiskTagType(item.riskLevel)"
                size="small"
              >
                {{ getRiskLabel(item.riskLevel) }}
              </ElTag>
              <ElTag :type="getStatusTagType(item.status)" size="small">
                {{ getStatusLabel(item.status) }}
              </ElTag>
            </div>
          </div>

          <div class="card-body">
            <div class="goal-text">{{ item.learningGoal }}</div>
            <div class="meta-info">
              <span>版本 {{ item.version }}</span>
              <span>·</span>
              <span>来源 {{ item.source || '-' }}</span>
            </div>
          </div>

          <div class="card-footer">
            <div class="time">{{ formatTime(item.createdAt) }}</div>
            <div class="actions" @click.stop>
              <template v-if="canApprove(item)">
                <ElButton type="success" size="small" @click="$emit('approve', item)">
                  通过
                </ElButton>
                <ElButton type="danger" size="small" plain @click="$emit('reject', item)">
                  拒绝
                </ElButton>
              </template>
              <ElButton v-else type="primary" size="small" plain @click="$emit('view', item)">
                查看
              </ElButton>
            </div>
          </div>
        </ElCard>
      </ElCol>
    </ElRow>

    <ElEmpty v-if="!loading && data.length === 0" description="暂无数据" />
  </div>
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
  const date = new Date(time)
  return date.toLocaleDateString('zh-CN')
}

const canApprove = (row: LearningUnit) => {
  return [LearningUnitStatus.AUTO_CLASSIFIED, LearningUnitStatus.HUMAN_REVIEW].includes(row.status)
}
</script>

<style scoped lang="scss">
.lu-card-list {
  min-height: 200px;
}

.lu-card {
  margin-bottom: 16px;
  cursor: pointer;
  transition: all 0.3s;

  &:hover {
    transform: translateY(-2px);
  }

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;

    .card-id {
      font-size: 13px;
      font-weight: 600;
      color: var(--el-color-primary);
    }

    .card-badges {
      display: flex;
      gap: 4px;
    }
  }

  .card-body {
    .goal-text {
      font-size: 14px;
      color: var(--el-text-color-primary);
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
      text-overflow: ellipsis;
      min-height: 42px;
      margin-bottom: 8px;
    }

    .meta-info {
      font-size: 12px;
      color: var(--el-text-color-secondary);

      span {
        margin-right: 4px;
      }
    }
  }

  .card-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--el-border-color-lighter);

    .time {
      font-size: 12px;
      color: var(--el-text-color-secondary);
    }

    .actions {
      display: flex;
      gap: 8px;
    }
  }
}
</style>
