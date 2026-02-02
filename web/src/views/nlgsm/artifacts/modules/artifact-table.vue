<!-- 工件表格 -->
<template>
  <ElTable :data="data" :loading="loading" stripe style="width: 100%">
    <ElTableColumn prop="id" label="ID" width="150">
      <template #default="{ row }">
        <ElButton link type="primary" @click="$emit('view', row)">
          {{ row.id }}
        </ElButton>
      </template>
    </ElTableColumn>

    <ElTableColumn prop="version" label="版本" width="80">
      <template #default="{ row }">
        v{{ row.version }}
      </template>
    </ElTableColumn>

    <ElTableColumn label="NLGSM 状态" width="120">
      <template #default="{ row }">
        <ElTag :color="getStateColor(row.nlState)" effect="dark" size="small">
          {{ getStateLabel(row.nlState) }}
        </ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn label="风险分数" width="120">
      <template #default="{ row }">
        <div class="risk-score-cell">
          <ElProgress
            :percentage="row.riskScore * 100"
            :stroke-width="8"
            :color="getRiskScoreColor(row.riskScore)"
            :show-text="false"
          />
          <span class="risk-value">{{ (row.riskScore * 100).toFixed(0) }}%</span>
        </div>
      </template>
    </ElTableColumn>

    <ElTableColumn label="审批状态" width="100">
      <template #default="{ row }">
        <ElTag :type="row.isApproved ? 'success' : 'warning'" size="small">
          {{ row.isApproved ? '已审批' : '待审批' }}
        </ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn label="审批人" min-width="150">
      <template #default="{ row }">
        <template v-if="row.approvers?.length">
          <ElTag
            v-for="approver in row.approvers.slice(0, 3)"
            :key="approver"
            size="small"
            class="approver-tag"
          >
            {{ approver }}
          </ElTag>
          <span v-if="row.approvers.length > 3" class="more-text">
            +{{ row.approvers.length - 3 }}
          </span>
        </template>
        <span v-else class="text-secondary">-</span>
      </template>
    </ElTableColumn>

    <ElTableColumn label="创建时间" width="160">
      <template #default="{ row }">
        {{ formatTime(row.createdAt) }}
      </template>
    </ElTableColumn>

    <ElTableColumn label="操作" width="140" fixed="right">
      <template #default="{ row }">
        <ElButton link type="primary" size="small" @click="$emit('view', row)">
          查看
        </ElButton>
        <ElButton link type="warning" size="small" @click="$emit('rollback', row)">
          回滚
        </ElButton>
      </template>
    </ElTableColumn>
  </ElTable>
</template>

<script setup lang="ts">
import type { Artifact } from '@/types/nlgsm'
import { NLGSMState, stateLabels, stateColors } from '@/types/nlgsm/enums'

interface Props {
  data: Artifact[]
  loading: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'view', row: Artifact): void
  (e: 'rollback', row: Artifact): void
}>()

const getStateLabel = (state: NLGSMState) => stateLabels[state] || state
const getStateColor = (state: NLGSMState) => stateColors[state] || '#9e9e9e'

const getRiskScoreColor = (score: number) => {
  if (score < 0.3) return '#4caf50'
  if (score < 0.6) return '#ff9800'
  return '#f44336'
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}
</script>

<style scoped lang="scss">
.risk-score-cell {
  display: flex;
  align-items: center;
  gap: 8px;

  .el-progress {
    flex: 1;
  }

  .risk-value {
    font-size: 12px;
    font-weight: 500;
    white-space: nowrap;
  }
}

.approver-tag {
  margin-right: 4px;
}

.more-text {
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.text-secondary {
  color: var(--el-text-color-secondary);
}
</style>
