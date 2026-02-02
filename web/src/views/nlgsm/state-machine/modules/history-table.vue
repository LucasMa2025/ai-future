<!-- 状态转换历史表格 -->
<template>
  <ElCard shadow="hover" class="history-card">
    <template #header>
      <div class="card-header">
        <span>转换历史</span>
        <ElButton :icon="Refresh" size="small" @click="$emit('refresh')">刷新</ElButton>
      </div>
    </template>

    <ElTable
      :data="history"
      :loading="loading"
      stripe
      style="width: 100%"
      max-height="400"
    >
      <ElTableColumn label="时间" width="160">
        <template #default="{ row }">
          {{ formatTime(row.createdAt) }}
        </template>
      </ElTableColumn>

      <ElTableColumn label="转换" min-width="180">
        <template #default="{ row }">
          <div class="transition-cell">
            <ElTag :color="getStateColor(row.fromState)" effect="dark" size="small">
              {{ getStateLabel(row.fromState) }}
            </ElTag>
            <ElIcon class="arrow-icon"><ArrowRight /></ElIcon>
            <ElTag :color="getStateColor(row.toState)" effect="dark" size="small">
              {{ getStateLabel(row.toState) }}
            </ElTag>
          </div>
        </template>
      </ElTableColumn>

      <ElTableColumn prop="triggerEvent" label="触发事件" min-width="140">
        <template #default="{ row }">
          <ElTooltip :content="row.triggerEvent" placement="top">
            <span class="event-text">{{ row.triggerEvent }}</span>
          </ElTooltip>
        </template>
      </ElTableColumn>

      <ElTableColumn prop="triggerSource" label="来源" width="100">
        <template #default="{ row }">
          {{ row.triggerSource || '-' }}
        </template>
      </ElTableColumn>

      <ElTableColumn label="决策" width="80">
        <template #default="{ row }">
          <ElTag :type="getDecisionTagType(row.decision)" size="small">
            {{ row.decision }}
          </ElTag>
        </template>
      </ElTableColumn>

      <ElTableColumn label="状态" width="80">
        <template #default="{ row }">
          <ElTag :type="row.success ? 'success' : 'danger'" size="small">
            {{ row.success ? '成功' : '失败' }}
          </ElTag>
        </template>
      </ElTableColumn>

      <ElTableColumn label="耗时" width="80">
        <template #default="{ row }">
          {{ row.durationMs }}ms
        </template>
      </ElTableColumn>

      <ElTableColumn label="操作" width="80" fixed="right">
        <template #default="{ row }">
          <ElButton link type="primary" size="small" @click="showDetail(row)">
            详情
          </ElButton>
        </template>
      </ElTableColumn>
    </ElTable>

    <!-- 分页 -->
    <div class="pagination-wrapper">
      <ElPagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next"
        @size-change="handleSizeChange"
        @current-change="handlePageChange"
      />
    </div>

    <!-- 详情对话框 -->
    <ElDialog v-model="detailVisible" title="转换详情" width="600px">
      <ElDescriptions v-if="selectedRow" :column="2" border>
        <ElDescriptionsItem label="转换 ID" :span="2">
          {{ selectedRow.id }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="原状态">
          {{ getStateLabel(selectedRow.fromState) }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="目标状态">
          {{ getStateLabel(selectedRow.toState) }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="触发事件" :span="2">
          {{ selectedRow.triggerEvent }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="触发来源">
          {{ selectedRow.triggerSource || '-' }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="决策">
          {{ selectedRow.decision }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="决策原因" :span="2">
          {{ selectedRow.decisionReason || '-' }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="执行动作" :span="2">
          <ElTag
            v-for="action in selectedRow.actionsExecuted"
            :key="action"
            size="small"
            class="action-tag"
          >
            {{ action }}
          </ElTag>
          <span v-if="selectedRow.actionsExecuted.length === 0">-</span>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="执行状态">
          <ElTag :type="selectedRow.success ? 'success' : 'danger'">
            {{ selectedRow.success ? '成功' : '失败' }}
          </ElTag>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="执行耗时">
          {{ selectedRow.durationMs }}ms
        </ElDescriptionsItem>
        <ElDescriptionsItem v-if="selectedRow.errorMessage" label="错误信息" :span="2">
          <span class="error-text">{{ selectedRow.errorMessage }}</span>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="创建时间" :span="2">
          {{ formatTime(selectedRow.createdAt) }}
        </ElDescriptionsItem>
      </ElDescriptions>
    </ElDialog>
  </ElCard>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Refresh, ArrowRight } from '@element-plus/icons-vue'
import type { StateTransition } from '@/types/nlgsm'
import { NLGSMState, stateLabels, stateColors } from '@/types/nlgsm/enums'

interface Props {
  history: StateTransition[]
  total: number
  loading: boolean
}

defineProps<Props>()

const emit = defineEmits<{
  (e: 'refresh'): void
  (e: 'page-change', page: number, pageSize: number): void
}>()

const currentPage = ref(1)
const pageSize = ref(20)
const detailVisible = ref(false)
const selectedRow = ref<StateTransition | null>(null)

const getStateLabel = (state: string) => {
  return stateLabels[state as NLGSMState] || state
}

const getStateColor = (state: string) => {
  return stateColors[state as NLGSMState] || '#9e9e9e'
}

const getDecisionTagType = (decision: string) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    allow: 'success',
    deny: 'danger',
    rollback: 'warning',
    halt: 'danger',
    diagnose: 'info'
  }
  return typeMap[decision] || 'info'
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const showDetail = (row: StateTransition) => {
  selectedRow.value = row
  detailVisible.value = true
}

const handleSizeChange = (size: number) => {
  pageSize.value = size
  emit('page-change', currentPage.value, size)
}

const handlePageChange = (page: number) => {
  currentPage.value = page
  emit('page-change', page, pageSize.value)
}
</script>

<style scoped lang="scss">
.history-card {
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.transition-cell {
  display: flex;
  align-items: center;
  gap: 6px;

  .arrow-icon {
    color: var(--el-text-color-secondary);
  }
}

.event-text {
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: inline-block;
}

.pagination-wrapper {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.action-tag {
  margin-right: 4px;
  margin-bottom: 4px;
}

.error-text {
  color: var(--el-color-danger);
}
</style>
