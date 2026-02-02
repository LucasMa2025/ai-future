<!-- 审计表格 -->
<template>
  <ElTable :data="data" :loading="loading" stripe style="width: 100%">
    <ElTableColumn prop="timestamp" label="时间" width="170">
      <template #default="{ row }">
        {{ formatTime(row.timestamp) }}
      </template>
    </ElTableColumn>

    <ElTableColumn prop="action" label="操作" width="150">
      <template #default="{ row }">
        <ElTag size="small">{{ row.action }}</ElTag>
      </template>
    </ElTableColumn>

    <ElTableColumn prop="actor" label="操作者" width="120" />

    <ElTableColumn prop="targetType" label="目标类型" width="120" />

    <ElTableColumn prop="targetId" label="目标 ID" width="150">
      <template #default="{ row }">
        <ElTooltip :content="row.targetId" placement="top">
          <span class="target-id">{{ row.targetId }}</span>
        </ElTooltip>
      </template>
    </ElTableColumn>

    <ElTableColumn label="详情" min-width="200">
      <template #default="{ row }">
        <span class="details-text">{{ formatDetails(row.details) }}</span>
      </template>
    </ElTableColumn>

    <ElTableColumn prop="ipAddress" label="IP 地址" width="130" />

    <ElTableColumn label="操作" width="120" fixed="right">
      <template #default="{ row }">
        <ElButton link type="primary" size="small" @click="$emit('view', row)">
          查看
        </ElButton>
        <ElButton link type="success" size="small" @click="$emit('verify', row)">
          验证
        </ElButton>
      </template>
    </ElTableColumn>
  </ElTable>
</template>

<script setup lang="ts">
import type { AuditLog } from '@/types/nlgsm'

interface Props {
  data: AuditLog[]
  loading: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'view', row: AuditLog): void
  (e: 'verify', row: AuditLog): void
}>()

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const formatDetails = (details: Record<string, any>) => {
  if (!details || Object.keys(details).length === 0) return '-'
  const entries = Object.entries(details).slice(0, 3)
  return entries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(', ')
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

.details-text {
  display: -webkit-box;
  -webkit-line-clamp: 1;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}
</style>
