<!-- 审计详情对话框 -->
<template>
  <ElDialog
    :model-value="modelValue"
    title="审计日志详情"
    width="700px"
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <template v-if="log">
      <ElDescriptions :column="2" border>
        <ElDescriptionsItem label="日志 ID" :span="2">
          {{ log.id }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="操作">
          <ElTag>{{ log.action }}</ElTag>
        </ElDescriptionsItem>

        <ElDescriptionsItem label="操作者">
          {{ log.actor }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="目标类型">
          {{ log.targetType }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="目标 ID">
          {{ log.targetId }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="时间" :span="2">
          {{ formatTime(log.timestamp) }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="IP 地址">
          {{ log.ipAddress || '-' }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="User Agent">
          <ElTooltip :content="log.userAgent" placement="top">
            <span class="ua-text">{{ log.userAgent || '-' }}</span>
          </ElTooltip>
        </ElDescriptionsItem>

        <ElDescriptionsItem label="完整性哈希" :span="2">
          <code class="hash-code">{{ log.integrityHash }}</code>
        </ElDescriptionsItem>
      </ElDescriptions>

      <!-- 详细内容 -->
      <div class="details-section">
        <h4>操作详情</h4>
        <pre>{{ JSON.stringify(log.details, null, 2) }}</pre>
      </div>
    </template>

    <template #footer>
      <ElButton @click="$emit('update:modelValue', false)">关闭</ElButton>
    </template>
  </ElDialog>
</template>

<script setup lang="ts">
import type { AuditLog } from '@/types/nlgsm'

interface Props {
  modelValue: boolean
  log: AuditLog | null
}

defineProps<Props>()

defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}
</script>

<style scoped lang="scss">
.ua-text {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  display: inline-block;
}

.hash-code {
  font-size: 12px;
  word-break: break-all;
}

.details-section {
  margin-top: 20px;

  h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
  }

  pre {
    margin: 0;
    padding: 16px;
    background: var(--el-fill-color-light);
    border-radius: 8px;
    font-size: 12px;
    max-height: 300px;
    overflow: auto;
  }
}
</style>
