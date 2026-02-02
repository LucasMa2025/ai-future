<!-- 事件时间线 -->
<template>
  <ElCard shadow="hover" class="timeline-card">
    <template #header>
      <div class="card-header">
        <span>最近事件</span>
        <ElButton link type="primary" @click="goToAudit">
          查看全部
          <ElIcon class="el-icon--right"><ArrowRight /></ElIcon>
        </ElButton>
      </div>
    </template>

    <ElSkeleton :loading="loading" animated :rows="10">
      <template #default>
        <div class="timeline-content">
          <ElTimeline v-if="events.length > 0">
            <ElTimelineItem
              v-for="event in events"
              :key="event.id"
              :timestamp="formatTime(event.timestamp)"
              :type="getTimelineType(event.severity)"
              :hollow="true"
              placement="top"
            >
              <div class="timeline-item-content">
                <div class="event-header">
                  <ElTag :type="getTagType(event.type)" size="small" effect="plain">
                    {{ getEventTypeLabel(event.type) }}
                  </ElTag>
                  <span class="event-title">{{ event.title }}</span>
                </div>
                <div class="event-description">{{ event.description }}</div>
              </div>
            </ElTimelineItem>
          </ElTimeline>

          <ElEmpty v-else description="暂无事件" :image-size="80" />
        </div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { ArrowRight } from '@element-plus/icons-vue'
import type { TimelineEvent } from '@/types/nlgsm'

interface Props {
  events: TimelineEvent[]
  loading: boolean
}

defineProps<Props>()
const router = useRouter()

const formatTime = (time: string) => {
  if (!time) return ''
  const date = new Date(time)
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  // 小于1小时显示分钟
  if (diff < 3600000) {
    const minutes = Math.floor(diff / 60000)
    return `${minutes} 分钟前`
  }
  // 小于24小时显示小时
  if (diff < 86400000) {
    const hours = Math.floor(diff / 3600000)
    return `${hours} 小时前`
  }
  // 否则显示日期
  return date.toLocaleDateString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const getTimelineType = (severity?: string) => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'danger' | 'info'> = {
    info: 'primary',
    success: 'success',
    warning: 'warning',
    error: 'danger'
  }
  return typeMap[severity || 'info'] || 'primary'
}

const getTagType = (type: string) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    state_change: '',
    approval: 'success',
    anomaly: 'danger',
    learning: 'warning',
    checkpoint: 'info'
  }
  return typeMap[type] || 'info'
}

const getEventTypeLabel = (type: string) => {
  const labelMap: Record<string, string> = {
    state_change: '状态变更',
    approval: '审批',
    anomaly: '异常',
    learning: '学习',
    checkpoint: '检查点'
  }
  return labelMap[type] || type
}

const goToAudit = () => {
  router.push({ name: 'AuditLogs' })
}
</script>

<style scoped lang="scss">
.timeline-card {
  height: 100%;
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.timeline-content {
  max-height: 400px;
  overflow-y: auto;

  .timeline-item-content {
    .event-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 4px;

      .event-title {
        font-size: 14px;
        font-weight: 500;
        color: var(--el-text-color-primary);
      }
    }

    .event-description {
      font-size: 13px;
      color: var(--el-text-color-secondary);
      line-height: 1.5;
    }
  }
}
</style>
