<!-- 事件时间线 -->
<template>
  <ElCard shadow="hover" class="event-timeline-card">
    <template #header>
      <div class="card-header">
        <span>事件时间线</span>
        <ElTag size="small">共 {{ totalEvents }} 个事件</ElTag>
      </div>
    </template>

    <div class="timeline-content">
      <ElTimeline>
        <ElTimelineItem
          v-for="(event, index) in sortedEvents"
          :key="index"
          :timestamp="formatTime(event.timestamp)"
          :type="getEventType(event)"
          :hollow="event.type === 'checkpoint' || event.type === 'direction_change'"
        >
          <div class="event-item">
            <div class="event-header">
              <ElTag :type="getEventTagType(event)" size="small">
                {{ getEventTypeLabel(event) }}
              </ElTag>
              <span v-if="event.checkpointId" class="checkpoint-id">
                {{ event.checkpointId.slice(0, 8) }}...
              </span>
            </div>
            <div class="event-description">{{ event.description || event.reason || '-' }}</div>
          </div>
        </ElTimelineItem>
      </ElTimeline>
    </div>
  </ElCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { TimelineEventItem, CheckpointMarker, DirectionChange } from '@/types/nlgsm'

interface Props {
  events: TimelineEventItem[]
  checkpoints: CheckpointMarker[]
  directionChanges: DirectionChange[]
}

const props = defineProps<Props>()

// 合并所有事件并排序
const sortedEvents = computed(() => {
  const allEvents: Array<TimelineEventItem & { type: string }> = []

  // 添加普通事件
  props.events.forEach((e) => {
    allEvents.push({ ...e })
  })

  // 添加检查点事件
  props.checkpoints.forEach((cp) => {
    allEvents.push({
      type: 'checkpoint',
      timestamp: cp.timestamp,
      description: cp.reason,
      checkpointId: cp.checkpointId
    })
  })

  // 添加方向变更事件
  props.directionChanges.forEach((dc) => {
    allEvents.push({
      type: 'direction_change',
      timestamp: dc.timestamp,
      description: `${dc.oldDirection} → ${dc.newDirection}`,
      reason: dc.reason
    })
  })

  // 按时间排序
  return allEvents.sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
  )
})

const totalEvents = computed(() => sortedEvents.value.length)

const formatTime = (time: string) => {
  if (!time) return ''
  return new Date(time).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

const getEventType = (event: any) => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'danger' | 'info'> = {
    checkpoint: 'success',
    direction_change: 'warning',
    state_change: 'primary',
    start: 'success',
    pause: 'warning',
    resume: 'primary',
    stop: 'danger'
  }
  return typeMap[event.type] || 'info'
}

const getEventTagType = (event: any) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    checkpoint: 'success',
    direction_change: 'warning',
    state_change: '',
    start: 'success',
    pause: 'warning',
    resume: '',
    stop: 'danger'
  }
  return typeMap[event.type] || 'info'
}

const getEventTypeLabel = (event: any) => {
  const labelMap: Record<string, string> = {
    checkpoint: '检查点',
    direction_change: '方向变更',
    state_change: '状态变更',
    start: '启动',
    pause: '暂停',
    resume: '恢复',
    stop: '停止'
  }
  return labelMap[event.type] || event.type
}
</script>

<style scoped lang="scss">
.event-timeline-card {
  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .timeline-content {
    max-height: 500px;
    overflow-y: auto;

    .event-item {
      .event-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;

        .checkpoint-id {
          font-size: 11px;
          font-family: monospace;
          color: var(--el-text-color-secondary);
        }
      }

      .event-description {
        font-size: 13px;
        color: var(--el-text-color-primary);
      }
    }
  }
}
</style>
