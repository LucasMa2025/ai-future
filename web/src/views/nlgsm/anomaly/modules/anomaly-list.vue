<!-- 异常列表 -->
<template>
  <ElCard shadow="hover" class="anomaly-list-card">
    <template #header>
      <div class="card-header">
        <div class="header-left">
          <span>实时告警</span>
          <ElBadge v-if="data.length > 0" :value="data.length" type="danger" />
        </div>
        <ElButton :icon="Refresh" size="small" @click="$emit('refresh')">刷新</ElButton>
      </div>
    </template>

    <div v-if="data.length > 0" class="anomaly-content">
      <div
        v-for="item in data"
        :key="item.id"
        class="anomaly-item"
        :class="getSeverityClass(item.severity)"
      >
        <div class="anomaly-header">
          <div class="severity-indicator">
            <ElIcon :size="20"><WarningFilled /></ElIcon>
          </div>
          <div class="anomaly-info">
            <div class="anomaly-title">
              <ElTag :type="getSeverityTagType(item.severity)" size="small">
                {{ getSeverityLabel(item.severity) }}
              </ElTag>
              <span class="score">分数: {{ (item.compositeScore * 100).toFixed(0) }}%</span>
            </div>
            <div class="anomaly-time">{{ formatTime(item.createdAt) }}</div>
          </div>
        </div>

        <div class="anomaly-body">
          <div class="recommendation">{{ item.recommendation }}</div>

          <div class="detectors">
            <span class="label">检测器:</span>
            <ElTag
              v-for="detector in item.detectedBy"
              :key="detector"
              size="small"
              effect="plain"
            >
              {{ detector }}
            </ElTag>
          </div>
        </div>

        <div class="anomaly-actions">
          <ElButton type="success" size="small" @click="$emit('resolve', item)">
            解决
          </ElButton>
          <ElButton type="warning" size="small" plain @click="$emit('ignore', item)">
            忽略
          </ElButton>
        </div>
      </div>
    </div>

    <ElEmpty v-else description="暂无告警" :image-size="80">
      <template #description>
        <div class="empty-desc">
          <ElIcon :size="48" color="#4caf50"><CircleCheck /></ElIcon>
          <p>系统运行正常，无异常告警</p>
        </div>
      </template>
    </ElEmpty>
  </ElCard>
</template>

<script setup lang="ts">
import { Refresh, WarningFilled, CircleCheck } from '@element-plus/icons-vue'
import type { AnomalyEvent } from '@/types/nlgsm'
import { AnomalySeverity } from '@/types/nlgsm/enums'

interface Props {
  data: AnomalyEvent[]
  loading: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'resolve', row: AnomalyEvent): void
  (e: 'ignore', row: AnomalyEvent): void
  (e: 'refresh'): void
}>()

const severityLabels: Record<string, string> = {
  low: '低',
  medium: '中',
  high: '高',
  critical: '严重'
}

const getSeverityLabel = (severity: AnomalySeverity) => severityLabels[severity] || severity

const getSeverityClass = (severity: AnomalySeverity) => {
  return `severity-${severity}`
}

const getSeverityTagType = (severity: AnomalySeverity) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    low: 'info',
    medium: 'warning',
    high: 'danger',
    critical: 'danger'
  }
  return typeMap[severity] || 'info'
}

const formatTime = (time: string) => {
  if (!time) return ''
  const date = new Date(time)
  const now = new Date()
  const diff = now.getTime() - date.getTime()

  if (diff < 60000) {
    return '刚刚'
  }
  if (diff < 3600000) {
    return `${Math.floor(diff / 60000)} 分钟前`
  }
  if (diff < 86400000) {
    return `${Math.floor(diff / 3600000)} 小时前`
  }
  return date.toLocaleString('zh-CN')
}
</script>

<style scoped lang="scss">
.anomaly-list-card {
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    .header-left {
      display: flex;
      align-items: center;
      gap: 8px;
    }
  }
}

.anomaly-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 600px;
  overflow-y: auto;

  .anomaly-item {
    padding: 16px;
    border-radius: 8px;
    border-left: 4px solid;

    &.severity-low {
      background: rgba(33, 150, 243, 0.05);
      border-left-color: #2196f3;
    }

    &.severity-medium {
      background: rgba(255, 152, 0, 0.05);
      border-left-color: #ff9800;
    }

    &.severity-high {
      background: rgba(244, 67, 54, 0.05);
      border-left-color: #f44336;
    }

    &.severity-critical {
      background: rgba(183, 28, 28, 0.05);
      border-left-color: #b71c1c;
    }

    .anomaly-header {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 12px;

      .severity-indicator {
        color: var(--el-color-danger);
      }

      .anomaly-info {
        flex: 1;

        .anomaly-title {
          display: flex;
          align-items: center;
          gap: 8px;

          .score {
            font-size: 12px;
            color: var(--el-text-color-secondary);
          }
        }

        .anomaly-time {
          font-size: 12px;
          color: var(--el-text-color-secondary);
          margin-top: 4px;
        }
      }
    }

    .anomaly-body {
      .recommendation {
        font-size: 14px;
        color: var(--el-text-color-primary);
        margin-bottom: 8px;
      }

      .detectors {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px;
        font-size: 12px;

        .label {
          color: var(--el-text-color-secondary);
        }
      }
    }

    .anomaly-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid var(--el-border-color-lighter);
    }
  }
}

.empty-desc {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;

  p {
    margin: 0;
    color: var(--el-text-color-secondary);
  }
}
</style>
