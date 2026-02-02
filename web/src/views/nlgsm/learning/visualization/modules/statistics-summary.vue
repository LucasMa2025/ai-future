<!-- 统计摘要 -->
<template>
  <ElRow :gutter="16" class="statistics-row">
    <ElCol :xs="12" :sm="8" :md="4">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(33, 150, 243, 0.1)">
          <ElIcon :size="24" color="#2196f3"><Timer /></ElIcon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ formatDuration(statistics.durationSeconds) }}</div>
          <div class="stat-label">运行时间</div>
        </div>
      </div>
    </ElCol>

    <ElCol :xs="12" :sm="8" :md="4">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(76, 175, 80, 0.1)">
          <ElIcon :size="24" color="#4caf50"><CircleCheck /></ElIcon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ statistics.completedSteps }} / {{ statistics.totalSteps }}</div>
          <div class="stat-label">完成步骤</div>
        </div>
      </div>
    </ElCol>

    <ElCol :xs="12" :sm="8" :md="4">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(156, 39, 176, 0.1)">
          <ElIcon :size="24" color="#9c27b0"><TrendCharts /></ElIcon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ statistics.progressPercent.toFixed(1) }}%</div>
          <div class="stat-label">总进度</div>
        </div>
      </div>
    </ElCol>

    <ElCol :xs="12" :sm="8" :md="4">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(255, 152, 0, 0.1)">
          <ElIcon :size="24" color="#ff9800"><Histogram /></ElIcon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ statistics.currentDepth }}</div>
          <div class="stat-label">当前深度</div>
        </div>
      </div>
    </ElCol>

    <ElCol :xs="12" :sm="8" :md="4">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(0, 150, 136, 0.1)">
          <ElIcon :size="24" color="#009688"><Flag /></ElIcon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ statistics.checkpointsCount }}</div>
          <div class="stat-label">检查点</div>
        </div>
      </div>
    </ElCol>

    <ElCol :xs="12" :sm="8" :md="4">
      <div class="stat-card">
        <div class="stat-icon" style="background: rgba(244, 67, 54, 0.1)">
          <ElIcon :size="24" color="#f44336"><Compass /></ElIcon>
        </div>
        <div class="stat-info">
          <div class="stat-value">{{ statistics.directionChangesCount }}</div>
          <div class="stat-label">方向变更</div>
        </div>
      </div>
    </ElCol>
  </ElRow>
</template>

<script setup lang="ts">
import { Timer, CircleCheck, TrendCharts, Histogram, Flag, Compass } from '@element-plus/icons-vue'
import type { LearningStatistics } from '@/types/nlgsm'

interface Props {
  statistics: LearningStatistics
}

defineProps<Props>()

const formatDuration = (seconds: number | null) => {
  if (!seconds) return '0s'

  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  if (minutes > 0) {
    return `${minutes}m ${secs}s`
  }
  return `${secs}s`
}
</script>

<style scoped lang="scss">
.statistics-row {
  margin-bottom: 20px;

  .stat-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    background: var(--el-bg-color);
    border-radius: 8px;
    border: 1px solid var(--el-border-color-lighter);
    margin-bottom: 16px;

    .stat-icon {
      display: flex;
      align-items: center;
      justify-content: center;
      width: 48px;
      height: 48px;
      border-radius: 12px;
      flex-shrink: 0;
    }

    .stat-info {
      .stat-value {
        font-size: 20px;
        font-weight: 600;
        color: var(--el-text-color-primary);
      }

      .stat-label {
        font-size: 12px;
        color: var(--el-text-color-secondary);
        margin-top: 2px;
      }
    }
  }
}
</style>
