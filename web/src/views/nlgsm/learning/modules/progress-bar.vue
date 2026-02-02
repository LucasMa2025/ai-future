<!-- 学习进度条 -->
<template>
  <ElCard shadow="hover" class="progress-card">
    <template #header>
      <span>学习进度</span>
    </template>

    <div v-if="progress" class="progress-content">
      <!-- 主进度条 -->
      <div class="main-progress">
        <div class="progress-info">
          <span class="percentage">{{ progress.progressPercent.toFixed(1) }}%</span>
          <span class="steps">
            {{ progress.completedSteps }} / {{ progress.totalSteps }} 步
          </span>
        </div>
        <ElProgress
          :percentage="progress.progressPercent"
          :stroke-width="20"
          :color="getProgressColor(progress.progressPercent)"
        />
      </div>

      <!-- 详细指标 -->
      <div class="metrics">
        <div class="metric-item">
          <div class="metric-icon" style="background: rgba(33, 150, 243, 0.1)">
            <ElIcon color="#2196f3"><TrendCharts /></ElIcon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ progress.completedSteps }}</div>
            <div class="metric-label">已完成步骤</div>
          </div>
        </div>

        <div class="metric-item">
          <div class="metric-icon" style="background: rgba(156, 39, 176, 0.1)">
            <ElIcon color="#9c27b0"><Histogram /></ElIcon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ progress.currentDepth }}</div>
            <div class="metric-label">当前深度</div>
          </div>
        </div>

        <div class="metric-item">
          <div class="metric-icon" style="background: rgba(255, 152, 0, 0.1)">
            <ElIcon color="#ff9800"><Timer /></ElIcon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ formatDuration }}</div>
            <div class="metric-label">运行时间</div>
          </div>
        </div>

        <div class="metric-item">
          <div class="metric-icon" style="background: rgba(76, 175, 80, 0.1)">
            <ElIcon color="#4caf50"><Flag /></ElIcon>
          </div>
          <div class="metric-info">
            <div class="metric-value">{{ session?.checkpoints?.length || 0 }}</div>
            <div class="metric-label">检查点</div>
          </div>
        </div>
      </div>
    </div>

    <ElEmpty v-else description="暂无进度数据" :image-size="60" />
  </ElCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { TrendCharts, Histogram, Timer, Flag } from '@element-plus/icons-vue'
import type { LearningSession, LearningProgress } from '@/types/nlgsm'

interface Props {
  progress: LearningProgress | null
  session: LearningSession
}

const props = defineProps<Props>()

const formatDuration = computed(() => {
  if (!props.session?.startedAt) return '-'

  const start = new Date(props.session.startedAt).getTime()
  const now = props.session.pausedAt ? new Date(props.session.pausedAt).getTime() : Date.now()
  const diff = now - start

  const hours = Math.floor(diff / 3600000)
  const minutes = Math.floor((diff % 3600000) / 60000)
  const seconds = Math.floor((diff % 60000) / 1000)

  if (hours > 0) {
    return `${hours}h ${minutes}m`
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds}s`
  }
  return `${seconds}s`
})

const getProgressColor = (percent: number) => {
  if (percent < 30) return '#ff9800'
  if (percent < 70) return '#2196f3'
  return '#4caf50'
}
</script>

<style scoped lang="scss">
.progress-card {
  margin-bottom: 20px;
}

.progress-content {
  .main-progress {
    margin-bottom: 24px;

    .progress-info {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;

      .percentage {
        font-size: 32px;
        font-weight: 600;
        color: var(--el-text-color-primary);
      }

      .steps {
        font-size: 14px;
        color: var(--el-text-color-secondary);
      }
    }
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;

    @media (max-width: 768px) {
      grid-template-columns: repeat(2, 1fr);
    }

    .metric-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px;
      background: var(--el-fill-color-light);
      border-radius: 8px;

      .metric-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 12px;
      }

      .metric-info {
        .metric-value {
          font-size: 20px;
          font-weight: 600;
          color: var(--el-text-color-primary);
        }

        .metric-label {
          font-size: 12px;
          color: var(--el-text-color-secondary);
        }
      }
    }
  }
}
</style>
