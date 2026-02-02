<!-- 检测器状态 -->
<template>
  <ElCard shadow="hover" class="detector-card">
    <template #header>
      <div class="card-header">
        <span>检测器状态</span>
        <ElButton type="primary" size="small" @click="$emit('trigger')">
          手动检测
        </ElButton>
      </div>
    </template>

    <ElSkeleton :loading="loading" animated :rows="5">
      <template #default>
        <div class="detector-list">
          <div
            v-for="detector in detectors"
            :key="detector.id"
            class="detector-item"
          >
            <div class="detector-info">
              <div class="detector-name">
                <ElSwitch
                  :model-value="detector.enabled"
                  size="small"
                  @change="handleToggle(detector)"
                />
                <span>{{ detector.name }}</span>
              </div>
              <div class="detector-meta">
                <span class="last-check">
                  上次检查: {{ formatTime(detector.lastCheckAt) }}
                </span>
              </div>
            </div>

            <div class="detector-score">
              <div class="score-value" :style="{ color: getScoreColor(detector.lastScore) }">
                {{ (detector.lastScore * 100).toFixed(0) }}%
              </div>
              <div class="threshold">阈值: {{ (detector.threshold * 100).toFixed(0) }}%</div>
            </div>

            <div class="alert-count">
              <ElBadge
                :value="detector.alertCount"
                :hidden="detector.alertCount === 0"
                :max="99"
              >
                <ElIcon :size="20"><Bell /></ElIcon>
              </ElBadge>
            </div>
          </div>
        </div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { Bell } from '@element-plus/icons-vue'
import type { DetectorStatus } from '@/types/nlgsm'

interface Props {
  detectors: DetectorStatus[]
  loading: boolean
}

defineProps<Props>()

const emit = defineEmits<{
  (e: 'update', id: string, data: { enabled?: boolean; threshold?: number }): void
  (e: 'trigger', detectorId?: string): void
}>()

const formatTime = (time: string) => {
  if (!time) return '-'
  const date = new Date(time)
  return date.toLocaleTimeString('zh-CN')
}

const getScoreColor = (score: number) => {
  if (score < 0.3) return '#4caf50'
  if (score < 0.6) return '#ff9800'
  return '#f44336'
}

const handleToggle = (detector: DetectorStatus) => {
  emit('update', detector.id, { enabled: !detector.enabled })
}
</script>

<style scoped lang="scss">
.detector-card {
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.detector-list {
  display: flex;
  flex-direction: column;
  gap: 12px;

  .detector-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--el-fill-color-light);
    border-radius: 8px;

    .detector-info {
      flex: 1;

      .detector-name {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        font-weight: 500;
      }

      .detector-meta {
        margin-top: 4px;
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }
    }

    .detector-score {
      text-align: right;

      .score-value {
        font-size: 18px;
        font-weight: 600;
      }

      .threshold {
        font-size: 11px;
        color: var(--el-text-color-secondary);
      }
    }

    .alert-count {
      margin-left: 8px;
    }
  }
}
</style>
