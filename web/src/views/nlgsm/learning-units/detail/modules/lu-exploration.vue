<!-- LU 探索路径 -->
<template>
  <ElCard shadow="hover" class="exploration-card">
    <template #header>
      <span>探索路径</span>
    </template>

    <ElSkeleton :loading="loading" animated :rows="10">
      <template #default>
        <div v-if="steps.length > 0" class="exploration-content">
          <ElTimeline>
            <ElTimelineItem
              v-for="(step, index) in steps"
              :key="step.stepId"
              :timestamp="`深度 ${step.depth}`"
              placement="top"
            >
              <div class="step-item">
                <div class="step-header">
                  <span class="step-index">#{{ index + 1 }}</span>
                  <span class="step-time">{{ formatTime(step.timestamp) }}</span>
                </div>
                <div class="step-action">
                  <strong>行动:</strong> {{ step.action }}
                </div>
                <div class="step-result">
                  <strong>结果:</strong> {{ step.result }}
                </div>
              </div>
            </ElTimelineItem>
          </ElTimeline>
        </div>

        <ElEmpty v-else description="暂无探索步骤" :image-size="80" />
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import type { ExplorationStep } from '@/types/nlgsm'

interface Props {
  steps: ExplorationStep[]
  loading: boolean
}

defineProps<Props>()

const formatTime = (time: string) => {
  if (!time) return ''
  return new Date(time).toLocaleTimeString('zh-CN')
}
</script>

<style scoped lang="scss">
.exploration-card {
  height: fit-content;
}

.exploration-content {
  max-height: 600px;
  overflow-y: auto;

  .step-item {
    .step-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;

      .step-index {
        font-size: 12px;
        font-weight: 600;
        color: var(--el-color-primary);
        background: var(--el-color-primary-light-9);
        padding: 2px 8px;
        border-radius: 10px;
      }

      .step-time {
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }
    }

    .step-action,
    .step-result {
      font-size: 13px;
      line-height: 1.6;
      margin-bottom: 4px;

      strong {
        color: var(--el-text-color-secondary);
        margin-right: 4px;
      }
    }
  }
}
</style>
