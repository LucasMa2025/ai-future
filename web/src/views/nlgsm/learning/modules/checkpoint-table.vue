<!-- 检查点表格 -->
<template>
  <ElCard shadow="hover" class="checkpoint-card">
    <template #header>
      <span>检查点</span>
    </template>

    <ElSkeleton :loading="loading" animated :rows="6">
      <template #default>
        <div v-if="checkpoints.length > 0" class="checkpoint-list">
          <div
            v-for="checkpoint in checkpoints"
            :key="checkpoint.checkpointId"
            class="checkpoint-item"
          >
            <div class="checkpoint-header">
              <ElIcon class="checkpoint-icon"><Flag /></ElIcon>
              <div class="checkpoint-info">
                <div class="checkpoint-id">{{ checkpoint.checkpointId.slice(0, 12) }}...</div>
                <div class="checkpoint-time">{{ formatTime(checkpoint.createdAt) }}</div>
              </div>
            </div>

            <div class="checkpoint-body">
              <div class="checkpoint-reason">{{ checkpoint.reason }}</div>
              <div class="checkpoint-progress">
                进度: {{ checkpoint.progressSnapshot.progressPercent.toFixed(1) }}%
              </div>
            </div>

            <div class="checkpoint-actions">
              <ElButton
                type="warning"
                size="small"
                plain
                @click="$emit('rollback', checkpoint.checkpointId)"
              >
                回滚到此
              </ElButton>
            </div>
          </div>
        </div>

        <ElEmpty v-else description="暂无检查点" :image-size="60" />
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { Flag } from '@element-plus/icons-vue'
import type { Checkpoint } from '@/types/nlgsm'

interface Props {
  checkpoints: Checkpoint[]
  loading: boolean
}

defineProps<Props>()

defineEmits<{
  (e: 'rollback', checkpointId: string): void
}>()

const formatTime = (time: string) => {
  if (!time) return ''
  const date = new Date(time)
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}
</script>

<style scoped lang="scss">
.checkpoint-card {
  .checkpoint-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-height: 500px;
    overflow-y: auto;

    .checkpoint-item {
      padding: 12px;
      background: var(--el-fill-color-light);
      border-radius: 8px;
      border-left: 3px solid var(--el-color-primary);

      .checkpoint-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;

        .checkpoint-icon {
          color: var(--el-color-primary);
        }

        .checkpoint-info {
          flex: 1;

          .checkpoint-id {
            font-size: 13px;
            font-weight: 500;
            font-family: monospace;
          }

          .checkpoint-time {
            font-size: 11px;
            color: var(--el-text-color-secondary);
          }
        }
      }

      .checkpoint-body {
        margin-bottom: 8px;

        .checkpoint-reason {
          font-size: 13px;
          color: var(--el-text-color-primary);
          margin-bottom: 4px;
        }

        .checkpoint-progress {
          font-size: 12px;
          color: var(--el-text-color-secondary);
        }
      }

      .checkpoint-actions {
        display: flex;
        justify-content: flex-end;
      }
    }
  }
}
</style>
