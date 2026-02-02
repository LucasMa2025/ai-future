<!-- 状态控制面板 -->
<template>
  <ElCard shadow="hover" class="control-panel-card">
    <template #header>
      <span>状态控制</span>
    </template>

    <ElSkeleton :loading="loading" animated :rows="10">
      <template #default>
        <div class="control-content">
          <!-- 当前状态 -->
          <div class="current-state-section">
            <div class="section-title">当前状态</div>
            <div class="state-badge" :style="{ backgroundColor: stateColor }">
              {{ stateLabel }}
            </div>
            <div v-if="currentState" class="state-info">
              <div class="info-row">
                <span class="label">进入时间:</span>
                <span class="value">{{ formatTime(currentState.enteredAt) }}</span>
              </div>
              <div class="info-row">
                <span class="label">触发事件:</span>
                <span class="value">{{ currentState.triggerEvent || '-' }}</span>
              </div>
              <div class="info-row">
                <span class="label">触发来源:</span>
                <span class="value">{{ currentState.triggerSource || '-' }}</span>
              </div>
              <div class="info-row">
                <span class="label">迭代次数:</span>
                <span class="value">{{ currentState.iterationCount }}</span>
              </div>
            </div>
          </div>

          <ElDivider />

          <!-- 可用转换 -->
          <div class="available-transitions-section">
            <div class="section-title">可触发事件</div>
            <div v-if="availableTransitions.length > 0" class="transition-list">
              <div
                v-for="transition in availableTransitions"
                :key="transition.eventType"
                class="transition-item"
              >
                <div class="transition-info">
                  <div class="event-name">{{ transition.eventType }}</div>
                  <div class="target-state">
                    → {{ getStateLabel(transition.targetState) }}
                  </div>
                </div>
                <ElButton
                  type="primary"
                  size="small"
                  @click="$emit('trigger-event', transition.eventType)"
                >
                  触发
                </ElButton>
              </div>
            </div>
            <ElEmpty v-else description="当前状态无可触发事件" :image-size="60" />
          </div>

          <ElDivider />

          <!-- 强制设置状态 -->
          <div class="force-state-section">
            <div class="section-title">
              <span>强制设置状态</span>
              <ElTag type="danger" size="small">管理员</ElTag>
            </div>
            <ElForm label-position="top" size="small">
              <ElFormItem label="目标状态">
                <ElSelect v-model="forceForm.targetState" placeholder="选择目标状态">
                  <ElOption
                    v-for="state in allStates"
                    :key="state.id"
                    :label="state.label"
                    :value="state.id"
                  />
                </ElSelect>
              </ElFormItem>
              <ElFormItem label="原因">
                <ElInput
                  v-model="forceForm.reason"
                  type="textarea"
                  :rows="2"
                  placeholder="请输入强制设置原因..."
                />
              </ElFormItem>
              <ElFormItem>
                <ElButton
                  type="danger"
                  :disabled="!forceForm.targetState || !forceForm.reason"
                  @click="handleForceState"
                >
                  强制设置
                </ElButton>
              </ElFormItem>
            </ElForm>
          </div>
        </div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { SystemState, AvailableTransition } from '@/types/nlgsm'
import { NLGSMState, stateLabels, stateColors } from '@/types/nlgsm/enums'

interface Props {
  currentState: SystemState | null
  availableTransitions: AvailableTransition[]
  loading: boolean
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'trigger-event', eventType: string): void
  (e: 'force-state', targetState: string, reason: string): void
}>()

const forceForm = ref({
  targetState: '',
  reason: ''
})

const allStates = Object.values(NLGSMState).map((id) => ({
  id,
  label: stateLabels[id]
}))

const stateLabel = computed(() => {
  if (!props.currentState?.state) return '未知'
  return stateLabels[props.currentState.state as NLGSMState] || props.currentState.state
})

const stateColor = computed(() => {
  if (!props.currentState?.state) return '#9e9e9e'
  return stateColors[props.currentState.state as NLGSMState] || '#9e9e9e'
})

const getStateLabel = (state: NLGSMState) => {
  return stateLabels[state] || state
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const handleForceState = () => {
  if (forceForm.value.targetState && forceForm.value.reason) {
    emit('force-state', forceForm.value.targetState, forceForm.value.reason)
    forceForm.value = { targetState: '', reason: '' }
  }
}
</script>

<style scoped lang="scss">
.control-panel-card {
  height: 100%;
  margin-bottom: 20px;
}

.control-content {
  .section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    font-size: 14px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }

  .current-state-section {
    .state-badge {
      display: inline-block;
      padding: 8px 20px;
      margin-bottom: 12px;
      font-size: 16px;
      font-weight: 600;
      color: #fff;
      border-radius: 20px;
    }

    .state-info {
      .info-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        font-size: 13px;
        border-bottom: 1px solid var(--el-border-color-lighter);

        &:last-child {
          border-bottom: none;
        }

        .label {
          color: var(--el-text-color-secondary);
        }

        .value {
          font-weight: 500;
          color: var(--el-text-color-primary);
        }
      }
    }
  }

  .available-transitions-section {
    .transition-list {
      display: flex;
      flex-direction: column;
      gap: 8px;

      .transition-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 12px;
        background: var(--el-fill-color-light);
        border-radius: 8px;

        .transition-info {
          .event-name {
            font-size: 13px;
            font-weight: 500;
            color: var(--el-text-color-primary);
          }

          .target-state {
            font-size: 12px;
            color: var(--el-text-color-secondary);
          }
        }
      }
    }
  }

  .force-state-section {
    :deep(.el-form-item) {
      margin-bottom: 12px;
    }
  }
}
</style>
