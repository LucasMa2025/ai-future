<!-- 状态机概览图 -->
<template>
  <ElCard shadow="hover" class="state-flow-card">
    <template #header>
      <div class="card-header">
        <span>状态机概览</span>
        <ElButton link type="primary" @click="goToStateMachine">
          查看详情
          <ElIcon class="el-icon--right"><ArrowRight /></ElIcon>
        </ElButton>
      </div>
    </template>

    <ElSkeleton :loading="loading" animated :rows="8">
      <template #default>
        <div class="state-flow-content">
          <!-- 当前状态指示 -->
          <div class="current-state-badge">
            <div class="state-indicator" :style="{ backgroundColor: stateColor }"></div>
            <div class="state-info">
              <div class="state-label">当前状态</div>
              <div class="state-name">{{ stateLabel }}</div>
            </div>
          </div>

          <!-- 简化状态流程图 -->
          <div class="state-flow-diagram">
            <div
              v-for="state in stateNodes"
              :key="state.id"
              class="state-node"
              :class="{ active: state.id === currentState?.state }"
              :style="{ '--state-color': state.color }"
            >
              <div class="node-dot"></div>
              <div class="node-label">{{ state.label }}</div>
            </div>
          </div>

          <!-- 状态信息 -->
          <div v-if="currentState" class="state-details">
            <div class="detail-item">
              <span class="detail-label">进入时间</span>
              <span class="detail-value">{{ formatTime(currentState.enteredAt) }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">触发事件</span>
              <span class="detail-value">{{ currentState.triggerEvent || '-' }}</span>
            </div>
            <div class="detail-item">
              <span class="detail-label">迭代次数</span>
              <span class="detail-value">{{ currentState.iterationCount }}</span>
            </div>
          </div>
        </div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowRight } from '@element-plus/icons-vue'
import type { SystemState } from '@/types/nlgsm'
import { NLGSMState, stateLabels, stateColors } from '@/types/nlgsm/enums'

interface Props {
  currentState: SystemState | null
  loading: boolean
}

const props = defineProps<Props>()
const router = useRouter()

const stateNodes = [
  { id: NLGSMState.FROZEN, label: '冻结', color: stateColors[NLGSMState.FROZEN] },
  { id: NLGSMState.LEARNING, label: '学习', color: stateColors[NLGSMState.LEARNING] },
  { id: NLGSMState.PAUSED, label: '暂停', color: stateColors[NLGSMState.PAUSED] },
  { id: NLGSMState.VALIDATION, label: '验证', color: stateColors[NLGSMState.VALIDATION] },
  { id: NLGSMState.RELEASE, label: '发布', color: stateColors[NLGSMState.RELEASE] },
  { id: NLGSMState.ROLLBACK, label: '回滚', color: stateColors[NLGSMState.ROLLBACK] },
  { id: NLGSMState.SAFE_HALT, label: '停机', color: stateColors[NLGSMState.SAFE_HALT] },
  { id: NLGSMState.DIAGNOSIS, label: '诊断', color: stateColors[NLGSMState.DIAGNOSIS] },
  { id: NLGSMState.RECOVERY_PLAN, label: '恢复', color: stateColors[NLGSMState.RECOVERY_PLAN] }
]

const stateLabel = computed(() => {
  if (!props.currentState?.state) return '未知'
  return stateLabels[props.currentState.state as NLGSMState] || props.currentState.state
})

const stateColor = computed(() => {
  if (!props.currentState?.state) return '#9e9e9e'
  return stateColors[props.currentState.state as NLGSMState] || '#9e9e9e'
})

const formatTime = (time: string) => {
  if (!time) return '-'
  const date = new Date(time)
  return date.toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit'
  })
}

const goToStateMachine = () => {
  router.push({ name: 'StateMachine' })
}
</script>

<style scoped lang="scss">
.state-flow-card {
  height: 100%;
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.state-flow-content {
  .current-state-badge {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px;
    margin-bottom: 20px;
    background: var(--el-fill-color-light);
    border-radius: 8px;

    .state-indicator {
      width: 48px;
      height: 48px;
      border-radius: 50%;
      animation: pulse 2s infinite;
    }

    .state-info {
      .state-label {
        font-size: 12px;
        color: var(--el-text-color-secondary);
      }

      .state-name {
        font-size: 20px;
        font-weight: 600;
        color: var(--el-text-color-primary);
      }
    }
  }

  .state-flow-diagram {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 20px;

    .state-node {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 12px;
      background: var(--el-fill-color);
      border-radius: 16px;
      font-size: 12px;
      color: var(--el-text-color-secondary);
      transition: all 0.3s;

      .node-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--state-color);
        opacity: 0.5;
      }

      &.active {
        background: var(--state-color);
        color: #fff;

        .node-dot {
          background: #fff;
          opacity: 1;
        }
      }
    }
  }

  .state-details {
    display: flex;
    flex-direction: column;
    gap: 12px;

    .detail-item {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid var(--el-border-color-lighter);

      &:last-child {
        border-bottom: none;
      }

      .detail-label {
        font-size: 13px;
        color: var(--el-text-color-secondary);
      }

      .detail-value {
        font-size: 13px;
        font-weight: 500;
        color: var(--el-text-color-primary);
      }
    }
  }
}

@keyframes pulse {
  0%,
  100% {
    box-shadow: 0 0 0 0 rgba(156, 39, 176, 0.4);
  }
  50% {
    box-shadow: 0 0 0 10px rgba(156, 39, 176, 0);
  }
}
</style>
