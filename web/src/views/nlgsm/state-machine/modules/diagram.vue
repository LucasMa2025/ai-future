<!-- 状态机图 -->
<template>
  <ElCard shadow="hover" class="diagram-card">
    <template #header>
      <div class="card-header">
        <span>NLGSM 状态机</span>
        <div class="header-actions">
          <ElTooltip content="刷新">
            <ElButton :icon="Refresh" circle size="small" @click="$emit('refresh')" />
          </ElTooltip>
        </div>
      </div>
    </template>

    <ElSkeleton :loading="loading" animated :rows="15">
      <template #default>
        <div class="diagram-container">
          <!-- SVG 状态机图 -->
          <svg viewBox="0 0 800 500" class="state-diagram">
            <defs>
              <!-- 箭头标记 -->
              <marker
                id="arrowhead"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#999" />
              </marker>
              <marker
                id="arrowhead-active"
                markerWidth="10"
                markerHeight="7"
                refX="9"
                refY="3.5"
                orient="auto"
              >
                <polygon points="0 0, 10 3.5, 0 7" fill="#409eff" />
              </marker>
            </defs>

            <!-- 转换连线 -->
            <g class="transitions">
              <path
                v-for="(transition, index) in transitionPaths"
                :key="index"
                :d="transition.path"
                class="transition-line"
                :class="{ active: isActiveTransition(transition) }"
                :marker-end="isActiveTransition(transition) ? 'url(#arrowhead-active)' : 'url(#arrowhead)'"
              />
            </g>

            <!-- 状态节点 -->
            <g
              v-for="node in stateNodes"
              :key="node.id"
              class="state-node"
              :class="{ current: node.id === currentState?.state }"
              :transform="`translate(${node.x}, ${node.y})`"
              @click="handleNodeClick(node)"
            >
              <circle :r="node.id === currentState?.state ? 45 : 40" :fill="node.color" />
              <text dy="5" text-anchor="middle" fill="#fff" font-size="14" font-weight="500">
                {{ node.label }}
              </text>
            </g>
          </svg>

          <!-- 图例 -->
          <div class="legend">
            <div class="legend-title">状态说明</div>
            <div class="legend-items">
              <div v-for="node in stateNodes" :key="node.id" class="legend-item">
                <span class="legend-dot" :style="{ backgroundColor: node.color }"></span>
                <span class="legend-label">{{ node.label }}</span>
              </div>
            </div>
          </div>
        </div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import type { SystemState, AvailableTransition } from '@/types/nlgsm'
import { NLGSMState, stateColors } from '@/types/nlgsm/enums'

interface Props {
  currentState: SystemState | null
  availableTransitions: AvailableTransition[]
  loading: boolean
}

const props = defineProps<Props>()

defineEmits<{
  (e: 'trigger-event', eventType: string): void
  (e: 'refresh'): void
}>()

// 状态节点位置（SVG 坐标系）
const stateNodes = computed(() => [
  { id: NLGSMState.FROZEN, label: '冻结态', x: 400, y: 60, color: stateColors[NLGSMState.FROZEN] },
  { id: NLGSMState.LEARNING, label: '学习态', x: 200, y: 150, color: stateColors[NLGSMState.LEARNING] },
  { id: NLGSMState.PAUSED, label: '暂停态', x: 100, y: 280, color: stateColors[NLGSMState.PAUSED] },
  { id: NLGSMState.VALIDATION, label: '验证态', x: 300, y: 280, color: stateColors[NLGSMState.VALIDATION] },
  { id: NLGSMState.RELEASE, label: '发布态', x: 600, y: 150, color: stateColors[NLGSMState.RELEASE] },
  { id: NLGSMState.ROLLBACK, label: '回滚态', x: 500, y: 280, color: stateColors[NLGSMState.ROLLBACK] },
  { id: NLGSMState.SAFE_HALT, label: '安全停机', x: 700, y: 280, color: stateColors[NLGSMState.SAFE_HALT] },
  { id: NLGSMState.DIAGNOSIS, label: '诊断态', x: 600, y: 400, color: stateColors[NLGSMState.DIAGNOSIS] },
  { id: NLGSMState.RECOVERY_PLAN, label: '恢复计划', x: 400, y: 440, color: stateColors[NLGSMState.RECOVERY_PLAN] }
])

// 状态转换路径
const transitionPaths = computed(() => {
  const nodeMap = new Map(stateNodes.value.map((n) => [n.id, n]))

  const transitions = [
    { from: NLGSMState.FROZEN, to: NLGSMState.LEARNING },
    { from: NLGSMState.LEARNING, to: NLGSMState.PAUSED },
    { from: NLGSMState.PAUSED, to: NLGSMState.LEARNING },
    { from: NLGSMState.LEARNING, to: NLGSMState.VALIDATION },
    { from: NLGSMState.VALIDATION, to: NLGSMState.FROZEN },
    { from: NLGSMState.VALIDATION, to: NLGSMState.LEARNING },
    { from: NLGSMState.FROZEN, to: NLGSMState.RELEASE },
    { from: NLGSMState.RELEASE, to: NLGSMState.FROZEN },
    { from: NLGSMState.LEARNING, to: NLGSMState.ROLLBACK },
    { from: NLGSMState.VALIDATION, to: NLGSMState.ROLLBACK },
    { from: NLGSMState.RELEASE, to: NLGSMState.ROLLBACK },
    { from: NLGSMState.ROLLBACK, to: NLGSMState.FROZEN },
    { from: NLGSMState.ROLLBACK, to: NLGSMState.SAFE_HALT },
    { from: NLGSMState.SAFE_HALT, to: NLGSMState.DIAGNOSIS },
    { from: NLGSMState.DIAGNOSIS, to: NLGSMState.RECOVERY_PLAN },
    { from: NLGSMState.RECOVERY_PLAN, to: NLGSMState.ROLLBACK },
    { from: NLGSMState.RECOVERY_PLAN, to: NLGSMState.FROZEN },
    { from: NLGSMState.RECOVERY_PLAN, to: NLGSMState.DIAGNOSIS }
  ]

  return transitions.map((t) => {
    const fromNode = nodeMap.get(t.from)
    const toNode = nodeMap.get(t.to)

    if (!fromNode || !toNode) return { from: t.from, to: t.to, path: '' }

    // 计算贝塞尔曲线控制点
    const dx = toNode.x - fromNode.x
    const dy = toNode.y - fromNode.y
    const mx = (fromNode.x + toNode.x) / 2
    const my = (fromNode.y + toNode.y) / 2

    // 添加一点弯曲
    const offset = 20
    const cx = mx - (dy / Math.abs(dy || 1)) * offset
    const cy = my + (dx / Math.abs(dx || 1)) * offset

    return {
      from: t.from,
      to: t.to,
      path: `M ${fromNode.x} ${fromNode.y} Q ${cx} ${cy} ${toNode.x} ${toNode.y}`
    }
  })
})

const isActiveTransition = (transition: { from: NLGSMState; to: NLGSMState }) => {
  if (!props.currentState) return false
  return (
    transition.from === props.currentState.state &&
    props.availableTransitions.some((t) => t.targetState === transition.to)
  )
}

const handleNodeClick = (node: { id: NLGSMState; label: string }) => {
  // 可以扩展为触发转换到该状态
  console.log('Clicked node:', node.id)
}
</script>

<style scoped lang="scss">
.diagram-card {
  height: 100%;
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.diagram-container {
  position: relative;

  .state-diagram {
    width: 100%;
    height: 500px;
    background: var(--el-fill-color-light);
    border-radius: 8px;

    .transition-line {
      fill: none;
      stroke: #999;
      stroke-width: 2;
      opacity: 0.5;
      transition: all 0.3s;

      &.active {
        stroke: #409eff;
        stroke-width: 3;
        opacity: 1;
      }
    }

    .state-node {
      cursor: pointer;
      transition: all 0.3s;

      circle {
        transition: all 0.3s;
        stroke: transparent;
        stroke-width: 3;
      }

      &:hover circle {
        transform: scale(1.1);
      }

      &.current circle {
        stroke: #409eff;
        stroke-width: 4;
        animation: pulse-ring 2s infinite;
      }
    }
  }

  .legend {
    position: absolute;
    bottom: 16px;
    left: 16px;
    padding: 12px;
    background: var(--el-bg-color);
    border-radius: 8px;
    box-shadow: var(--el-box-shadow-light);

    .legend-title {
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 8px;
      color: var(--el-text-color-primary);
    }

    .legend-items {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 6px;

      .legend-item {
        display: flex;
        align-items: center;
        gap: 4px;

        .legend-dot {
          width: 10px;
          height: 10px;
          border-radius: 50%;
        }

        .legend-label {
          font-size: 11px;
          color: var(--el-text-color-secondary);
        }
      }
    }
  }
}

@keyframes pulse-ring {
  0% {
    box-shadow: 0 0 0 0 rgba(64, 158, 255, 0.6);
  }
  70% {
    box-shadow: 0 0 0 15px rgba(64, 158, 255, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(64, 158, 255, 0);
  }
}
</style>
