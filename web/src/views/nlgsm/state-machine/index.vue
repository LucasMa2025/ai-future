<!-- 状态机页面 -->
<template>
  <div class="state-machine-page">
    <ElRow :gutter="20">
      <!-- 状态机图 -->
      <ElCol :sm="24" :lg="16">
        <StateDiagram
          :current-state="currentState"
          :available-transitions="availableTransitions"
          :loading="loading"
          @trigger-event="handleTriggerEvent"
        />
      </ElCol>

      <!-- 控制面板 -->
      <ElCol :sm="24" :lg="8">
        <ControlPanel
          :current-state="currentState"
          :available-transitions="availableTransitions"
          :loading="loading"
          @trigger-event="handleTriggerEvent"
          @force-state="handleForceState"
        />
      </ElCol>
    </ElRow>

    <!-- 转换历史 -->
    <HistoryTable :history="history" :total="historyTotal" :loading="loading" @refresh="fetchHistory" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useStateMachineStore } from '@/store/modules/state-machine'
import StateDiagram from './modules/diagram.vue'
import ControlPanel from './modules/control-panel.vue'
import HistoryTable from './modules/history-table.vue'

defineOptions({ name: 'StateMachine' })

const stateMachineStore = useStateMachineStore()

const loading = computed(() => stateMachineStore.loading)
const currentState = computed(() => stateMachineStore.currentState)
const availableTransitions = computed(() => stateMachineStore.availableTransitions)
const history = computed(() => stateMachineStore.history)
const historyTotal = computed(() => stateMachineStore.historyTotal)

const handleTriggerEvent = async (eventType: string) => {
  try {
    await ElMessageBox.confirm(`确定要触发事件 "${eventType}" 吗？`, '确认操作', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })

    const result = await stateMachineStore.triggerEvent(eventType)

    if (result.success) {
      ElMessage.success(`状态转换成功: ${result.fromState} -> ${result.toState}`)
    } else {
      ElMessage.error(result.error || '状态转换失败')
    }
  } catch {
    // 用户取消
  }
}

const handleForceState = async (targetState: string, reason: string) => {
  try {
    await ElMessageBox.confirm(
      `确定要强制设置状态为 "${targetState}" 吗？\n原因: ${reason}`,
      '危险操作',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'error'
      }
    )

    const result = await stateMachineStore.forceState(targetState, reason)

    if (result.success) {
      ElMessage.success(`状态已强制设置为: ${targetState}`)
    } else {
      ElMessage.error(result.error || '操作失败')
    }
  } catch {
    // 用户取消
  }
}

const fetchHistory = async () => {
  await stateMachineStore.fetchHistory()
}

onMounted(async () => {
  await stateMachineStore.initialize()
})
</script>

<style scoped lang="scss">
.state-machine-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
