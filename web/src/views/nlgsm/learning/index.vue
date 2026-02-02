<!-- 学习控制页面 -->
<template>
  <div class="learning-control-page">
    <ElRow :gutter="20">
      <!-- 左侧：控制面板 + 进度 -->
      <ElCol :sm="24" :lg="16">
        <!-- 控制面板 -->
        <ControlPanel
          :session="currentSession"
          :loading="loading"
          :available-actions="availableActions"
          @start="handleStart"
          @pause="handlePause"
          @resume="handleResume"
          @stop="handleStop"
          @redirect="handleRedirect"
          @checkpoint="handleCheckpoint"
        />

        <!-- 会话信息 -->
        <SessionInfo v-if="currentSession" :session="currentSession" />

        <!-- 进度条 -->
        <ProgressBar
          v-if="currentSession"
          :progress="progress"
          :session="currentSession"
        />
      </ElCol>

      <!-- 右侧：检查点 -->
      <ElCol :sm="24" :lg="8">
        <CheckpointTable
          :checkpoints="checkpoints"
          :loading="loading"
          @rollback="handleRollback"
        />
      </ElCol>
    </ElRow>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useLearningControlStore } from '@/store/modules/learning-control'
import ControlPanel from './modules/control-panel.vue'
import SessionInfo from './modules/session-info.vue'
import ProgressBar from './modules/progress-bar.vue'
import CheckpointTable from './modules/checkpoint-table.vue'

defineOptions({ name: 'LearningControl' })

const router = useRouter()
const store = useLearningControlStore()

const loading = computed(() => store.loading)
const currentSession = computed(() => store.currentSession)
const progress = computed(() => store.progress)
const checkpoints = computed(() => store.checkpoints)
const availableActions = computed(() => store.availableActions)

const handleStart = async (goal: string, scope?: any) => {
  try {
    await store.startLearning(goal, scope)
    ElMessage.success('学习会话已启动')
  } catch (e) {
    ElMessage.error('启动失败')
  }
}

const handlePause = async () => {
  try {
    const { value } = await ElMessageBox.prompt('请输入暂停原因', '暂停学习', {
      confirmButtonText: '暂停',
      cancelButtonText: '取消',
      inputPlaceholder: '请输入暂停原因...'
    })

    await store.pauseLearning(value || '手动暂停')
    ElMessage.success('学习已暂停')
  } catch {
    // 用户取消
  }
}

const handleResume = async () => {
  try {
    await ElMessageBox.confirm('确定要恢复学习吗？', '恢复学习', {
      confirmButtonText: '恢复',
      cancelButtonText: '取消',
      type: 'info'
    })

    await store.resumeLearning()
    ElMessage.success('学习已恢复')
  } catch {
    // 用户取消
  }
}

const handleStop = async () => {
  try {
    const { value } = await ElMessageBox.prompt('请输入停止原因', '停止学习', {
      confirmButtonText: '停止',
      cancelButtonText: '取消',
      inputPlaceholder: '请输入停止原因...',
      type: 'warning'
    })

    await store.stopLearning(value || '手动停止', true)
    ElMessage.success('学习已停止')
  } catch {
    // 用户取消
  }
}

const handleRedirect = async (newDirection: string, reason: string) => {
  try {
    await store.redirectLearning(newDirection, reason)
    ElMessage.success('学习方向已调整')
  } catch (e) {
    ElMessage.error('调整失败')
  }
}

const handleCheckpoint = async () => {
  try {
    await store.createCheckpoint('手动创建')
    ElMessage.success('检查点已创建')
  } catch (e) {
    ElMessage.error('创建失败')
  }
}

const handleRollback = async (checkpointId: string) => {
  try {
    const { value } = await ElMessageBox.prompt('请输入回滚原因', '回滚到检查点', {
      confirmButtonText: '回滚',
      cancelButtonText: '取消',
      inputPlaceholder: '请输入回滚原因...',
      type: 'warning'
    })

    await store.rollbackToCheckpoint(checkpointId, value || '手动回滚')
    ElMessage.success('回滚成功')
  } catch {
    // 用户取消
  }
}

onMounted(async () => {
  await store.initialize()
})
</script>

<style scoped lang="scss">
.learning-control-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
