<!-- 学习控制面板 -->
<template>
  <ElCard shadow="hover" class="control-panel-card">
    <template #header>
      <div class="card-header">
        <span>学习控制</span>
        <div class="status-badge" :class="statusClass">
          {{ statusLabel }}
        </div>
      </div>
    </template>

    <div class="control-content">
      <!-- 无会话时显示启动表单 -->
      <div v-if="!session" class="start-form">
        <ElForm :model="startForm" :rules="startRules" ref="startFormRef" label-position="top">
          <ElFormItem label="学习目标" prop="goal">
            <ElInput
              v-model="startForm.goal"
              type="textarea"
              :rows="3"
              placeholder="请输入学习目标..."
            />
          </ElFormItem>
          <ElFormItem>
            <ElButton type="primary" :loading="loading" @click="handleStart">
              <ElIcon class="el-icon--left"><VideoPlay /></ElIcon>
              启动学习
            </ElButton>
          </ElFormItem>
        </ElForm>
      </div>

      <!-- 有会话时显示控制按钮 -->
      <div v-else class="control-buttons">
        <ElButton
          v-if="availableActions.includes('pause')"
          type="warning"
          :loading="loading"
          @click="$emit('pause')"
        >
          <ElIcon class="el-icon--left"><VideoPause /></ElIcon>
          暂停
        </ElButton>

        <ElButton
          v-if="availableActions.includes('resume')"
          type="success"
          :loading="loading"
          @click="$emit('resume')"
        >
          <ElIcon class="el-icon--left"><VideoPlay /></ElIcon>
          恢复
        </ElButton>

        <ElButton
          v-if="availableActions.includes('stop')"
          type="danger"
          :loading="loading"
          @click="$emit('stop')"
        >
          <ElIcon class="el-icon--left"><Close /></ElIcon>
          停止
        </ElButton>

        <ElButton
          v-if="availableActions.includes('redirect')"
          type="primary"
          plain
          :loading="loading"
          @click="showRedirectDialog = true"
        >
          <ElIcon class="el-icon--left"><Compass /></ElIcon>
          调整方向
        </ElButton>

        <ElButton
          v-if="availableActions.includes('checkpoint')"
          type="info"
          plain
          :loading="loading"
          @click="$emit('checkpoint')"
        >
          <ElIcon class="el-icon--left"><Flag /></ElIcon>
          创建检查点
        </ElButton>

        <ElButton @click="goToVisualization">
          <ElIcon class="el-icon--left"><DataLine /></ElIcon>
          查看可视化
        </ElButton>
      </div>
    </div>

    <!-- 调整方向对话框 -->
    <ElDialog v-model="showRedirectDialog" title="调整学习方向" width="500px">
      <ElForm :model="redirectForm" label-position="top">
        <ElFormItem label="新的学习方向" required>
          <ElInput
            v-model="redirectForm.newDirection"
            type="textarea"
            :rows="3"
            placeholder="请输入新的学习方向..."
          />
        </ElFormItem>
        <ElFormItem label="调整原因" required>
          <ElInput
            v-model="redirectForm.reason"
            type="textarea"
            :rows="2"
            placeholder="请输入调整原因..."
          />
        </ElFormItem>
      </ElForm>
      <template #footer>
        <ElButton @click="showRedirectDialog = false">取消</ElButton>
        <ElButton
          type="primary"
          :disabled="!redirectForm.newDirection || !redirectForm.reason"
          @click="handleRedirect"
        >
          确认调整
        </ElButton>
      </template>
    </ElDialog>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import type { FormInstance, FormRules } from 'element-plus'
import {
  VideoPlay,
  VideoPause,
  Close,
  Compass,
  Flag,
  DataLine
} from '@element-plus/icons-vue'
import type { LearningSession } from '@/types/nlgsm'
import { LearningControlAction } from '@/types/nlgsm/enums'

interface Props {
  session: LearningSession | null
  loading: boolean
  availableActions: LearningControlAction[]
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'start', goal: string, scope?: any): void
  (e: 'pause'): void
  (e: 'resume'): void
  (e: 'stop'): void
  (e: 'redirect', newDirection: string, reason: string): void
  (e: 'checkpoint'): void
}>()

const router = useRouter()
const startFormRef = ref<FormInstance>()
const showRedirectDialog = ref(false)

const startForm = ref({
  goal: ''
})

const redirectForm = ref({
  newDirection: '',
  reason: ''
})

const startRules: FormRules = {
  goal: [{ required: true, message: '请输入学习目标', trigger: 'blur' }]
}

const statusClass = computed(() => {
  if (!props.session) return 'inactive'
  if (props.session.isPaused) return 'paused'
  return 'active'
})

const statusLabel = computed(() => {
  if (!props.session) return '未启动'
  if (props.session.isPaused) return '已暂停'
  return '学习中'
})

const handleStart = async () => {
  if (!startFormRef.value) return

  await startFormRef.value.validate((valid) => {
    if (valid) {
      emit('start', startForm.value.goal)
      startForm.value.goal = ''
    }
  })
}

const handleRedirect = () => {
  emit('redirect', redirectForm.value.newDirection, redirectForm.value.reason)
  showRedirectDialog.value = false
  redirectForm.value = { newDirection: '', reason: '' }
}

const goToVisualization = () => {
  router.push({ name: 'LearningVisualization' })
}
</script>

<style scoped lang="scss">
.control-panel-card {
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;

    .status-badge {
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 500;

      &.inactive {
        background: var(--el-fill-color);
        color: var(--el-text-color-secondary);
      }

      &.active {
        background: rgba(76, 175, 80, 0.1);
        color: #4caf50;
      }

      &.paused {
        background: rgba(255, 152, 0, 0.1);
        color: #ff9800;
      }
    }
  }
}

.control-content {
  .start-form {
    max-width: 600px;
  }

  .control-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
  }
}
</style>
