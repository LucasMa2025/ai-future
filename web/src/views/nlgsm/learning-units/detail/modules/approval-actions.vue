<!-- 审批操作面板 -->
<template>
  <div class="approval-actions">
    <ElButton type="success" :loading="loading" @click="handleApprove">
      <ElIcon class="el-icon--left"><Check /></ElIcon>
      审批通过
    </ElButton>

    <ElButton type="warning" :loading="loading" @click="showCorrectDialog = true">
      <ElIcon class="el-icon--left"><Edit /></ElIcon>
      修正
    </ElButton>

    <ElButton type="danger" :loading="loading" @click="showRejectDialog = true">
      <ElIcon class="el-icon--left"><Close /></ElIcon>
      拒绝
    </ElButton>

    <!-- 拒绝对话框 -->
    <ElDialog v-model="showRejectDialog" title="拒绝审批" width="500px">
      <ElForm label-position="top">
        <ElFormItem label="拒绝原因" required>
          <ElInput
            v-model="rejectReason"
            type="textarea"
            :rows="3"
            placeholder="请输入拒绝原因..."
          />
        </ElFormItem>
      </ElForm>
      <template #footer>
        <ElButton @click="showRejectDialog = false">取消</ElButton>
        <ElButton type="danger" :loading="loading" :disabled="!rejectReason" @click="handleReject">
          确认拒绝
        </ElButton>
      </template>
    </ElDialog>

    <!-- 修正对话框 -->
    <ElDialog v-model="showCorrectDialog" title="修正 Learning Unit" width="600px">
      <ElForm label-position="top">
        <ElFormItem label="修正说明" required>
          <ElInput
            v-model="correctComment"
            type="textarea"
            :rows="2"
            placeholder="请输入修正说明..."
          />
        </ElFormItem>
        <ElFormItem label="修正内容">
          <ElInput
            v-model="correctContent"
            type="textarea"
            :rows="5"
            placeholder="请输入修正后的知识内容（JSON 格式）..."
          />
        </ElFormItem>
      </ElForm>
      <template #footer>
        <ElButton @click="showCorrectDialog = false">取消</ElButton>
        <ElButton
          type="primary"
          :loading="loading"
          :disabled="!correctComment"
          @click="handleCorrect"
        >
          提交修正
        </ElButton>
      </template>
    </ElDialog>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Check, Close, Edit } from '@element-plus/icons-vue'
import { learningUnitsApi } from '@/api/learning-units'

interface Props {
  luId: string
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'approved'): void
  (e: 'rejected'): void
  (e: 'corrected'): void
}>()

const loading = ref(false)
const showRejectDialog = ref(false)
const showCorrectDialog = ref(false)
const rejectReason = ref('')
const correctComment = ref('')
const correctContent = ref('')

const handleApprove = async () => {
  try {
    await ElMessageBox.confirm('确定要通过此 Learning Unit 吗？', '确认审批', {
      confirmButtonText: '通过',
      cancelButtonText: '取消',
      type: 'info'
    })

    loading.value = true
    await learningUnitsApi.approve(props.luId, '审批通过')
    emit('approved')
  } catch {
    // 用户取消
  } finally {
    loading.value = false
  }
}

const handleReject = async () => {
  if (!rejectReason.value) {
    ElMessage.warning('请输入拒绝原因')
    return
  }

  loading.value = true
  try {
    await learningUnitsApi.reject(props.luId, rejectReason.value)
    showRejectDialog.value = false
    rejectReason.value = ''
    emit('rejected')
  } catch (e) {
    console.error('Reject failed:', e)
  } finally {
    loading.value = false
  }
}

const handleCorrect = async () => {
  if (!correctComment.value) {
    ElMessage.warning('请输入修正说明')
    return
  }

  loading.value = true
  try {
    let corrections = {}
    if (correctContent.value) {
      try {
        corrections = JSON.parse(correctContent.value)
      } catch {
        ElMessage.error('修正内容格式错误，请输入有效的 JSON')
        loading.value = false
        return
      }
    }

    await learningUnitsApi.correct(props.luId, corrections, correctComment.value)
    showCorrectDialog.value = false
    correctComment.value = ''
    correctContent.value = ''
    emit('corrected')
    ElMessage.success('修正已提交')
  } catch (e) {
    console.error('Correct failed:', e)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped lang="scss">
.approval-actions {
  display: flex;
  gap: 12px;
}
</style>
