<!-- 审批详情对话框 -->
<template>
  <ElDialog
    :model-value="modelValue"
    title="审批详情"
    width="700px"
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <template v-if="approval">
      <ElDescriptions :column="2" border>
        <ElDescriptionsItem label="审批 ID" :span="2">
          {{ approval.id }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="目标类型">
          {{ getTargetTypeLabel(approval.targetType) }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="目标 ID">
          <ElButton link type="primary" @click="goToTarget">
            {{ approval.targetId }}
          </ElButton>
        </ElDescriptionsItem>

        <ElDescriptionsItem label="风险等级">
          <ElTag :type="getRiskTagType(approval.riskLevel)" size="small">
            {{ getRiskLabel(approval.riskLevel) }}
          </ElTag>
        </ElDescriptionsItem>

        <ElDescriptionsItem label="状态">
          <ElTag :type="getStatusTagType(approval.status)" size="small">
            {{ getStatusLabel(approval.status) }}
          </ElTag>
        </ElDescriptionsItem>

        <ElDescriptionsItem label="所需审批人数">
          {{ approval.requiredApprovers }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="已审批人数">
          {{ approval.currentApprovers?.length || 0 }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="创建时间">
          {{ formatTime(approval.createdAt) }}
        </ElDescriptionsItem>

        <ElDescriptionsItem label="过期时间">
          {{ approval.expiresAt ? formatTime(approval.expiresAt) : '-' }}
        </ElDescriptionsItem>
      </ElDescriptions>

      <!-- 审批历史 -->
      <div v-if="approval.decisions?.length" class="decisions-section">
        <h4>审批历史</h4>
        <ElTimeline>
          <ElTimelineItem
            v-for="decision in approval.decisions"
            :key="decision.createdAt"
            :type="getDecisionTimelineType(decision.decision)"
            :timestamp="formatTime(decision.createdAt)"
          >
            <div class="decision-item">
              <strong>{{ decision.username }}</strong>
              <ElTag :type="getDecisionTagType(decision.decision)" size="small">
                {{ getDecisionLabel(decision.decision) }}
              </ElTag>
              <div v-if="decision.comment" class="decision-comment">
                {{ decision.comment }}
              </div>
            </div>
          </ElTimelineItem>
        </ElTimeline>
      </div>

      <!-- 审批操作 -->
      <div v-if="approval.status === 'pending'" class="action-section">
        <ElDivider />
        <ElForm label-position="top">
          <ElFormItem label="审批意见">
            <ElInput
              v-model="comment"
              type="textarea"
              :rows="3"
              placeholder="请输入审批意见（可选）..."
            />
          </ElFormItem>
        </ElForm>
      </div>
    </template>

    <template #footer>
      <ElButton @click="$emit('update:modelValue', false)">关闭</ElButton>
      <template v-if="approval?.status === 'pending'">
        <ElButton type="danger" @click="handleReject">拒绝</ElButton>
        <ElButton type="success" @click="handleApprove">通过</ElButton>
      </template>
    </template>
  </ElDialog>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import type { Approval } from '@/types/nlgsm'
import { RiskLevel, riskLabels, ApprovalStatus, ApprovalDecision } from '@/types/nlgsm/enums'

interface Props {
  modelValue: boolean
  approval: Approval | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'approve', id: string, comment: string): void
  (e: 'reject', id: string, comment: string): void
}>()

const router = useRouter()
const comment = ref('')

watch(
  () => props.modelValue,
  (val) => {
    if (!val) {
      comment.value = ''
    }
  }
)

const targetTypeLabels: Record<string, string> = {
  learning_unit: 'Learning Unit',
  artifact: '工件',
  state_transition: '状态转换'
}

const statusLabels: Record<string, string> = {
  pending: '待审批',
  completed: '已完成',
  rejected: '已拒绝',
  expired: '已过期'
}

const decisionLabels: Record<string, string> = {
  approve: '通过',
  reject: '拒绝',
  correct: '修正',
  terminate: '终止'
}

const getTargetTypeLabel = (type: string) => targetTypeLabels[type] || type
const getStatusLabel = (status: ApprovalStatus) => statusLabels[status] || status
const getRiskLabel = (risk: RiskLevel) => riskLabels[risk] || risk
const getDecisionLabel = (decision: string) => decisionLabels[decision] || decision

const getRiskTagType = (risk: RiskLevel) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    low: 'success',
    medium: 'warning',
    high: 'danger',
    critical: 'danger'
  }
  return typeMap[risk] || 'info'
}

const getStatusTagType = (status: ApprovalStatus) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    pending: 'warning',
    completed: 'success',
    rejected: 'danger',
    expired: 'info'
  }
  return typeMap[status] || 'info'
}

const getDecisionTagType = (decision: string) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    approve: 'success',
    reject: 'danger',
    correct: 'warning',
    terminate: 'danger'
  }
  return typeMap[decision] || 'info'
}

const getDecisionTimelineType = (decision: string) => {
  const typeMap: Record<string, 'primary' | 'success' | 'warning' | 'danger' | 'info'> = {
    approve: 'success',
    reject: 'danger',
    correct: 'warning',
    terminate: 'danger'
  }
  return typeMap[decision] || 'primary'
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const goToTarget = () => {
  if (!props.approval) return

  const { targetType, targetId } = props.approval

  if (targetType === 'learning_unit') {
    router.push({ name: 'LearningUnitDetail', params: { id: targetId } })
  } else if (targetType === 'artifact') {
    router.push({ name: 'ArtifactDetail', params: { id: targetId } })
  }

  emit('update:modelValue', false)
}

const handleApprove = () => {
  if (props.approval) {
    emit('approve', props.approval.id, comment.value)
  }
}

const handleReject = () => {
  if (props.approval) {
    emit('reject', props.approval.id, comment.value)
  }
}
</script>

<style scoped lang="scss">
.decisions-section {
  margin-top: 20px;

  h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
  }

  .decision-item {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;

    .decision-comment {
      width: 100%;
      margin-top: 4px;
      font-size: 13px;
      color: var(--el-text-color-secondary);
    }
  }
}

.action-section {
  margin-top: 20px;
}
</style>
