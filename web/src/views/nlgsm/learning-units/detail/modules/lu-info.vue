<!-- LU 基本信息 -->
<template>
  <ElCard shadow="hover" class="info-card">
    <template #header>
      <span>基本信息</span>
    </template>

    <ElDescriptions :column="2" border>
      <ElDescriptionsItem label="ID" :span="2">
        {{ lu.id }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="版本">
        v{{ lu.version }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="来源">
        {{ lu.source || '-' }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="学习目标" :span="2">
        {{ lu.learningGoal }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="创建时间">
        {{ formatTime(lu.createdAt) }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="更新时间">
        {{ formatTime(lu.updatedAt) }}
      </ElDescriptionsItem>

      <ElDescriptionsItem v-if="lu.provenance" label="来源追溯" :span="2">
        <div class="provenance-info">
          <div>原始来源: {{ lu.provenance.origin }}</div>
          <div>链深度: {{ lu.provenance.chainDepth }}</div>
          <div v-if="lu.provenance.parentLuId">
            父 LU:
            <ElButton link type="primary" @click="goToParent">
              {{ lu.provenance.parentLuId }}
            </ElButton>
          </div>
        </div>
      </ElDescriptionsItem>

      <ElDescriptionsItem v-if="lu.approvers?.length" label="审批人" :span="2">
        <ElTag v-for="approver in lu.approvers" :key="approver" size="small" class="approver-tag">
          {{ approver }}
        </ElTag>
      </ElDescriptionsItem>

      <ElDescriptionsItem v-if="lu.approvedAt" label="审批时间">
        {{ formatTime(lu.approvedAt) }}
      </ElDescriptionsItem>

      <ElDescriptionsItem v-if="lu.rejectedReason" label="拒绝原因" :span="2">
        <span class="reject-reason">{{ lu.rejectedReason }}</span>
      </ElDescriptionsItem>
    </ElDescriptions>

    <!-- 约束条件 -->
    <div v-if="lu.constraints?.length" class="constraints-section">
      <h4>约束条件</h4>
      <ElTable :data="lu.constraints" stripe size="small">
        <ElTableColumn prop="condition" label="条件" min-width="200" />
        <ElTableColumn prop="decision" label="决策" width="150" />
      </ElTable>
    </div>
  </ElCard>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import type { LearningUnit } from '@/types/nlgsm'

interface Props {
  lu: LearningUnit
}

const props = defineProps<Props>()
const router = useRouter()

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const goToParent = () => {
  if (props.lu.provenance?.parentLuId) {
    router.push({
      name: 'LearningUnitDetail',
      params: { id: props.lu.provenance.parentLuId }
    })
  }
}
</script>

<style scoped lang="scss">
.info-card {
  margin-bottom: 20px;
}

.provenance-info {
  font-size: 13px;
  line-height: 1.8;
}

.approver-tag {
  margin-right: 8px;
}

.reject-reason {
  color: var(--el-color-danger);
}

.constraints-section {
  margin-top: 20px;

  h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--el-text-color-primary);
  }
}
</style>
