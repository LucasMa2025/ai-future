<!-- Learning Unit 详情页面 -->
<template>
  <div class="lu-detail-page">
    <ElSkeleton :loading="loading" animated :rows="20">
      <template #default>
        <div v-if="lu" class="detail-content">
          <!-- 页头 -->
          <div class="page-header">
            <div class="header-left">
              <ElButton :icon="ArrowLeft" @click="goBack">返回</ElButton>
              <div class="title-section">
                <h2>{{ lu.id }}</h2>
                <div class="badges">
                  <ElTag v-if="lu.riskLevel" :type="getRiskTagType(lu.riskLevel)" size="large">
                    {{ getRiskLabel(lu.riskLevel) }}
                  </ElTag>
                  <ElTag :type="getStatusTagType(lu.status)" size="large">
                    {{ getStatusLabel(lu.status) }}
                  </ElTag>
                </div>
              </div>
            </div>
            <div class="header-right">
              <ApprovalActions
                v-if="canApprove"
                :lu-id="lu.id"
                @approved="handleApproved"
                @rejected="handleRejected"
              />
            </div>
          </div>

          <ElRow :gutter="20">
            <!-- 左侧：基本信息 + 知识内容 -->
            <ElCol :sm="24" :lg="16">
              <LUInfo :lu="lu" />
              <LUKnowledge :lu="lu" />
            </ElCol>

            <!-- 右侧：探索路径 -->
            <ElCol :sm="24" :lg="8">
              <LUExploration :steps="explorationSteps" :loading="explorationLoading" />
            </ElCol>
          </ElRow>
        </div>

        <ElEmpty v-else description="未找到该 Learning Unit" />
      </template>
    </ElSkeleton>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { learningUnitsApi } from '@/api/learning-units'
import type { LearningUnit, ExplorationStep } from '@/types/nlgsm'
import { LearningUnitStatus, RiskLevel, riskLabels } from '@/types/nlgsm/enums'
import LUInfo from './modules/lu-info.vue'
import LUKnowledge from './modules/lu-knowledge.vue'
import LUExploration from './modules/lu-exploration.vue'
import ApprovalActions from './modules/approval-actions.vue'

defineOptions({ name: 'LearningUnitDetail' })

const route = useRoute()
const router = useRouter()

const loading = ref(true)
const explorationLoading = ref(false)
const lu = ref<LearningUnit | null>(null)
const explorationSteps = ref<ExplorationStep[]>([])

const statusLabels: Record<string, string> = {
  [LearningUnitStatus.PENDING]: '待处理',
  [LearningUnitStatus.AUTO_CLASSIFIED]: '已分类',
  [LearningUnitStatus.HUMAN_REVIEW]: '人工审核',
  [LearningUnitStatus.APPROVED]: '已通过',
  [LearningUnitStatus.CORRECTED]: '已修正',
  [LearningUnitStatus.REJECTED]: '已拒绝',
  [LearningUnitStatus.TERMINATED]: '已终止'
}

const canApprove = computed(() => {
  if (!lu.value) return false
  return [LearningUnitStatus.AUTO_CLASSIFIED, LearningUnitStatus.HUMAN_REVIEW].includes(lu.value.status)
})

const getRiskLabel = (risk: RiskLevel) => riskLabels[risk] || risk
const getStatusLabel = (status: LearningUnitStatus) => statusLabels[status] || status

const getRiskTagType = (risk: RiskLevel) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    [RiskLevel.LOW]: 'success',
    [RiskLevel.MEDIUM]: 'warning',
    [RiskLevel.HIGH]: 'danger',
    [RiskLevel.CRITICAL]: 'danger'
  }
  return typeMap[risk] || 'info'
}

const getStatusTagType = (status: LearningUnitStatus) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    [LearningUnitStatus.PENDING]: 'info',
    [LearningUnitStatus.AUTO_CLASSIFIED]: '',
    [LearningUnitStatus.HUMAN_REVIEW]: 'warning',
    [LearningUnitStatus.APPROVED]: 'success',
    [LearningUnitStatus.CORRECTED]: 'success',
    [LearningUnitStatus.REJECTED]: 'danger',
    [LearningUnitStatus.TERMINATED]: 'info'
  }
  return typeMap[status] || 'info'
}

const fetchLU = async () => {
  const id = route.params.id as string
  loading.value = true
  try {
    lu.value = await learningUnitsApi.getById(id)
    explorationSteps.value = lu.value.explorationSteps || []
  } catch (e) {
    console.error('Failed to fetch LU:', e)
    lu.value = null
  } finally {
    loading.value = false
  }
}

const fetchExploration = async () => {
  if (!lu.value) return
  explorationLoading.value = true
  try {
    const result = await learningUnitsApi.getExplorationPath(lu.value.id)
    explorationSteps.value = result.steps
  } catch (e) {
    console.error('Failed to fetch exploration:', e)
  } finally {
    explorationLoading.value = false
  }
}

const goBack = () => {
  router.push({ name: 'LearningUnits' })
}

const handleApproved = () => {
  ElMessage.success('审批通过')
  fetchLU()
}

const handleRejected = () => {
  ElMessage.success('已拒绝')
  fetchLU()
}

onMounted(() => {
  fetchLU()
})
</script>

<style scoped lang="scss">
.lu-detail-page {
  .page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--el-border-color-lighter);

    .header-left {
      display: flex;
      align-items: flex-start;
      gap: 16px;

      .title-section {
        h2 {
          margin: 0 0 8px 0;
          font-size: 20px;
          font-weight: 600;
        }

        .badges {
          display: flex;
          gap: 8px;
        }
      }
    }
  }
}
</style>
