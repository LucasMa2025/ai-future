<!-- LU 搜索筛选 -->
<template>
  <ElCard shadow="hover" class="search-card">
    <ElForm :model="form" inline>
      <ElFormItem label="状态">
        <ElSelect
          :model-value="status"
          placeholder="全部状态"
          clearable
          style="width: 140px"
          @update:model-value="$emit('update:status', $event)"
        >
          <ElOption
            v-for="item in statusOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </ElSelect>
      </ElFormItem>

      <ElFormItem label="风险等级">
        <ElSelect
          :model-value="riskLevel"
          placeholder="全部等级"
          clearable
          style="width: 140px"
          @update:model-value="$emit('update:riskLevel', $event)"
        >
          <ElOption
            v-for="item in riskOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </ElSelect>
      </ElFormItem>

      <ElFormItem label="搜索">
        <ElInput
          :model-value="search"
          placeholder="ID / 学习目标"
          clearable
          style="width: 200px"
          @update:model-value="$emit('update:search', $event)"
          @keyup.enter="$emit('search')"
        />
      </ElFormItem>

      <ElFormItem>
        <ElButton type="primary" @click="$emit('search')">查询</ElButton>
        <ElButton @click="$emit('reset')">重置</ElButton>
      </ElFormItem>
    </ElForm>
  </ElCard>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { LearningUnitStatus, RiskLevel } from '@/types/nlgsm/enums'

interface Props {
  status: string
  riskLevel: string
  search: string
}

defineProps<Props>()

defineEmits<{
  (e: 'update:status', value: string): void
  (e: 'update:riskLevel', value: string): void
  (e: 'update:search', value: string): void
  (e: 'search'): void
  (e: 'reset'): void
}>()

const form = reactive({})

const statusOptions = [
  { value: LearningUnitStatus.PENDING, label: '待处理' },
  { value: LearningUnitStatus.AUTO_CLASSIFIED, label: '已分类' },
  { value: LearningUnitStatus.HUMAN_REVIEW, label: '人工审核' },
  { value: LearningUnitStatus.APPROVED, label: '已通过' },
  { value: LearningUnitStatus.CORRECTED, label: '已修正' },
  { value: LearningUnitStatus.REJECTED, label: '已拒绝' },
  { value: LearningUnitStatus.TERMINATED, label: '已终止' }
]

const riskOptions = [
  { value: RiskLevel.LOW, label: '低风险' },
  { value: RiskLevel.MEDIUM, label: '中风险' },
  { value: RiskLevel.HIGH, label: '高风险' },
  { value: RiskLevel.CRITICAL, label: '严重' }
]
</script>

<style scoped lang="scss">
.search-card {
  :deep(.el-form-item) {
    margin-bottom: 0;
  }
}
</style>
