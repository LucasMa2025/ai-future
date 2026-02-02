<!-- 审计搜索 -->
<template>
  <ElCard shadow="hover" class="search-card">
    <ElForm inline>
      <ElFormItem label="操作类型">
        <ElSelect
          :model-value="action"
          placeholder="全部"
          clearable
          filterable
          style="width: 140px"
          @update:model-value="$emit('update:action', $event)"
        >
          <ElOption v-for="item in actionTypes" :key="item" :label="item" :value="item" />
        </ElSelect>
      </ElFormItem>

      <ElFormItem label="操作者">
        <ElInput
          :model-value="actor"
          placeholder="用户名"
          clearable
          style="width: 140px"
          @update:model-value="$emit('update:actor', $event)"
        />
      </ElFormItem>

      <ElFormItem label="目标类型">
        <ElSelect
          :model-value="targetType"
          placeholder="全部"
          clearable
          filterable
          style="width: 140px"
          @update:model-value="$emit('update:targetType', $event)"
        >
          <ElOption v-for="item in targetTypes" :key="item" :label="item" :value="item" />
        </ElSelect>
      </ElFormItem>

      <ElFormItem label="时间范围">
        <ElDatePicker
          :model-value="dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          value-format="YYYY-MM-DD"
          style="width: 240px"
          @update:model-value="$emit('update:dateRange', $event || [])"
        />
      </ElFormItem>

      <ElFormItem>
        <ElButton type="primary" @click="$emit('search')">查询</ElButton>
        <ElButton @click="$emit('reset')">重置</ElButton>
        <ElButton type="success" @click="$emit('export')">导出</ElButton>
      </ElFormItem>
    </ElForm>
  </ElCard>
</template>

<script setup lang="ts">
interface Props {
  action: string
  actor: string
  targetType: string
  dateRange: string[]
  actionTypes: string[]
  targetTypes: string[]
}

defineProps<Props>()

defineEmits<{
  (e: 'update:action', value: string): void
  (e: 'update:actor', value: string): void
  (e: 'update:targetType', value: string): void
  (e: 'update:dateRange', value: string[]): void
  (e: 'search'): void
  (e: 'reset'): void
  (e: 'export'): void
}>()
</script>

<style scoped lang="scss">
.search-card {
  :deep(.el-form-item) {
    margin-bottom: 0;
  }
}
</style>
