<!-- 审批搜索筛选 -->
<template>
  <div class="search-bar">
    <ElForm inline>
      <ElFormItem label="状态">
        <ElSelect
          :model-value="status"
          placeholder="全部状态"
          clearable
          style="width: 120px"
          @update:model-value="$emit('update:status', $event)"
        >
          <ElOption label="待审批" value="pending" />
          <ElOption label="已完成" value="completed" />
          <ElOption label="已拒绝" value="rejected" />
          <ElOption label="已过期" value="expired" />
        </ElSelect>
      </ElFormItem>

      <ElFormItem label="目标类型">
        <ElSelect
          :model-value="targetType"
          placeholder="全部类型"
          clearable
          style="width: 140px"
          @update:model-value="$emit('update:targetType', $event)"
        >
          <ElOption label="Learning Unit" value="learning_unit" />
          <ElOption label="工件" value="artifact" />
          <ElOption label="状态转换" value="state_transition" />
        </ElSelect>
      </ElFormItem>

      <ElFormItem>
        <ElButton type="primary" @click="$emit('search')">查询</ElButton>
        <ElButton @click="$emit('reset')">重置</ElButton>
      </ElFormItem>
    </ElForm>
  </div>
</template>

<script setup lang="ts">
interface Props {
  status: string
  targetType: string
}

defineProps<Props>()

defineEmits<{
  (e: 'update:status', value: string): void
  (e: 'update:targetType', value: string): void
  (e: 'search'): void
  (e: 'reset'): void
}>()
</script>

<style scoped lang="scss">
.search-bar {
  margin-bottom: 16px;

  :deep(.el-form-item) {
    margin-bottom: 0;
  }
}
</style>
