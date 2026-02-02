<!-- 工件搜索 -->
<template>
  <ElCard shadow="hover" class="search-card">
    <ElForm inline>
      <ElFormItem label="搜索">
        <ElInput
          :model-value="search"
          placeholder="ID / 名称"
          clearable
          style="width: 200px"
          @update:model-value="$emit('update:search', $event)"
          @keyup.enter="$emit('search')"
        />
      </ElFormItem>

      <ElFormItem label="审批状态">
        <ElSelect
          :model-value="isApproved"
          placeholder="全部"
          clearable
          style="width: 120px"
          @update:model-value="$emit('update:isApproved', $event)"
        >
          <ElOption label="已审批" value="true" />
          <ElOption label="未审批" value="false" />
        </ElSelect>
      </ElFormItem>

      <ElFormItem>
        <ElButton type="primary" @click="$emit('search')">查询</ElButton>
        <ElButton @click="$emit('reset')">重置</ElButton>
      </ElFormItem>
    </ElForm>
  </ElCard>
</template>

<script setup lang="ts">
interface Props {
  search: string
  isApproved: string
}

defineProps<Props>()

defineEmits<{
  (e: 'update:search', value: string): void
  (e: 'update:isApproved', value: string): void
  (e: 'search'): void
  (e: 'reset'): void
}>()
</script>

<style scoped lang="scss">
.search-card {
  :deep(.el-form-item) {
    margin-bottom: 0;
  }
}
</style>
