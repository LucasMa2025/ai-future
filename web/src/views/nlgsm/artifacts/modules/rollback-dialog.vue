<!-- 回滚对话框 -->
<template>
  <ElDialog
    :model-value="modelValue"
    title="回滚工件"
    width="500px"
    @update:model-value="$emit('update:modelValue', $event)"
  >
    <template v-if="artifact">
      <ElAlert
        title="回滚操作将创建新版本并恢复到目标版本的状态"
        type="warning"
        :closable="false"
        show-icon
        class="rollback-alert"
      />

      <ElForm label-position="top" :model="form" :rules="rules" ref="formRef">
        <ElFormItem label="当前版本">
          <ElInput :model-value="`v${artifact.version}`" disabled />
        </ElFormItem>

        <ElFormItem label="目标版本" prop="targetVersion">
          <ElSelect v-model="form.targetVersion" placeholder="选择目标版本" style="width: 100%">
            <ElOption
              v-for="ver in availableVersions"
              :key="ver"
              :label="`v${ver}`"
              :value="ver"
            />
          </ElSelect>
        </ElFormItem>

        <ElFormItem label="回滚原因" prop="reason">
          <ElInput
            v-model="form.reason"
            type="textarea"
            :rows="3"
            placeholder="请输入回滚原因..."
          />
        </ElFormItem>
      </ElForm>
    </template>

    <template #footer>
      <ElButton @click="$emit('update:modelValue', false)">取消</ElButton>
      <ElButton type="warning" @click="handleConfirm">确认回滚</ElButton>
    </template>
  </ElDialog>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import type { FormInstance, FormRules } from 'element-plus'
import type { Artifact } from '@/types/nlgsm'

interface Props {
  modelValue: boolean
  artifact: Artifact | null
}

const props = defineProps<Props>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'confirm', targetVersion: number, reason: string): void
}>()

const formRef = ref<FormInstance>()
const form = ref({
  targetVersion: 0,
  reason: ''
})

const rules: FormRules = {
  targetVersion: [{ required: true, message: '请选择目标版本', trigger: 'change' }],
  reason: [{ required: true, message: '请输入回滚原因', trigger: 'blur' }]
}

const availableVersions = computed(() => {
  if (!props.artifact) return []
  const versions = []
  for (let i = 1; i < props.artifact.version; i++) {
    versions.push(i)
  }
  return versions.reverse()
})

watch(
  () => props.modelValue,
  (val) => {
    if (!val) {
      form.value = { targetVersion: 0, reason: '' }
    }
  }
)

const handleConfirm = async () => {
  if (!formRef.value) return

  await formRef.value.validate((valid) => {
    if (valid) {
      emit('confirm', form.value.targetVersion, form.value.reason)
    }
  })
}
</script>

<style scoped lang="scss">
.rollback-alert {
  margin-bottom: 20px;
}
</style>
