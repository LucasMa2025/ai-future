<!-- 版本差异对比 -->
<template>
  <ElCard shadow="hover" class="diff-card">
    <template #header>
      <div class="card-header">
        <span>版本对比</span>
        <div class="version-selector">
          <ElSelect v-model="fromVersion" placeholder="从版本" size="small" style="width: 100px">
            <ElOption
              v-for="ver in availableFromVersions"
              :key="ver"
              :label="`v${ver}`"
              :value="ver"
            />
          </ElSelect>
          <span class="arrow">→</span>
          <ElSelect v-model="toVersion" placeholder="到版本" size="small" style="width: 100px">
            <ElOption
              v-for="ver in availableToVersions"
              :key="ver"
              :label="`v${ver}`"
              :value="ver"
            />
          </ElSelect>
          <ElButton
            type="primary"
            size="small"
            :disabled="!canCompare"
            :loading="loading"
            @click="fetchDiff"
          >
            对比
          </ElButton>
        </div>
      </div>
    </template>

    <div v-if="diffList.length > 0" class="diff-content">
      <div v-for="(diff, index) in diffList" :key="index" class="diff-item">
        <div class="diff-header">
          <span class="field-name">{{ diff.field }}</span>
          <ElTag :type="getChangeTagType(diff.changeType)" size="small">
            {{ getChangeTypeLabel(diff.changeType) }}
          </ElTag>
        </div>
        <div class="diff-body">
          <div v-if="diff.changeType !== 'added'" class="old-value">
            <span class="value-label">旧值:</span>
            <code>{{ formatValue(diff.oldValue) }}</code>
          </div>
          <div v-if="diff.changeType !== 'removed'" class="new-value">
            <span class="value-label">新值:</span>
            <code>{{ formatValue(diff.newValue) }}</code>
          </div>
        </div>
      </div>
    </div>

    <ElEmpty v-else-if="compared" description="两个版本没有差异" :image-size="60" />
    <div v-else class="placeholder-text">选择版本进行对比</div>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { artifactsApi } from '@/api/artifacts'
import type { VersionDiff } from '@/types/nlgsm'

interface Props {
  artifactId: string
  currentVersion: number
}

const props = defineProps<Props>()

const loading = ref(false)
const compared = ref(false)
const fromVersion = ref(0)
const toVersion = ref(0)
const diffList = ref<VersionDiff[]>([])

const availableFromVersions = computed(() => {
  const versions = []
  for (let i = 1; i <= props.currentVersion; i++) {
    versions.push(i)
  }
  return versions
})

const availableToVersions = computed(() => {
  const versions = []
  for (let i = 1; i <= props.currentVersion; i++) {
    if (i !== fromVersion.value) {
      versions.push(i)
    }
  }
  return versions
})

const canCompare = computed(() => {
  return fromVersion.value > 0 && toVersion.value > 0 && fromVersion.value !== toVersion.value
})

const fetchDiff = async () => {
  if (!canCompare.value) return

  loading.value = true
  compared.value = false
  try {
    diffList.value = await artifactsApi.getDiff(props.artifactId, fromVersion.value, toVersion.value)
    compared.value = true
  } catch (e) {
    console.error('Failed to fetch diff:', e)
    diffList.value = []
  } finally {
    loading.value = false
  }
}

const getChangeTagType = (type: string) => {
  const typeMap: Record<string, 'success' | 'danger' | 'warning'> = {
    added: 'success',
    removed: 'danger',
    modified: 'warning'
  }
  return typeMap[type] || 'info'
}

const getChangeTypeLabel = (type: string) => {
  const labelMap: Record<string, string> = {
    added: '新增',
    removed: '删除',
    modified: '修改'
  }
  return labelMap[type] || type
}

const formatValue = (value: any) => {
  if (typeof value === 'object') {
    return JSON.stringify(value, null, 2)
  }
  return String(value)
}
</script>

<style scoped lang="scss">
.diff-card {
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;

    .version-selector {
      display: flex;
      align-items: center;
      gap: 8px;

      .arrow {
        color: var(--el-text-color-secondary);
      }
    }
  }

  .diff-content {
    .diff-item {
      padding: 12px;
      margin-bottom: 12px;
      background: var(--el-fill-color-light);
      border-radius: 8px;

      &:last-child {
        margin-bottom: 0;
      }

      .diff-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;

        .field-name {
          font-weight: 600;
          font-size: 13px;
        }
      }

      .diff-body {
        font-size: 13px;

        .old-value,
        .new-value {
          display: flex;
          gap: 8px;
          padding: 6px 0;

          .value-label {
            color: var(--el-text-color-secondary);
            white-space: nowrap;
          }

          code {
            flex: 1;
            padding: 4px 8px;
            background: var(--el-bg-color);
            border-radius: 4px;
            word-break: break-all;
            white-space: pre-wrap;
          }
        }

        .old-value code {
          background: rgba(244, 67, 54, 0.1);
        }

        .new-value code {
          background: rgba(76, 175, 80, 0.1);
        }
      }
    }
  }

  .placeholder-text {
    text-align: center;
    padding: 40px;
    color: var(--el-text-color-secondary);
  }
}
</style>
