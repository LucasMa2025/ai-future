<!-- LU 知识内容 -->
<template>
  <ElCard shadow="hover" class="knowledge-card">
    <template #header>
      <div class="card-header">
        <span>知识内容</span>
        <ElButton size="small" @click="expanded = !expanded">
          {{ expanded ? '收起' : '展开' }}
        </ElButton>
      </div>
    </template>

    <div class="knowledge-content" :class="{ expanded }">
      <pre>{{ lu.knowledge || '暂无知识内容' }}</pre>
    </div>
  </ElCard>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import type { LearningUnit } from '@/types/nlgsm'

interface Props {
  lu: LearningUnit
}

defineProps<Props>()

const expanded = ref(false)
</script>

<style scoped lang="scss">
.knowledge-card {
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.knowledge-content {
  max-height: 200px;
  overflow: hidden;
  transition: max-height 0.3s ease;

  &.expanded {
    max-height: none;
  }

  pre {
    margin: 0;
    padding: 16px;
    background: var(--el-fill-color-light);
    border-radius: 8px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
  }
}
</style>
