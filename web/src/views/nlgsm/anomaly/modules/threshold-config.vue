<!-- 阈值配置 -->
<template>
  <ElCard shadow="hover" class="threshold-card">
    <template #header>
      <span>阈值配置</span>
    </template>

    <div class="threshold-list">
      <div
        v-for="detector in detectors"
        :key="detector.id"
        class="threshold-item"
      >
        <div class="item-label">{{ detector.name }}</div>
        <div class="item-control">
          <ElSlider
            :model-value="detector.threshold * 100"
            :min="0"
            :max="100"
            :step="5"
            show-stops
            :marks="marks"
            @change="handleChange(detector.id, $event)"
          />
        </div>
      </div>
    </div>
  </ElCard>
</template>

<script setup lang="ts">
import type { DetectorStatus } from '@/types/nlgsm'

interface Props {
  detectors: DetectorStatus[]
}

defineProps<Props>()

const emit = defineEmits<{
  (e: 'update', id: string, data: { enabled?: boolean; threshold?: number }): void
}>()

const marks = {
  0: '0%',
  25: '25%',
  50: '50%',
  75: '75%',
  100: '100%'
}

const handleChange = (id: string, value: number) => {
  emit('update', id, { threshold: value / 100 })
}
</script>

<style scoped lang="scss">
.threshold-card {
  .threshold-list {
    display: flex;
    flex-direction: column;
    gap: 24px;

    .threshold-item {
      .item-label {
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 8px;
      }

      .item-control {
        padding: 0 10px;
      }
    }
  }
}
</style>
