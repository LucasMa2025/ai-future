<!-- 学习可视化页面 -->
<template>
  <div class="learning-visualization-page">
    <!-- 页头 -->
    <div class="page-header">
      <ElButton :icon="ArrowLeft" @click="goBack">返回控制面板</ElButton>
      <ElButton :icon="Refresh" :loading="loading" @click="fetchData">刷新数据</ElButton>
    </div>

    <ElSkeleton :loading="loading && !visualizationData" animated :rows="20">
      <template #default>
        <div v-if="visualizationData" class="visualization-content">
          <!-- 统计摘要 -->
          <StatisticsSummary :statistics="visualizationData.statistics" />

          <ElRow :gutter="20">
            <!-- 进度曲线图 -->
            <ElCol :sm="24" :lg="12">
              <ProgressCurve :data="visualizationData.progressCurve" />
            </ElCol>

            <!-- 状态流转图 -->
            <ElCol :sm="24" :lg="12">
              <StateFlowGraph :data="visualizationData.stateFlow" />
            </ElCol>
          </ElRow>

          <!-- 事件时间线 -->
          <EventTimeline
            :events="visualizationData.timelineEvents"
            :checkpoints="visualizationData.checkpointMarkers"
            :direction-changes="visualizationData.directionChangeMarkers"
          />
        </div>

        <ElEmpty v-else description="暂无可视化数据" />
      </template>
    </ElSkeleton>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ArrowLeft, Refresh } from '@element-plus/icons-vue'
import { useLearningControlStore } from '@/store/modules/learning-control'
import type { LearningVisualizationData } from '@/types/nlgsm'
import StatisticsSummary from './modules/statistics-summary.vue'
import ProgressCurve from './modules/progress-curve.vue'
import StateFlowGraph from './modules/state-flow-graph.vue'
import EventTimeline from './modules/event-timeline.vue'

defineOptions({ name: 'LearningVisualization' })

const router = useRouter()
const store = useLearningControlStore()

const loading = ref(false)
const visualizationData = ref<LearningVisualizationData | null>(null)

let pollInterval: ReturnType<typeof setInterval> | null = null

const fetchData = async () => {
  loading.value = true
  try {
    await store.fetchVisualizationData()
    visualizationData.value = store.visualizationData
  } catch (e) {
    console.error('Failed to fetch visualization data:', e)
  } finally {
    loading.value = false
  }
}

const goBack = () => {
  router.push({ name: 'LearningControl' })
}

// 实时更新
const startPolling = () => {
  pollInterval = setInterval(fetchData, 5000) // 5秒更新一次
}

const stopPolling = () => {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

onMounted(() => {
  fetchData()
  startPolling()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped lang="scss">
.learning-visualization-page {
  display: flex;
  flex-direction: column;
  gap: 20px;

  .page-header {
    display: flex;
    gap: 12px;
  }
}
</style>
