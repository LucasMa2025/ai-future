<!-- 进度曲线图 -->
<template>
  <ElCard shadow="hover" class="progress-curve-card">
    <template #header>
      <span>进度曲线</span>
    </template>

    <div ref="chartRef" class="chart-container"></div>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { useTheme } from '@/hooks/core/useTheme'
import type { ProgressCurvePoint } from '@/types/nlgsm'

interface Props {
  data: ProgressCurvePoint[]
}

const props = defineProps<Props>()
const { isDarkMode } = useTheme()

const chartRef = ref<HTMLElement>()
let chartInstance: echarts.ECharts | null = null

const initChart = () => {
  if (!chartRef.value) return

  chartInstance = echarts.init(chartRef.value)
  updateChart()

  window.addEventListener('resize', handleResize)
}

const updateChart = () => {
  if (!chartInstance) return

  const times = props.data.map((d) => {
    const date = new Date(d.timestamp)
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  })
  const progress = props.data.map((d) => d.progressPercent)
  const depth = props.data.map((d) => d.depth)

  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: {
      data: ['进度', '深度'],
      bottom: 0,
      textStyle: {
        color: isDarkMode.value ? '#ccc' : '#666'
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '15%',
      top: '10%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: times,
      axisLine: {
        lineStyle: { color: isDarkMode.value ? '#555' : '#ddd' }
      },
      axisLabel: {
        color: isDarkMode.value ? '#aaa' : '#666'
      }
    },
    yAxis: [
      {
        type: 'value',
        name: '进度 %',
        min: 0,
        max: 100,
        splitLine: {
          lineStyle: { color: isDarkMode.value ? '#333' : '#eee' }
        },
        axisLabel: {
          color: isDarkMode.value ? '#aaa' : '#666'
        }
      },
      {
        type: 'value',
        name: '深度',
        splitLine: { show: false },
        axisLabel: {
          color: isDarkMode.value ? '#aaa' : '#666'
        }
      }
    ],
    series: [
      {
        name: '进度',
        type: 'line',
        smooth: true,
        areaStyle: {
          opacity: 0.3,
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(76, 175, 80, 0.8)' },
            { offset: 1, color: 'rgba(76, 175, 80, 0.1)' }
          ])
        },
        data: progress,
        itemStyle: { color: '#4caf50' }
      },
      {
        name: '深度',
        type: 'line',
        yAxisIndex: 1,
        smooth: true,
        data: depth,
        itemStyle: { color: '#9c27b0' }
      }
    ]
  }

  chartInstance.setOption(option)
}

const handleResize = () => {
  chartInstance?.resize()
}

watch(
  () => props.data,
  () => {
    updateChart()
  },
  { deep: true }
)

watch(isDarkMode, () => {
  updateChart()
})

onMounted(() => {
  initChart()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance?.dispose()
})
</script>

<style scoped lang="scss">
.progress-curve-card {
  margin-bottom: 20px;

  .chart-container {
    height: 300px;
    width: 100%;
  }
}
</style>
