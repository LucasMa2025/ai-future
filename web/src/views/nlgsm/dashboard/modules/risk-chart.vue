<!-- 风险趋势图 -->
<template>
  <ElCard shadow="hover" class="risk-chart-card">
    <template #header>
      <div class="card-header">
        <span>风险趋势</span>
        <ElRadioGroup v-model="timeRange" size="small">
          <ElRadioButton label="7d">7天</ElRadioButton>
          <ElRadioButton label="30d">30天</ElRadioButton>
        </ElRadioGroup>
      </div>
    </template>

    <ElSkeleton :loading="loading" animated :rows="8">
      <template #default>
        <div ref="chartRef" class="chart-container"></div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { useTheme } from '@/hooks/core/useTheme'

interface Props {
  loading: boolean
}

defineProps<Props>()

const { isDarkMode } = useTheme()
const chartRef = ref<HTMLElement>()
const timeRange = ref('7d')
let chartInstance: echarts.ECharts | null = null

// 模拟数据
const generateMockData = (days: number) => {
  const data = []
  const now = new Date()
  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(now)
    date.setDate(date.getDate() - i)
    data.push({
      date: date.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' }),
      low: Math.floor(Math.random() * 10) + 5,
      medium: Math.floor(Math.random() * 8) + 2,
      high: Math.floor(Math.random() * 5),
      critical: Math.floor(Math.random() * 2)
    })
  }
  return data
}

const initChart = () => {
  if (!chartRef.value) return

  chartInstance = echarts.init(chartRef.value)
  updateChart()

  window.addEventListener('resize', handleResize)
}

const updateChart = () => {
  if (!chartInstance) return

  const days = timeRange.value === '7d' ? 7 : 30
  const data = generateMockData(days)

  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'cross' }
    },
    legend: {
      data: ['低风险', '中风险', '高风险', '严重'],
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
      data: data.map((d) => d.date),
      axisLine: {
        lineStyle: { color: isDarkMode.value ? '#555' : '#ddd' }
      },
      axisLabel: {
        color: isDarkMode.value ? '#aaa' : '#666'
      }
    },
    yAxis: {
      type: 'value',
      splitLine: {
        lineStyle: { color: isDarkMode.value ? '#333' : '#eee' }
      },
      axisLabel: {
        color: isDarkMode.value ? '#aaa' : '#666'
      }
    },
    series: [
      {
        name: '低风险',
        type: 'line',
        stack: 'Total',
        smooth: true,
        areaStyle: { opacity: 0.3 },
        emphasis: { focus: 'series' },
        data: data.map((d) => d.low),
        itemStyle: { color: '#4caf50' }
      },
      {
        name: '中风险',
        type: 'line',
        stack: 'Total',
        smooth: true,
        areaStyle: { opacity: 0.3 },
        emphasis: { focus: 'series' },
        data: data.map((d) => d.medium),
        itemStyle: { color: '#ff9800' }
      },
      {
        name: '高风险',
        type: 'line',
        stack: 'Total',
        smooth: true,
        areaStyle: { opacity: 0.3 },
        emphasis: { focus: 'series' },
        data: data.map((d) => d.high),
        itemStyle: { color: '#f44336' }
      },
      {
        name: '严重',
        type: 'line',
        stack: 'Total',
        smooth: true,
        areaStyle: { opacity: 0.3 },
        emphasis: { focus: 'series' },
        data: data.map((d) => d.critical),
        itemStyle: { color: '#b71c1c' }
      }
    ]
  }

  chartInstance.setOption(option)
}

const handleResize = () => {
  chartInstance?.resize()
}

watch(timeRange, () => {
  updateChart()
})

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
.risk-chart-card {
  height: 100%;
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .chart-container {
    height: 300px;
    width: 100%;
  }
}
</style>
