<!-- 状态流转图 -->
<template>
  <ElCard shadow="hover" class="state-flow-card">
    <template #header>
      <span>状态流转</span>
    </template>

    <div ref="chartRef" class="chart-container"></div>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import * as echarts from 'echarts'
import { useTheme } from '@/hooks/core/useTheme'
import type { StateFlowItem } from '@/types/nlgsm'
import { stateColors, stateLabels, NLGSMState } from '@/types/nlgsm/enums'

interface Props {
  data: StateFlowItem[]
}

const props = defineProps<Props>()
const { isDarkMode } = useTheme()

const chartRef = ref<HTMLElement>()
let chartInstance: echarts.ECharts | null = null

// 计算节点和边
const graphData = computed(() => {
  const nodes = new Map<string, { name: string; value: number }>()
  const links: { source: string; target: string; value: number }[] = []

  // 统计状态出现次数
  props.data.forEach((item) => {
    const from = item.fromState
    const to = item.toState

    if (!nodes.has(from)) {
      nodes.set(from, { name: stateLabels[from as NLGSMState] || from, value: 0 })
    }
    if (!nodes.has(to)) {
      nodes.set(to, { name: stateLabels[to as NLGSMState] || to, value: 0 })
    }

    nodes.get(to)!.value++

    // 查找是否已有相同的边
    const existingLink = links.find((l) => l.source === from && l.target === to)
    if (existingLink) {
      existingLink.value++
    } else {
      links.push({ source: from, target: to, value: 1 })
    }
  })

  return {
    nodes: Array.from(nodes.entries()).map(([id, data]) => ({
      id,
      name: data.name,
      value: data.value,
      itemStyle: {
        color: stateColors[id as NLGSMState] || '#9e9e9e'
      }
    })),
    links
  }
})

const initChart = () => {
  if (!chartRef.value) return

  chartInstance = echarts.init(chartRef.value)
  updateChart()

  window.addEventListener('resize', handleResize)
}

const updateChart = () => {
  if (!chartInstance) return

  const { nodes, links } = graphData.value

  const option: echarts.EChartsOption = {
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (params.dataType === 'node') {
          return `${params.name}: ${params.value} 次进入`
        }
        return `${params.data.source} → ${params.data.target}: ${params.data.value} 次`
      }
    },
    series: [
      {
        type: 'graph',
        layout: 'circular',
        symbolSize: 50,
        roam: true,
        label: {
          show: true,
          fontSize: 12,
          color: isDarkMode.value ? '#eee' : '#333'
        },
        edgeSymbol: ['circle', 'arrow'],
        edgeSymbolSize: [4, 10],
        edgeLabel: {
          show: true,
          fontSize: 10,
          formatter: '{c}'
        },
        data: nodes,
        links: links.map((link) => ({
          ...link,
          lineStyle: {
            width: Math.min(link.value * 2, 8),
            curveness: 0.2
          }
        })),
        lineStyle: {
          color: isDarkMode.value ? '#666' : '#aaa',
          opacity: 0.8
        }
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
.state-flow-card {
  margin-bottom: 20px;

  .chart-container {
    height: 300px;
    width: 100%;
  }
}
</style>
