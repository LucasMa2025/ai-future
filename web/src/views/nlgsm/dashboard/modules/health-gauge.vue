<!-- 系统健康度仪表盘 -->
<template>
  <ElCard shadow="hover" class="health-gauge-card">
    <template #header>
      <div class="card-header">
        <span>系统健康度</span>
        <ElTag :type="healthStatus.type" size="small">{{ healthStatus.label }}</ElTag>
      </div>
    </template>

    <ElSkeleton :loading="loading" animated :rows="8">
      <template #default>
        <div class="health-content">
          <div ref="gaugeRef" class="gauge-container"></div>

          <div class="health-metrics">
            <div class="metric-item">
              <div class="metric-icon" style="background: rgba(76, 175, 80, 0.1)">
                <ElIcon color="#4caf50"><CircleCheck /></ElIcon>
              </div>
              <div class="metric-info">
                <div class="metric-value">{{ metrics.uptime }}%</div>
                <div class="metric-label">运行时间</div>
              </div>
            </div>

            <div class="metric-item">
              <div class="metric-icon" style="background: rgba(33, 150, 243, 0.1)">
                <ElIcon color="#2196f3"><Cpu /></ElIcon>
              </div>
              <div class="metric-info">
                <div class="metric-value">{{ metrics.cpu }}%</div>
                <div class="metric-label">CPU 使用</div>
              </div>
            </div>

            <div class="metric-item">
              <div class="metric-icon" style="background: rgba(156, 39, 176, 0.1)">
                <ElIcon color="#9c27b0"><Coin /></ElIcon>
              </div>
              <div class="metric-info">
                <div class="metric-value">{{ metrics.memory }}%</div>
                <div class="metric-label">内存使用</div>
              </div>
            </div>

            <div class="metric-item">
              <div class="metric-icon" style="background: rgba(255, 152, 0, 0.1)">
                <ElIcon color="#ff9800"><Timer /></ElIcon>
              </div>
              <div class="metric-info">
                <div class="metric-value">{{ metrics.latency }}ms</div>
                <div class="metric-label">响应延迟</div>
              </div>
            </div>
          </div>
        </div>
      </template>
    </ElSkeleton>
  </ElCard>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import * as echarts from 'echarts'
import { CircleCheck, Cpu, Coin, Timer } from '@element-plus/icons-vue'
import { useTheme } from '@/hooks/core/useTheme'

interface Props {
  health: number
  loading: boolean
}

const props = defineProps<Props>()
const { isDarkMode } = useTheme()

const gaugeRef = ref<HTMLElement>()
let chartInstance: echarts.ECharts | null = null

// 模拟指标数据
const metrics = ref({
  uptime: 99.9,
  cpu: 45,
  memory: 62,
  latency: 23
})

const healthStatus = computed(() => {
  const h = props.health
  if (h >= 90) return { label: '优秀', type: 'success' as const }
  if (h >= 70) return { label: '良好', type: '' as const }
  if (h >= 50) return { label: '一般', type: 'warning' as const }
  return { label: '异常', type: 'danger' as const }
})

const initChart = () => {
  if (!gaugeRef.value) return

  chartInstance = echarts.init(gaugeRef.value)
  updateChart()

  window.addEventListener('resize', handleResize)
}

const updateChart = () => {
  if (!chartInstance) return

  const healthValue = props.health || 85

  const option: echarts.EChartsOption = {
    series: [
      {
        type: 'gauge',
        startAngle: 200,
        endAngle: -20,
        min: 0,
        max: 100,
        splitNumber: 10,
        itemStyle: {
          color: getHealthColor(healthValue)
        },
        progress: {
          show: true,
          width: 20
        },
        pointer: {
          show: false
        },
        axisLine: {
          lineStyle: {
            width: 20,
            color: [[1, isDarkMode.value ? '#333' : '#e0e0e0']]
          }
        },
        axisTick: {
          show: false
        },
        splitLine: {
          show: false
        },
        axisLabel: {
          show: false
        },
        anchor: {
          show: false
        },
        title: {
          show: false
        },
        detail: {
          valueAnimation: true,
          width: '60%',
          lineHeight: 40,
          borderRadius: 8,
          offsetCenter: [0, '10%'],
          fontSize: 36,
          fontWeight: 'bold',
          formatter: '{value}%',
          color: isDarkMode.value ? '#fff' : '#333'
        },
        data: [
          {
            value: healthValue
          }
        ]
      }
    ]
  }

  chartInstance.setOption(option)
}

const getHealthColor = (value: number) => {
  if (value >= 90) return '#4caf50'
  if (value >= 70) return '#2196f3'
  if (value >= 50) return '#ff9800'
  return '#f44336'
}

const handleResize = () => {
  chartInstance?.resize()
}

watch(
  () => props.health,
  () => {
    updateChart()
  }
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
.health-gauge-card {
  height: 100%;
  margin-bottom: 20px;

  .card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
}

.health-content {
  .gauge-container {
    height: 200px;
    width: 100%;
  }

  .health-metrics {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 16px;

    .metric-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 12px;
      background: var(--el-fill-color-light);
      border-radius: 8px;

      .metric-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 8px;
      }

      .metric-info {
        .metric-value {
          font-size: 18px;
          font-weight: 600;
          color: var(--el-text-color-primary);
        }

        .metric-label {
          font-size: 12px;
          color: var(--el-text-color-secondary);
        }
      }
    }
  }
}
</style>
