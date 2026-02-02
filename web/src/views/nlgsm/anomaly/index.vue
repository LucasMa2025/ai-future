<!-- 异常监控页面 -->
<template>
  <div class="anomaly-page">
    <ElRow :gutter="20">
      <!-- 实时告警 -->
      <ElCol :sm="24" :lg="16">
        <AnomalyList
          :data="list"
          :loading="loading"
          @resolve="handleResolve"
          @ignore="handleIgnore"
          @refresh="fetchList"
        />
      </ElCol>

      <!-- 右侧面板 -->
      <ElCol :sm="24" :lg="8">
        <!-- 检测器状态 -->
        <DetectorStatus
          :detectors="detectors"
          :loading="detectorsLoading"
          @update="handleUpdateDetector"
          @trigger="handleTriggerDetection"
        />

        <!-- 阈值配置 -->
        <ThresholdConfig
          :detectors="detectors"
          @update="handleUpdateDetector"
        />
      </ElCol>
    </ElRow>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { anomalyApi } from '@/api/anomaly'
import type { AnomalyEvent, DetectorStatus as DetectorStatusType } from '@/types/nlgsm'
import AnomalyList from './modules/anomaly-list.vue'
import DetectorStatus from './modules/detector-status.vue'
import ThresholdConfig from './modules/threshold-config.vue'

defineOptions({ name: 'AnomalyMonitor' })

const loading = ref(false)
const detectorsLoading = ref(false)
const list = ref<AnomalyEvent[]>([])
const detectors = ref<DetectorStatusType[]>([])

let pollInterval: ReturnType<typeof setInterval> | null = null

const fetchList = async () => {
  loading.value = true
  try {
    const result = await anomalyApi.getList({
      pageSize: 50,
      status: 'open'
    })
    list.value = result.items
  } catch (e) {
    console.error('Failed to fetch anomalies:', e)
  } finally {
    loading.value = false
  }
}

const fetchDetectors = async () => {
  detectorsLoading.value = true
  try {
    detectors.value = await anomalyApi.getDetectors()
  } catch (e) {
    console.error('Failed to fetch detectors:', e)
  } finally {
    detectorsLoading.value = false
  }
}

const handleResolve = async (row: AnomalyEvent) => {
  try {
    const { value } = await ElMessageBox.prompt('请输入解决方案描述', '解决异常', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      inputPlaceholder: '请描述如何解决此异常...',
      inputValidator: (val) => !!val || '请输入解决方案'
    })

    await anomalyApi.resolve(row.id, { resolution: value })
    ElMessage.success('异常已解决')
    fetchList()
  } catch {
    // 用户取消
  }
}

const handleIgnore = async (row: AnomalyEvent) => {
  try {
    const { value } = await ElMessageBox.prompt('请输入忽略原因', '忽略异常', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      inputPlaceholder: '请说明忽略此异常的原因...',
      inputValidator: (val) => !!val || '请输入忽略原因',
      type: 'warning'
    })

    await anomalyApi.ignore(row.id, value)
    ElMessage.success('异常已忽略')
    fetchList()
  } catch {
    // 用户取消
  }
}

const handleUpdateDetector = async (id: string, data: { enabled?: boolean; threshold?: number }) => {
  try {
    await anomalyApi.updateDetector(id, data)
    ElMessage.success('配置已更新')
    fetchDetectors()
  } catch (e) {
    ElMessage.error('更新失败')
  }
}

const handleTriggerDetection = async (detectorId?: string) => {
  try {
    const result = await anomalyApi.triggerDetection(detectorId)
    const detected = result.results.filter((r) => r.anomalyDetected).length

    if (detected > 0) {
      ElMessage.warning(`检测完成，发现 ${detected} 个异常`)
    } else {
      ElMessage.success('检测完成，未发现异常')
    }

    fetchList()
  } catch (e) {
    ElMessage.error('检测失败')
  }
}

// 定时刷新
const startPolling = () => {
  pollInterval = setInterval(() => {
    fetchList()
  }, 30000) // 30秒刷新一次
}

const stopPolling = () => {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

onMounted(() => {
  fetchList()
  fetchDetectors()
  startPolling()
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped lang="scss">
.anomaly-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
</style>
