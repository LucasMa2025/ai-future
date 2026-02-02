<!-- NLGSM 治理概览页面 -->
<template>
  <div class="nlgsm-dashboard">
    <!-- 统计卡片 -->
    <StatsCards :stats="stats" :loading="loading" />

    <ElRow :gutter="20">
      <!-- 状态机概览 -->
      <ElCol :sm="24" :md="12" :lg="10">
        <StateFlow :current-state="currentState" :loading="loading" />
      </ElCol>
      <!-- 事件时间线 -->
      <ElCol :sm="24" :md="12" :lg="14">
        <EventTimeline :events="timelineEvents" :loading="loading" />
      </ElCol>
    </ElRow>

    <ElRow :gutter="20">
      <!-- 风险趋势图 -->
      <ElCol :sm="24" :md="12" :lg="12">
        <RiskChart :loading="loading" />
      </ElCol>
      <!-- 系统健康度 -->
      <ElCol :sm="24" :md="12" :lg="12">
        <HealthGauge :health="stats?.systemHealth || 0" :loading="loading" />
      </ElCol>
    </ElRow>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue'
import { useStateMachineStore } from '@/store/modules/state-machine'
import StatsCards from './modules/stats-cards.vue'
import StateFlow from './modules/state-flow.vue'
import EventTimeline from './modules/event-timeline.vue'
import RiskChart from './modules/risk-chart.vue'
import HealthGauge from './modules/health-gauge.vue'

defineOptions({ name: 'NLGSMDashboard' })

const stateMachineStore = useStateMachineStore()

const loading = computed(() => stateMachineStore.loading)
const stats = computed(() => stateMachineStore.dashboardStats)
const currentState = computed(() => stateMachineStore.currentState)
const timelineEvents = computed(() => stateMachineStore.timelineEvents)

onMounted(async () => {
  await stateMachineStore.initialize()
})
</script>

<style scoped lang="scss">
.nlgsm-dashboard {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
