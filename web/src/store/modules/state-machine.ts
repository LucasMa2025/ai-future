/**
 * 状态机状态管理
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { stateMachineApi } from '@/api/state-machine'
import type {
  SystemState,
  StateTransition,
  AvailableTransition,
  DashboardStats,
  TimelineEvent,
  NLGSMState
} from '@/types/nlgsm'
import { stateLabels, stateColors } from '@/types/nlgsm/enums'

export const useStateMachineStore = defineStore(
  'stateMachine',
  () => {
    // ==================== State ====================
    const currentState = ref<SystemState | null>(null)
    const availableTransitions = ref<AvailableTransition[]>([])
    const history = ref<StateTransition[]>([])
    const historyTotal = ref(0)
    const dashboardStats = ref<DashboardStats | null>(null)
    const timelineEvents = ref<TimelineEvent[]>([])
    const loading = ref(false)
    const error = ref<string | null>(null)

    // ==================== Computed ====================
    const state = computed(() => currentState.value?.state || null)

    const stateLabel = computed(() => {
      if (!currentState.value?.state) return '未知'
      return stateLabels[currentState.value.state as NLGSMState] || currentState.value.state
    })

    const stateColor = computed(() => {
      if (!currentState.value?.state) return '#9e9e9e'
      return stateColors[currentState.value.state as NLGSMState] || '#9e9e9e'
    })

    const canTriggerEvent = computed(() => availableTransitions.value.length > 0)

    // ==================== Actions ====================

    /**
     * 获取当前状态
     */
    const fetchCurrentState = async () => {
      loading.value = true
      error.value = null
      try {
        currentState.value = await stateMachineApi.getCurrentState()
      } catch (e) {
        error.value = e instanceof Error ? e.message : '获取状态失败'
        currentState.value = null
      } finally {
        loading.value = false
      }
    }

    /**
     * 获取可用转换
     */
    const fetchAvailableTransitions = async () => {
      try {
        availableTransitions.value = await stateMachineApi.getAvailableTransitions()
      } catch (e) {
        availableTransitions.value = []
      }
    }

    /**
     * 触发事件
     */
    const triggerEvent = async (eventType: string, metadata?: Record<string, any>) => {
      loading.value = true
      try {
        const result = await stateMachineApi.triggerEvent({
          eventType,
          metadata
        })

        if (result.success) {
          // 刷新状态
          await fetchCurrentState()
          await fetchAvailableTransitions()
          await fetchHistory()
        }

        return result
      } finally {
        loading.value = false
      }
    }

    /**
     * 强制设置状态
     */
    const forceState = async (targetState: string, reason: string) => {
      loading.value = true
      try {
        const result = await stateMachineApi.forceState({ targetState, reason })

        if (result.success) {
          await fetchCurrentState()
          await fetchAvailableTransitions()
          await fetchHistory()
        }

        return result
      } finally {
        loading.value = false
      }
    }

    /**
     * 获取转换历史
     */
    const fetchHistory = async (page = 1, pageSize = 20) => {
      try {
        const result = await stateMachineApi.getHistory({ page, pageSize })
        history.value = result.items
        historyTotal.value = result.total
      } catch (e) {
        history.value = []
        historyTotal.value = 0
      }
    }

    /**
     * 获取仪表盘统计
     */
    const fetchDashboardStats = async () => {
      try {
        dashboardStats.value = await stateMachineApi.getDashboardStats()
      } catch (e) {
        dashboardStats.value = null
      }
    }

    /**
     * 获取时间线事件
     */
    const fetchTimelineEvents = async (limit = 20) => {
      try {
        timelineEvents.value = await stateMachineApi.getTimelineEvents({ limit })
      } catch (e) {
        timelineEvents.value = []
      }
    }

    /**
     * 初始化（获取所有数据）
     */
    const initialize = async () => {
      await Promise.all([
        fetchCurrentState(),
        fetchAvailableTransitions(),
        fetchDashboardStats(),
        fetchTimelineEvents(),
        fetchHistory()
      ])
    }

    return {
      // State
      currentState,
      availableTransitions,
      history,
      historyTotal,
      dashboardStats,
      timelineEvents,
      loading,
      error,

      // Computed
      state,
      stateLabel,
      stateColor,
      canTriggerEvent,

      // Actions
      fetchCurrentState,
      fetchAvailableTransitions,
      triggerEvent,
      forceState,
      fetchHistory,
      fetchDashboardStats,
      fetchTimelineEvents,
      initialize
    }
  },
  {
    persist: false
  }
)
