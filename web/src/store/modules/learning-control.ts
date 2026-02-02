/**
 * 学习控制状态管理 (v4.0)
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { learningControlApi } from '@/api/learning-control'
import type {
  LearningSession,
  Checkpoint,
  LearningVisualizationData,
  LearningProgress,
  LearningScope
} from '@/types/nlgsm'
import { LearningSessionState, LearningControlAction } from '@/types/nlgsm/enums'

export const useLearningControlStore = defineStore(
  'learningControl',
  () => {
    // ==================== State ====================
    const currentSession = ref<LearningSession | null>(null)
    const checkpoints = ref<Checkpoint[]>([])
    const visualizationData = ref<LearningVisualizationData | null>(null)
    const sessionHistory = ref<LearningSession[]>([])
    const loading = ref(false)
    const error = ref<string | null>(null)

    // ==================== Computed ====================
    const isActive = computed(() => currentSession.value !== null)

    const isPaused = computed(() => currentSession.value?.isPaused ?? false)

    const currentState = computed(() => currentSession.value?.state ?? LearningSessionState.FROZEN)

    const progress = computed(() => currentSession.value?.progress ?? null)

    const availableActions = computed((): LearningControlAction[] => {
      if (!currentSession.value) {
        return [LearningControlAction.START]
      }

      const state = currentSession.value.state

      if (state === LearningSessionState.LEARNING) {
        return [
          LearningControlAction.PAUSE,
          LearningControlAction.STOP,
          LearningControlAction.REDIRECT,
          LearningControlAction.CHECKPOINT
        ]
      } else if (state === LearningSessionState.PAUSED) {
        return [
          LearningControlAction.RESUME,
          LearningControlAction.STOP,
          LearningControlAction.REDIRECT,
          LearningControlAction.ROLLBACK
        ]
      }

      return [LearningControlAction.START]
    })

    const stateLabel = computed(() => {
      const labels: Record<string, string> = {
        learning: '学习中',
        paused: '已暂停',
        frozen: '已停止',
        validation: '验证中'
      }
      return currentSession.value ? labels[currentSession.value.state] : '未开始'
    })

    // ==================== Actions ====================

    /**
     * 获取当前会话
     */
    const fetchCurrentSession = async () => {
      loading.value = true
      error.value = null
      try {
        const response = await learningControlApi.getCurrentSession()
        if (response.success && response.data) {
          currentSession.value = response.data
        } else {
          currentSession.value = null
        }
      } catch (e) {
        error.value = e instanceof Error ? e.message : '获取会话失败'
        currentSession.value = null
      } finally {
        loading.value = false
      }
    }

    /**
     * 启动学习
     */
    const startLearning = async (goal: string, scope?: LearningScope) => {
      loading.value = true
      error.value = null
      try {
        const response = await learningControlApi.startSession({ goal, scope })
        if (response.success) {
          await fetchCurrentSession()
        }
        return response
      } catch (e) {
        error.value = e instanceof Error ? e.message : '启动学习失败'
        throw e
      } finally {
        loading.value = false
      }
    }

    /**
     * 暂停学习
     */
    const pauseLearning = async (reason: string) => {
      loading.value = true
      try {
        const response = await learningControlApi.pause({ reason })
        if (response.success) {
          await fetchCurrentSession()
        }
        return response
      } finally {
        loading.value = false
      }
    }

    /**
     * 恢复学习
     */
    const resumeLearning = async () => {
      loading.value = true
      try {
        const response = await learningControlApi.resume()
        if (response.success) {
          await fetchCurrentSession()
        }
        return response
      } finally {
        loading.value = false
      }
    }

    /**
     * 停止学习
     */
    const stopLearning = async (reason: string, saveProgress = true) => {
      loading.value = true
      try {
        const response = await learningControlApi.stop({ reason, saveProgress })
        if (response.success) {
          currentSession.value = null
        }
        return response
      } finally {
        loading.value = false
      }
    }

    /**
     * 调整学习方向
     */
    const redirectLearning = async (
      newDirection: string,
      reason: string,
      newScope?: LearningScope
    ) => {
      loading.value = true
      try {
        const response = await learningControlApi.redirect({
          newDirection,
          reason,
          newScope
        })
        if (response.success) {
          await fetchCurrentSession()
        }
        return response
      } finally {
        loading.value = false
      }
    }

    /**
     * 创建检查点
     */
    const createCheckpoint = async (reason = 'manual', metadata?: Record<string, any>) => {
      try {
        const response = await learningControlApi.createCheckpoint({ reason, metadata })
        if (response.success) {
          await fetchCheckpoints()
        }
        return response
      } catch (e) {
        throw e
      }
    }

    /**
     * 回滚到检查点
     */
    const rollbackToCheckpoint = async (checkpointId: string, reason: string) => {
      loading.value = true
      try {
        const response = await learningControlApi.rollback({ checkpointId, reason })
        if (response.success) {
          await fetchCurrentSession()
        }
        return response
      } finally {
        loading.value = false
      }
    }

    /**
     * 获取检查点列表
     */
    const fetchCheckpoints = async (sessionId?: string) => {
      try {
        const response = await learningControlApi.getCheckpoints(sessionId)
        if (response.success) {
          checkpoints.value = response.data
        }
      } catch (e) {
        console.error('Failed to fetch checkpoints:', e)
      }
    }

    /**
     * 获取可视化数据
     */
    const fetchVisualizationData = async (sessionId?: string) => {
      try {
        const response = await learningControlApi.getVisualizationData(sessionId)
        if (response.success) {
          visualizationData.value = response.data
        }
      } catch (e) {
        console.error('Failed to fetch visualization data:', e)
      }
    }

    /**
     * 获取会话历史
     */
    const fetchSessionHistory = async (limit = 20) => {
      try {
        const response = await learningControlApi.getSessionHistory(limit)
        if (response.success) {
          sessionHistory.value = response.data
        }
      } catch (e) {
        console.error('Failed to fetch session history:', e)
      }
    }

    /**
     * 初始化
     */
    const initialize = async () => {
      await Promise.all([fetchCurrentSession(), fetchCheckpoints(), fetchSessionHistory()])
    }

    return {
      // State
      currentSession,
      checkpoints,
      visualizationData,
      sessionHistory,
      loading,
      error,

      // Computed
      isActive,
      isPaused,
      currentState,
      progress,
      availableActions,
      stateLabel,

      // Actions
      fetchCurrentSession,
      startLearning,
      pauseLearning,
      resumeLearning,
      stopLearning,
      redirectLearning,
      createCheckpoint,
      rollbackToCheckpoint,
      fetchCheckpoints,
      fetchVisualizationData,
      fetchSessionHistory,
      initialize
    }
  },
  {
    persist: false
  }
)
