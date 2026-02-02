/**
 * 通知状态管理
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Notification } from '@/types/nlgsm'

export const useNotificationsStore = defineStore(
  'notifications',
  () => {
    // ==================== State ====================
    const notifications = ref<Notification[]>([])
    const maxNotifications = ref(100)

    // ==================== Computed ====================
    const unreadCount = computed(() => notifications.value.filter((n) => !n.read).length)

    const recentNotifications = computed(() => notifications.value.slice(0, 10))

    const hasUnread = computed(() => unreadCount.value > 0)

    // ==================== Actions ====================

    /**
     * 添加通知
     */
    const addNotification = (notification: Omit<Notification, 'id' | 'read' | 'createdAt'>) => {
      const newNotification: Notification = {
        ...notification,
        id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        read: false,
        createdAt: new Date().toISOString()
      }

      notifications.value.unshift(newNotification)

      // 限制通知数量
      if (notifications.value.length > maxNotifications.value) {
        notifications.value = notifications.value.slice(0, maxNotifications.value)
      }

      return newNotification
    }

    /**
     * 标记为已读
     */
    const markAsRead = (id: string) => {
      const notification = notifications.value.find((n) => n.id === id)
      if (notification) {
        notification.read = true
      }
    }

    /**
     * 标记全部为已读
     */
    const markAllAsRead = () => {
      notifications.value.forEach((n) => {
        n.read = true
      })
    }

    /**
     * 删除通知
     */
    const removeNotification = (id: string) => {
      const index = notifications.value.findIndex((n) => n.id === id)
      if (index !== -1) {
        notifications.value.splice(index, 1)
      }
    }

    /**
     * 清空所有通知
     */
    const clearAll = () => {
      notifications.value = []
    }

    /**
     * 清空已读通知
     */
    const clearRead = () => {
      notifications.value = notifications.value.filter((n) => !n.read)
    }

    /**
     * 处理 WebSocket 通知
     */
    const handleWebSocketNotification = (data: {
      type: string
      title?: string
      message?: string
      data?: Record<string, any>
    }) => {
      const typeMap: Record<string, string> = {
        learning_paused: '学习已暂停',
        learning_resumed: '学习已恢复',
        learning_stopped: '学习已停止',
        learning_redirected: '学习方向已调整',
        checkpoint_created: '检查点已创建',
        state_changed: '状态已变更',
        approval_required: '需要审批',
        anomaly_detected: '检测到异常',
        safe_halt_triggered: '安全停机已触发'
      }

      addNotification({
        type: data.type,
        title: data.title || typeMap[data.type] || '系统通知',
        message: data.message || '',
        metadata: data.data
      })
    }

    return {
      // State
      notifications,
      maxNotifications,

      // Computed
      unreadCount,
      recentNotifications,
      hasUnread,

      // Actions
      addNotification,
      markAsRead,
      markAllAsRead,
      removeNotification,
      clearAll,
      clearRead,
      handleWebSocketNotification
    }
  },
  {
    persist: {
      key: 'notifications',
      storage: localStorage,
      pick: ['notifications']
    }
  }
)
