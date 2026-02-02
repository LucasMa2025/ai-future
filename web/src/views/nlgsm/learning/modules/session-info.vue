<!-- 会话信息 -->
<template>
  <ElCard shadow="hover" class="session-info-card">
    <template #header>
      <span>会话信息</span>
    </template>

    <ElDescriptions :column="2" border>
      <ElDescriptionsItem label="会话 ID">
        {{ session.sessionId }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="状态">
        <ElTag :type="getStateTagType(session.state)" size="small">
          {{ getStateLabel(session.state) }}
        </ElTag>
      </ElDescriptionsItem>

      <ElDescriptionsItem label="学习目标" :span="2">
        {{ session.goal || '-' }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="开始时间">
        {{ formatTime(session.startedAt) }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="暂停时间">
        {{ session.pausedAt ? formatTime(session.pausedAt) : '-' }}
      </ElDescriptionsItem>

      <ElDescriptionsItem v-if="session.pauseReason" label="暂停原因" :span="2">
        {{ session.pauseReason }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="检查点数">
        {{ session.checkpoints?.length || 0 }}
      </ElDescriptionsItem>

      <ElDescriptionsItem label="方向变更次数">
        {{ session.directionChanges?.length || 0 }}
      </ElDescriptionsItem>
    </ElDescriptions>

    <!-- 方向变更历史 -->
    <div v-if="session.directionChanges?.length" class="direction-changes">
      <h4>方向变更历史</h4>
      <ElTimeline>
        <ElTimelineItem
          v-for="(change, index) in session.directionChanges.slice(0, 5)"
          :key="index"
          :timestamp="formatTime(change.timestamp)"
        >
          <div class="change-item">
            <div class="change-direction">
              <span class="old">{{ change.oldDirection }}</span>
              <ElIcon><ArrowRight /></ElIcon>
              <span class="new">{{ change.newDirection }}</span>
            </div>
            <div class="change-reason">原因: {{ change.reason }}</div>
            <div class="change-actor">操作者: {{ change.actor }}</div>
          </div>
        </ElTimelineItem>
      </ElTimeline>
    </div>
  </ElCard>
</template>

<script setup lang="ts">
import { ArrowRight } from '@element-plus/icons-vue'
import type { LearningSession } from '@/types/nlgsm'
import { LearningSessionState } from '@/types/nlgsm/enums'

interface Props {
  session: LearningSession
}

defineProps<Props>()

const stateLabels: Record<string, string> = {
  learning: '学习中',
  paused: '已暂停',
  frozen: '已停止',
  validation: '验证中'
}

const getStateLabel = (state: LearningSessionState) => stateLabels[state] || state

const getStateTagType = (state: LearningSessionState) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    learning: 'success',
    paused: 'warning',
    frozen: 'info',
    validation: ''
  }
  return typeMap[state] || 'info'
}

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}
</script>

<style scoped lang="scss">
.session-info-card {
  margin-bottom: 20px;
}

.direction-changes {
  margin-top: 20px;

  h4 {
    margin: 0 0 12px 0;
    font-size: 14px;
    font-weight: 600;
  }

  .change-item {
    .change-direction {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;

      .old {
        color: var(--el-text-color-secondary);
        text-decoration: line-through;
      }

      .new {
        font-weight: 500;
        color: var(--el-color-primary);
      }
    }

    .change-reason,
    .change-actor {
      font-size: 12px;
      color: var(--el-text-color-secondary);
      margin-top: 4px;
    }
  }
}
</style>
