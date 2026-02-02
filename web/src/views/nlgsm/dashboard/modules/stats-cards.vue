<!-- 统计卡片组 -->
<template>
  <ElRow :gutter="20" class="stats-cards">
    <ElCol v-for="card in cardList" :key="card.key" :xs="24" :sm="12" :md="8" :lg="4" :xl="4">
      <ElCard shadow="hover" :body-style="{ padding: '20px' }" class="stat-card">
        <ElSkeleton :loading="loading" animated :rows="2">
          <template #default>
            <div class="stat-card-content">
              <div class="stat-icon" :style="{ backgroundColor: card.bgColor }">
                <ElIcon :size="24" :color="card.iconColor">
                  <component :is="card.icon" />
                </ElIcon>
              </div>
              <div class="stat-info">
                <div class="stat-value">
                  <ArtCountTo
                    :target="card.value"
                    :duration="1000"
                    :decimals="0"
                    separator=","
                  />
                </div>
                <div class="stat-label">{{ card.label }}</div>
              </div>
            </div>
          </template>
        </ElSkeleton>
      </ElCard>
    </ElCol>
  </ElRow>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { DashboardStats } from '@/types/nlgsm'
import { stateLabels } from '@/types/nlgsm/enums'
import ArtCountTo from '@/components/core/text-effect/art-count-to/index.vue'
import {
  Odometer,
  CircleCheck,
  Refresh,
  Box,
  Warning
} from '@element-plus/icons-vue'

interface Props {
  stats: DashboardStats | null
  loading: boolean
}

const props = defineProps<Props>()

const cardList = computed(() => {
  const s = props.stats
  return [
    {
      key: 'state',
      label: '当前状态',
      value: 1, // 状态用图标表示
      displayValue: s?.currentState ? stateLabels[s.currentState] : '未知',
      icon: Odometer,
      iconColor: '#9c27b0',
      bgColor: 'rgba(156, 39, 176, 0.1)'
    },
    {
      key: 'pending',
      label: '待审批',
      value: s?.pendingApprovals || 0,
      icon: CircleCheck,
      iconColor: '#ff9800',
      bgColor: 'rgba(255, 152, 0, 0.1)'
    },
    {
      key: 'transitions',
      label: '今日转换',
      value: s?.todayTransitions || 0,
      icon: Refresh,
      iconColor: '#2196f3',
      bgColor: 'rgba(33, 150, 243, 0.1)'
    },
    {
      key: 'artifacts',
      label: '活跃工件',
      value: s?.activeArtifacts || 0,
      icon: Box,
      iconColor: '#4caf50',
      bgColor: 'rgba(76, 175, 80, 0.1)'
    },
    {
      key: 'anomalies',
      label: '未处理异常',
      value: s?.openAnomalies || 0,
      icon: Warning,
      iconColor: '#f44336',
      bgColor: 'rgba(244, 67, 54, 0.1)'
    }
  ]
})
</script>

<style scoped lang="scss">
.stats-cards {
  margin-bottom: 0;
}

.stat-card {
  height: 100%;
  margin-bottom: 20px;

  .stat-card-content {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .stat-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    border-radius: 12px;
    flex-shrink: 0;
  }

  .stat-info {
    flex: 1;
    min-width: 0;
  }

  .stat-value {
    font-size: 28px;
    font-weight: 600;
    line-height: 1.2;
    color: var(--el-text-color-primary);
  }

  .stat-label {
    margin-top: 4px;
    font-size: 14px;
    color: var(--el-text-color-secondary);
  }
}
</style>
