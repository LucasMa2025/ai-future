<!-- 工件详情页面 -->
<template>
  <div class="artifact-detail-page">
    <ElSkeleton :loading="loading" animated :rows="20">
      <template #default>
        <div v-if="artifact" class="detail-content">
          <!-- 页头 -->
          <div class="page-header">
            <div class="header-left">
              <ElButton :icon="ArrowLeft" @click="goBack">返回</ElButton>
              <div class="title-section">
                <h2>{{ artifact.id }}</h2>
                <div class="badges">
                  <ElTag effect="dark" size="large">v{{ artifact.version }}</ElTag>
                  <ElTag :type="artifact.isApproved ? 'success' : 'warning'" size="large">
                    {{ artifact.isApproved ? '已审批' : '待审批' }}
                  </ElTag>
                </div>
              </div>
            </div>
          </div>

          <ElRow :gutter="20">
            <!-- 左侧 -->
            <ElCol :sm="24" :lg="16">
              <!-- 基本信息 -->
              <ElCard shadow="hover" class="info-card">
                <template #header>基本信息</template>
                <ElDescriptions :column="2" border>
                  <ElDescriptionsItem label="ID">{{ artifact.id }}</ElDescriptionsItem>
                  <ElDescriptionsItem label="版本">v{{ artifact.version }}</ElDescriptionsItem>
                  <ElDescriptionsItem label="NLGSM 状态">
                    <ElTag :color="getStateColor(artifact.nlState)" effect="dark" size="small">
                      {{ getStateLabel(artifact.nlState) }}
                    </ElTag>
                  </ElDescriptionsItem>
                  <ElDescriptionsItem label="风险分数">
                    {{ (artifact.riskScore * 100).toFixed(1) }}%
                  </ElDescriptionsItem>
                  <ElDescriptionsItem label="审批人" :span="2">
                    <ElTag
                      v-for="approver in artifact.approvers"
                      :key="approver"
                      size="small"
                      class="approver-tag"
                    >
                      {{ approver }}
                    </ElTag>
                  </ElDescriptionsItem>
                  <ElDescriptionsItem label="完整性哈希" :span="2">
                    <code>{{ artifact.integrityHash }}</code>
                    <ElButton
                      link
                      type="primary"
                      size="small"
                      @click="verifyIntegrity"
                      :loading="verifying"
                    >
                      验证
                    </ElButton>
                  </ElDescriptionsItem>
                  <ElDescriptionsItem label="创建时间">
                    {{ formatTime(artifact.createdAt) }}
                  </ElDescriptionsItem>
                  <ElDescriptionsItem label="更新时间">
                    {{ formatTime(artifact.updatedAt) }}
                  </ElDescriptionsItem>
                </ElDescriptions>
              </ElCard>

              <!-- 版本对比 -->
              <VersionDiff
                :artifact-id="artifact.id"
                :current-version="artifact.version"
              />
            </ElCol>

            <!-- 右侧 -->
            <ElCol :sm="24" :lg="8">
              <!-- 指标 -->
              <ElCard shadow="hover" class="metrics-card">
                <template #header>性能指标</template>
                <div class="metrics-list">
                  <div
                    v-for="(value, key) in artifact.metrics"
                    :key="key"
                    class="metric-item"
                  >
                    <span class="metric-key">{{ key }}</span>
                    <span class="metric-value">{{ formatMetricValue(value) }}</span>
                  </div>
                </div>
              </ElCard>

              <!-- 快照 -->
              <ElCard shadow="hover" class="snapshot-card">
                <template #header>
                  <div class="card-header">
                    <span>参数快照</span>
                    <ElButton size="small" @click="snapshotExpanded = !snapshotExpanded">
                      {{ snapshotExpanded ? '收起' : '展开' }}
                    </ElButton>
                  </div>
                </template>
                <div class="snapshot-content" :class="{ expanded: snapshotExpanded }">
                  <pre>{{ JSON.stringify(artifact.snapshot, null, 2) }}</pre>
                </div>
              </ElCard>
            </ElCol>
          </ElRow>
        </div>

        <ElEmpty v-else description="未找到该工件" />
      </template>
    </ElSkeleton>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { artifactsApi } from '@/api/artifacts'
import type { Artifact } from '@/types/nlgsm'
import { NLGSMState, stateLabels, stateColors } from '@/types/nlgsm/enums'
import VersionDiff from './modules/version-diff.vue'

defineOptions({ name: 'ArtifactDetail' })

const route = useRoute()
const router = useRouter()

const loading = ref(true)
const verifying = ref(false)
const snapshotExpanded = ref(false)
const artifact = ref<Artifact | null>(null)

const getStateLabel = (state: NLGSMState) => stateLabels[state] || state
const getStateColor = (state: NLGSMState) => stateColors[state] || '#9e9e9e'

const formatTime = (time: string) => {
  if (!time) return '-'
  return new Date(time).toLocaleString('zh-CN')
}

const formatMetricValue = (value: number) => {
  if (typeof value !== 'number') return value
  return value.toFixed(4)
}

const fetchArtifact = async () => {
  const id = route.params.id as string
  loading.value = true
  try {
    artifact.value = await artifactsApi.getById(id)
  } catch (e) {
    console.error('Failed to fetch artifact:', e)
    artifact.value = null
  } finally {
    loading.value = false
  }
}

const verifyIntegrity = async () => {
  if (!artifact.value) return

  verifying.value = true
  try {
    const result = await artifactsApi.verifyIntegrity(artifact.value.id)
    if (result.valid) {
      ElMessage.success('完整性验证通过')
    } else {
      ElMessage.error('完整性验证失败：哈希不匹配')
    }
  } catch (e) {
    ElMessage.error('验证失败')
  } finally {
    verifying.value = false
  }
}

const goBack = () => {
  router.push({ name: 'Artifacts' })
}

onMounted(() => {
  fetchArtifact()
})
</script>

<style scoped lang="scss">
.artifact-detail-page {
  .page-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--el-border-color-lighter);

    .header-left {
      display: flex;
      align-items: flex-start;
      gap: 16px;

      .title-section {
        h2 {
          margin: 0 0 8px 0;
          font-size: 20px;
          font-weight: 600;
        }

        .badges {
          display: flex;
          gap: 8px;
        }
      }
    }
  }

  .info-card,
  .metrics-card,
  .snapshot-card {
    margin-bottom: 20px;
  }

  .approver-tag {
    margin-right: 8px;
  }

  .metrics-list {
    .metric-item {
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid var(--el-border-color-lighter);

      &:last-child {
        border-bottom: none;
      }

      .metric-key {
        font-size: 13px;
        color: var(--el-text-color-secondary);
      }

      .metric-value {
        font-size: 13px;
        font-weight: 600;
        font-family: monospace;
      }
    }
  }

  .snapshot-card {
    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .snapshot-content {
      max-height: 200px;
      overflow: hidden;
      transition: max-height 0.3s;

      &.expanded {
        max-height: none;
      }

      pre {
        margin: 0;
        padding: 12px;
        background: var(--el-fill-color-light);
        border-radius: 8px;
        font-size: 12px;
        overflow-x: auto;
      }
    }
  }
}
</style>
