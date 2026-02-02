<!-- 审批中心页面 -->
<template>
  <div class="approvals-page">
    <!-- 统计卡片 -->
    <ElRow :gutter="16" class="stats-row">
      <ElCol :xs="12" :sm="6" :md="4">
        <div class="stat-item pending" @click="switchTab('pending')">
          <div class="stat-value">{{ stats?.myPending || 0 }}</div>
          <div class="stat-label">待我审批</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="4">
        <div class="stat-item" @click="switchTab('all')">
          <div class="stat-value">{{ stats?.pending || 0 }}</div>
          <div class="stat-label">全部待审</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="4">
        <div class="stat-item completed">
          <div class="stat-value">{{ stats?.completed || 0 }}</div>
          <div class="stat-label">已完成</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="4">
        <div class="stat-item rejected">
          <div class="stat-value">{{ stats?.rejected || 0 }}</div>
          <div class="stat-label">已拒绝</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="4">
        <div class="stat-item expired">
          <div class="stat-value">{{ stats?.expired || 0 }}</div>
          <div class="stat-label">已过期</div>
        </div>
      </ElCol>
    </ElRow>

    <!-- 主内容 -->
    <ElCard shadow="hover">
      <template #header>
        <div class="card-header">
          <ElTabs v-model="activeTab" @tab-change="handleTabChange">
            <ElTabPane label="待我审批" name="pending" />
            <ElTabPane label="我的审批" name="my" />
            <ElTabPane label="全部" name="all" />
          </ElTabs>
          <ElButton :icon="Refresh" size="small" @click="fetchList">刷新</ElButton>
        </div>
      </template>

      <!-- 搜索 -->
      <ApprovalSearch
        v-model:status="searchParams.status"
        v-model:target-type="searchParams.targetType"
        @search="handleSearch"
        @reset="handleReset"
      />

      <!-- 表格 -->
      <ApprovalTable
        :data="list"
        :loading="loading"
        @view="handleView"
        @approve="handleApprove"
        @reject="handleReject"
      />

      <!-- 分页 -->
      <div class="pagination-wrapper">
        <ElPagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :total="pagination.total"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </ElCard>

    <!-- 详情对话框 -->
    <ApprovalDetailDialog
      v-model="detailVisible"
      :approval="selectedApproval"
      @approve="handleApproveSubmit"
      @reject="handleRejectSubmit"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { approvalsApi } from '@/api/approvals'
import type { Approval } from '@/types/nlgsm'
import ApprovalSearch from './modules/approval-search.vue'
import ApprovalTable from './modules/approval-table.vue'
import ApprovalDetailDialog from './modules/approval-detail-dialog.vue'

defineOptions({ name: 'Approvals' })

const loading = ref(false)
const activeTab = ref('pending')
const list = ref<Approval[]>([])
const stats = ref<any>(null)
const detailVisible = ref(false)
const selectedApproval = ref<Approval | null>(null)

const searchParams = reactive({
  status: '',
  targetType: ''
})

const pagination = reactive({
  page: 1,
  pageSize: 20,
  total: 0
})

const fetchList = async () => {
  loading.value = true
  try {
    let result

    if (activeTab.value === 'pending') {
      const pendingList = await approvalsApi.getPending()
      result = { items: pendingList, total: pendingList.length }
    } else if (activeTab.value === 'my') {
      result = await approvalsApi.getMyApprovals({
        page: pagination.page,
        pageSize: pagination.pageSize
      })
    } else {
      result = await approvalsApi.getList({
        page: pagination.page,
        pageSize: pagination.pageSize,
        status: searchParams.status || undefined,
        targetType: searchParams.targetType || undefined
      })
    }

    list.value = result.items
    pagination.total = result.total
  } catch (e) {
    console.error('Failed to fetch approvals:', e)
  } finally {
    loading.value = false
  }
}

const fetchStats = async () => {
  try {
    stats.value = await approvalsApi.getStats()
  } catch (e) {
    console.error('Failed to fetch stats:', e)
  }
}

const switchTab = (tab: string) => {
  activeTab.value = tab
  pagination.page = 1
  fetchList()
}

const handleTabChange = () => {
  pagination.page = 1
  fetchList()
}

const handleSearch = () => {
  pagination.page = 1
  fetchList()
}

const handleReset = () => {
  searchParams.status = ''
  searchParams.targetType = ''
  pagination.page = 1
  fetchList()
}

const handleSizeChange = () => {
  pagination.page = 1
  fetchList()
}

const handlePageChange = () => {
  fetchList()
}

const handleView = (row: Approval) => {
  selectedApproval.value = row
  detailVisible.value = true
}

const handleApprove = (row: Approval) => {
  selectedApproval.value = row
  detailVisible.value = true
}

const handleReject = (row: Approval) => {
  selectedApproval.value = row
  detailVisible.value = true
}

const handleApproveSubmit = async (id: string, comment: string) => {
  try {
    await approvalsApi.submitDecision(id, { decision: 'approve', comment })
    ElMessage.success('审批通过')
    detailVisible.value = false
    fetchList()
    fetchStats()
  } catch (e) {
    console.error('Approve failed:', e)
  }
}

const handleRejectSubmit = async (id: string, comment: string) => {
  try {
    await approvalsApi.submitDecision(id, { decision: 'reject', comment })
    ElMessage.success('已拒绝')
    detailVisible.value = false
    fetchList()
    fetchStats()
  } catch (e) {
    console.error('Reject failed:', e)
  }
}

onMounted(() => {
  fetchList()
  fetchStats()
})
</script>

<style scoped lang="scss">
.approvals-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.stats-row {
  .stat-item {
    padding: 16px;
    background: var(--el-bg-color);
    border-radius: 8px;
    border: 1px solid var(--el-border-color-lighter);
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;

    &:hover {
      border-color: var(--el-color-primary);
    }

    .stat-value {
      font-size: 24px;
      font-weight: 600;
      color: var(--el-text-color-primary);
    }

    .stat-label {
      font-size: 12px;
      color: var(--el-text-color-secondary);
      margin-top: 4px;
    }

    &.pending .stat-value {
      color: #ff9800;
    }

    &.completed .stat-value {
      color: #4caf50;
    }

    &.rejected .stat-value {
      color: #f44336;
    }

    &.expired .stat-value {
      color: #9e9e9e;
    }
  }
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;

  :deep(.el-tabs__header) {
    margin-bottom: 0;
  }
}

.pagination-wrapper {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}
</style>
