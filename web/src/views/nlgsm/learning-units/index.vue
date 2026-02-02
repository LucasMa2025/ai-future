<!-- Learning Units 列表页面 -->
<template>
  <div class="learning-units-page">
    <!-- 搜索筛选 -->
    <LUSearch
      v-model:status="searchParams.status"
      v-model:risk-level="searchParams.riskLevel"
      v-model:search="searchParams.search"
      @search="handleSearch"
      @reset="handleReset"
    />

    <!-- 统计卡片 -->
    <ElRow :gutter="16" class="stats-row">
      <ElCol :xs="12" :sm="6" :md="6">
        <div class="stat-item">
          <div class="stat-value">{{ stats?.total || 0 }}</div>
          <div class="stat-label">总数</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="6">
        <div class="stat-item pending">
          <div class="stat-value">{{ stats?.pendingApproval || 0 }}</div>
          <div class="stat-label">待审批</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="6">
        <div class="stat-item high">
          <div class="stat-value">{{ (stats?.byRiskLevel?.high || 0) + (stats?.byRiskLevel?.critical || 0) }}</div>
          <div class="stat-label">高风险</div>
        </div>
      </ElCol>
      <ElCol :xs="12" :sm="6" :md="6">
        <div class="stat-item approved">
          <div class="stat-value">{{ stats?.byStatus?.approved || 0 }}</div>
          <div class="stat-label">已通过</div>
        </div>
      </ElCol>
    </ElRow>

    <!-- 列表 -->
    <ElCard shadow="hover">
      <template #header>
        <div class="card-header">
          <span>Learning Units 列表</span>
          <div class="header-actions">
            <ElRadioGroup v-model="viewMode" size="small">
              <ElRadioButton label="table">
                <ElIcon><List /></ElIcon>
              </ElRadioButton>
              <ElRadioButton label="card">
                <ElIcon><Grid /></ElIcon>
              </ElRadioButton>
            </ElRadioGroup>
            <ElButton :icon="Refresh" size="small" @click="fetchList">刷新</ElButton>
          </div>
        </div>
      </template>

      <!-- 表格视图 -->
      <LUTable
        v-if="viewMode === 'table'"
        :data="list"
        :loading="loading"
        @view="handleView"
        @approve="handleApprove"
        @reject="handleReject"
      />

      <!-- 卡片视图 -->
      <LUCardList
        v-else
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Refresh, List, Grid } from '@element-plus/icons-vue'
import { learningUnitsApi } from '@/api/learning-units'
import type { LearningUnit } from '@/types/nlgsm'
import LUSearch from './modules/lu-search.vue'
import LUTable from './modules/lu-table.vue'
import LUCardList from './modules/lu-card.vue'

defineOptions({ name: 'LearningUnits' })

const router = useRouter()
const loading = ref(false)
const viewMode = ref<'table' | 'card'>('table')
const list = ref<LearningUnit[]>([])
const stats = ref<any>(null)

const searchParams = reactive({
  status: '',
  riskLevel: '',
  search: ''
})

const pagination = reactive({
  page: 1,
  pageSize: 20,
  total: 0
})

const fetchList = async () => {
  loading.value = true
  try {
    const result = await learningUnitsApi.getList({
      page: pagination.page,
      pageSize: pagination.pageSize,
      status: searchParams.status || undefined,
      riskLevel: searchParams.riskLevel || undefined,
      search: searchParams.search || undefined
    })
    list.value = result.items
    pagination.total = result.total
  } catch (e) {
    console.error('Failed to fetch learning units:', e)
  } finally {
    loading.value = false
  }
}

const fetchStats = async () => {
  try {
    stats.value = await learningUnitsApi.getStats()
  } catch (e) {
    console.error('Failed to fetch stats:', e)
  }
}

const handleSearch = () => {
  pagination.page = 1
  fetchList()
}

const handleReset = () => {
  searchParams.status = ''
  searchParams.riskLevel = ''
  searchParams.search = ''
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

const handleView = (row: LearningUnit) => {
  router.push({ name: 'LearningUnitDetail', params: { id: row.id } })
}

const handleApprove = async (row: LearningUnit) => {
  try {
    await ElMessageBox.confirm(`确定要通过 Learning Unit "${row.id}" 吗？`, '确认审批', {
      confirmButtonText: '通过',
      cancelButtonText: '取消',
      type: 'warning'
    })

    await learningUnitsApi.approve(row.id, '审批通过')
    ElMessage.success('审批通过')
    fetchList()
    fetchStats()
  } catch {
    // 用户取消
  }
}

const handleReject = async (row: LearningUnit) => {
  try {
    const { value } = await ElMessageBox.prompt('请输入拒绝原因', '拒绝审批', {
      confirmButtonText: '拒绝',
      cancelButtonText: '取消',
      inputPlaceholder: '请输入拒绝原因...',
      inputValidator: (val) => !!val || '请输入拒绝原因',
      type: 'warning'
    })

    await learningUnitsApi.reject(row.id, value)
    ElMessage.success('已拒绝')
    fetchList()
    fetchStats()
  } catch {
    // 用户取消
  }
}

onMounted(() => {
  fetchList()
  fetchStats()
})
</script>

<style scoped lang="scss">
.learning-units-page {
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

    &.high .stat-value {
      color: #f44336;
    }

    &.approved .stat-value {
      color: #4caf50;
    }
  }
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;

  .header-actions {
    display: flex;
    gap: 8px;
  }
}

.pagination-wrapper {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}
</style>
