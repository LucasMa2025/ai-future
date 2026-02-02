<!-- 操作日志页面 -->
<template>
  <div class="operation-log-page art-full-height">
    <!-- 搜索栏 -->
    <ElCard shadow="never" class="search-card">
      <ElForm :model="searchForm" inline>
        <ElFormItem label="用户名">
          <ElInput v-model="searchForm.username" placeholder="请输入用户名" clearable />
        </ElFormItem>
        <ElFormItem label="请求方法">
          <ElSelect v-model="searchForm.method" placeholder="全部" clearable style="width: 120px">
            <ElOption label="GET" value="GET" />
            <ElOption label="POST" value="POST" />
            <ElOption label="PUT" value="PUT" />
            <ElOption label="DELETE" value="DELETE" />
          </ElSelect>
        </ElFormItem>
        <ElFormItem label="请求路径">
          <ElInput v-model="searchForm.path" placeholder="请输入路径" clearable />
        </ElFormItem>
        <ElFormItem label="状态">
          <ElSelect v-model="searchForm.is_success" placeholder="全部" clearable style="width: 100px">
            <ElOption label="成功" :value="true" />
            <ElOption label="失败" :value="false" />
          </ElSelect>
        </ElFormItem>
        <ElFormItem label="时间范围">
          <ElDatePicker
            v-model="dateRange"
            type="datetimerange"
            range-separator="至"
            start-placeholder="开始时间"
            end-placeholder="结束时间"
            value-format="YYYY-MM-DDTHH:mm:ss"
            style="width: 360px"
          />
        </ElFormItem>
        <ElFormItem>
          <ElButton type="primary" @click="handleSearch">
            <template #icon><i class="ri-search-line" /></template>
            查询
          </ElButton>
          <ElButton @click="handleReset">重置</ElButton>
        </ElFormItem>
      </ElForm>
    </ElCard>

    <ElCard shadow="never" class="art-table-card">
      <!-- 工具栏 -->
      <div class="toolbar">
        <div class="left">
          <ElButton @click="handleExport" :loading="exporting">
            <template #icon><i class="ri-download-line" /></template>
            导出
          </ElButton>
          <ElButton type="info" @click="showStatsDialog = true">
            <template #icon><i class="ri-bar-chart-line" /></template>
            统计
          </ElButton>
        </div>
        <div class="right">
          <ElButton :icon="RefreshRight" circle @click="loadData" />
        </div>
      </div>

      <!-- 表格 -->
      <ElTable
        v-loading="loading"
        :data="logs"
        border
        stripe
        style="width: 100%"
      >
        <ElTableColumn prop="id" label="ID" width="80" />
        <ElTableColumn prop="username" label="用户" width="120">
          <template #default="{ row }">
            <span>{{ row.username || '-' }}</span>
          </template>
        </ElTableColumn>
        <ElTableColumn prop="method" label="方法" width="80">
          <template #default="{ row }">
            <ElTag :color="getMethodColor(row.method)" size="small">
              {{ row.method }}
            </ElTag>
          </template>
        </ElTableColumn>
        <ElTableColumn prop="path" label="请求路径" min-width="200" show-overflow-tooltip />
        <ElTableColumn prop="ip_address" label="IP地址" width="140" />
        <ElTableColumn prop="status_code" label="状态码" width="90">
          <template #default="{ row }">
            <ElTag :type="row.status_code < 400 ? 'success' : 'danger'" size="small">
              {{ row.status_code || '-' }}
            </ElTag>
          </template>
        </ElTableColumn>
        <ElTableColumn prop="response_time_ms" label="响应时间" width="100">
          <template #default="{ row }">
            <span :class="getResponseTimeClass(row.response_time_ms)">
              {{ row.response_time_ms ? `${row.response_time_ms}ms` : '-' }}
            </span>
          </template>
        </ElTableColumn>
        <ElTableColumn prop="is_success" label="结果" width="80">
          <template #default="{ row }">
            <ElTag :type="row.is_success ? 'success' : 'danger'" size="small">
              {{ row.is_success ? '成功' : '失败' }}
            </ElTag>
          </template>
        </ElTableColumn>
        <ElTableColumn prop="created_at" label="时间" width="180">
          <template #default="{ row }">
            {{ formatDateTime(row.created_at) }}
          </template>
        </ElTableColumn>
        <ElTableColumn label="操作" width="80" fixed="right">
          <template #default="{ row }">
            <ElButton type="primary" link size="small" @click="showDetail(row)">
              详情
            </ElButton>
          </template>
        </ElTableColumn>
      </ElTable>

      <!-- 分页 -->
      <div class="pagination-container">
        <ElPagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :total="pagination.total"
          :page-sizes="[20, 50, 100]"
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </ElCard>

    <!-- 详情弹窗 -->
    <ElDialog v-model="detailVisible" title="日志详情" width="600px">
      <ElDescriptions v-if="currentLog" :column="1" border>
        <ElDescriptionsItem label="ID">{{ currentLog.id }}</ElDescriptionsItem>
        <ElDescriptionsItem label="用户">{{ currentLog.username || '-' }}</ElDescriptionsItem>
        <ElDescriptionsItem label="请求ID">{{ currentLog.request_id || '-' }}</ElDescriptionsItem>
        <ElDescriptionsItem label="请求方法">
          <ElTag :color="getMethodColor(currentLog.method)" size="small">
            {{ currentLog.method }}
          </ElTag>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="请求路径">{{ currentLog.path }}</ElDescriptionsItem>
        <ElDescriptionsItem label="查询参数">
          <pre class="json-content">{{ formatJson(currentLog.query_params) }}</pre>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="IP地址">{{ currentLog.ip_address || '-' }}</ElDescriptionsItem>
        <ElDescriptionsItem label="User-Agent">
          <span class="user-agent">{{ currentLog.user_agent || '-' }}</span>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="状态码">
          <ElTag :type="currentLog.status_code < 400 ? 'success' : 'danger'">
            {{ currentLog.status_code }}
          </ElTag>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="响应时间">
          {{ currentLog.response_time_ms ? `${currentLog.response_time_ms}ms` : '-' }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="功能代码">{{ currentLog.function_code || '-' }}</ElDescriptionsItem>
        <ElDescriptionsItem label="结果">
          <ElTag :type="currentLog.is_success ? 'success' : 'danger'">
            {{ currentLog.is_success ? '成功' : '失败' }}
          </ElTag>
        </ElDescriptionsItem>
        <ElDescriptionsItem v-if="currentLog.error_message" label="错误信息">
          <span class="error-message">{{ currentLog.error_message }}</span>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="时间">{{ formatDateTime(currentLog.created_at) }}</ElDescriptionsItem>
      </ElDescriptions>
    </ElDialog>

    <!-- 统计弹窗 -->
    <ElDialog v-model="showStatsDialog" title="操作日志统计" width="800px">
      <div v-loading="statsLoading" class="stats-container">
        <template v-if="statistics">
          <ElRow :gutter="20" class="stats-cards">
            <ElCol :span="6">
              <div class="stat-card">
                <div class="stat-value">{{ statistics.total_requests }}</div>
                <div class="stat-label">总请求数</div>
              </div>
            </ElCol>
            <ElCol :span="6">
              <div class="stat-card success">
                <div class="stat-value">{{ statistics.success_count }}</div>
                <div class="stat-label">成功</div>
              </div>
            </ElCol>
            <ElCol :span="6">
              <div class="stat-card danger">
                <div class="stat-value">{{ statistics.failure_count }}</div>
                <div class="stat-label">失败</div>
              </div>
            </ElCol>
            <ElCol :span="6">
              <div class="stat-card info">
                <div class="stat-value">{{ statistics.avg_response_time_ms }}ms</div>
                <div class="stat-label">平均响应</div>
              </div>
            </ElCol>
          </ElRow>

          <div class="stats-section">
            <h4>按方法统计</h4>
            <div class="method-stats">
              <div v-for="(count, method) in statistics.by_method" :key="method" class="method-item">
                <ElTag :color="getMethodColor(method as string)" size="small">{{ method }}</ElTag>
                <span>{{ count }}</span>
              </div>
            </div>
          </div>

          <div class="stats-section">
            <h4>活跃用户 Top 10</h4>
            <ElTable :data="statistics.by_user" size="small" max-height="200">
              <ElTableColumn prop="username" label="用户" />
              <ElTableColumn prop="count" label="请求数" width="100" />
            </ElTable>
          </div>
        </template>
      </div>
    </ElDialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { RefreshRight } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { operationLogsApi } from '@/api/operation-logs'
import type { OperationLog, OperationLogStatistics } from '@/types/system'
import { HttpMethodColors, HttpMethod } from '@/types/system'

defineOptions({ name: 'OperationLog' })

// 搜索表单
const searchForm = ref({
  username: '',
  method: '',
  path: '',
  is_success: undefined as boolean | undefined,
})
const dateRange = ref<[string, string] | null>(null)

// 数据
const logs = ref<OperationLog[]>([])
const pagination = ref({
  page: 1,
  pageSize: 20,
  total: 0
})

// 状态
const loading = ref(false)
const exporting = ref(false)
const detailVisible = ref(false)
const currentLog = ref<OperationLog | null>(null)
const showStatsDialog = ref(false)
const statsLoading = ref(false)
const statistics = ref<OperationLogStatistics | null>(null)

// 加载数据
const loadData = async () => {
  loading.value = true
  try {
    const params: any = {
      page: pagination.value.page,
      page_size: pagination.value.pageSize,
    }
    
    if (searchForm.value.username) params.username = searchForm.value.username
    if (searchForm.value.method) params.method = searchForm.value.method
    if (searchForm.value.path) params.path = searchForm.value.path
    if (searchForm.value.is_success !== undefined) params.is_success = searchForm.value.is_success
    if (dateRange.value) {
      params.start_date = dateRange.value[0]
      params.end_date = dateRange.value[1]
    }
    
    const res = await operationLogsApi.getList(params)
    logs.value = res.data?.items || []
    pagination.value.total = res.data?.total || 0
  } catch (error) {
    console.error('加载日志失败:', error)
  } finally {
    loading.value = false
  }
}

// 搜索
const handleSearch = () => {
  pagination.value.page = 1
  loadData()
}

// 重置
const handleReset = () => {
  searchForm.value = {
    username: '',
    method: '',
    path: '',
    is_success: undefined,
  }
  dateRange.value = null
  handleSearch()
}

// 导出
const handleExport = async () => {
  exporting.value = true
  try {
    const params: any = {}
    if (searchForm.value.username) params.username = searchForm.value.username
    if (searchForm.value.method) params.method = searchForm.value.method
    if (searchForm.value.is_success !== undefined) params.is_success = searchForm.value.is_success
    if (dateRange.value) {
      params.start_date = dateRange.value[0]
      params.end_date = dateRange.value[1]
    }
    
    const res = await operationLogsApi.exportLogs(params)
    
    // 下载文件
    const blob = new Blob([res as any], { type: 'text/csv' })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `operation_logs_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    window.URL.revokeObjectURL(url)
    
    ElMessage.success('导出成功')
  } catch (error) {
    ElMessage.error('导出失败')
    console.error('导出失败:', error)
  } finally {
    exporting.value = false
  }
}

// 显示详情
const showDetail = (row: OperationLog) => {
  currentLog.value = row
  detailVisible.value = true
}

// 加载统计
const loadStatistics = async () => {
  statsLoading.value = true
  try {
    const res = await operationLogsApi.getStatistics()
    statistics.value = res.data
  } catch (error) {
    console.error('加载统计失败:', error)
  } finally {
    statsLoading.value = false
  }
}

// 分页变化
const handleSizeChange = () => {
  pagination.value.page = 1
  loadData()
}

const handlePageChange = () => {
  loadData()
}

// 工具函数
const getMethodColor = (method: string): string => {
  return HttpMethodColors[method as HttpMethod] || '#909399'
}

const getResponseTimeClass = (time: number | null): string => {
  if (!time) return ''
  if (time < 100) return 'fast'
  if (time < 500) return 'normal'
  return 'slow'
}

const formatDateTime = (dateStr: string): string => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('zh-CN')
}

const formatJson = (obj: any): string => {
  if (!obj) return '-'
  try {
    return JSON.stringify(obj, null, 2)
  } catch {
    return String(obj)
  }
}

// 监听统计弹窗
watch(showStatsDialog, (val) => {
  if (val && !statistics.value) {
    loadStatistics()
  }
})

onMounted(() => {
  loadData()
})
</script>

<style scoped lang="scss">
.operation-log-page {
  padding: 16px;
}

.search-card {
  margin-bottom: 16px;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  
  .left {
    display: flex;
    gap: 8px;
  }
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.json-content {
  margin: 0;
  padding: 8px;
  background-color: var(--el-fill-color-light);
  border-radius: 4px;
  font-size: 12px;
  max-height: 150px;
  overflow: auto;
}

.user-agent {
  word-break: break-all;
  font-size: 12px;
  color: var(--el-text-color-secondary);
}

.error-message {
  color: var(--el-color-danger);
}

.fast { color: var(--el-color-success); }
.normal { color: var(--el-color-warning); }
.slow { color: var(--el-color-danger); }

.stats-container {
  min-height: 300px;
}

.stats-cards {
  margin-bottom: 24px;
  
  .stat-card {
    text-align: center;
    padding: 20px;
    background-color: var(--el-fill-color-light);
    border-radius: 8px;
    
    .stat-value {
      font-size: 28px;
      font-weight: 600;
      margin-bottom: 8px;
    }
    
    .stat-label {
      color: var(--el-text-color-secondary);
    }
    
    &.success .stat-value { color: var(--el-color-success); }
    &.danger .stat-value { color: var(--el-color-danger); }
    &.info .stat-value { color: var(--el-color-primary); }
  }
}

.stats-section {
  margin-bottom: 20px;
  
  h4 {
    margin-bottom: 12px;
    font-weight: 500;
  }
}

.method-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  
  .method-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background-color: var(--el-fill-color-light);
    border-radius: 4px;
  }
}
</style>
