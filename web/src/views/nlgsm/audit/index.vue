<!-- 审计日志页面 -->
<template>
  <div class="audit-page">
    <!-- 搜索筛选 -->
    <AuditSearch
      v-model:action="searchParams.action"
      v-model:actor="searchParams.actor"
      v-model:target-type="searchParams.targetType"
      v-model:date-range="searchParams.dateRange"
      :action-types="actionTypes"
      :target-types="targetTypes"
      @search="handleSearch"
      @reset="handleReset"
      @export="handleExport"
    />

    <!-- 列表 -->
    <ElCard shadow="hover">
      <template #header>
        <div class="card-header">
          <span>审计日志</span>
          <ElButton :icon="Refresh" size="small" @click="fetchList">刷新</ElButton>
        </div>
      </template>

      <AuditTable
        :data="list"
        :loading="loading"
        @view="handleView"
        @verify="handleVerify"
      />

      <div class="pagination-wrapper">
        <ElPagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.pageSize"
          :total="pagination.total"
          :page-sizes="[20, 50, 100]"
          layout="total, sizes, prev, pager, next"
          @size-change="handleSizeChange"
          @current-change="handlePageChange"
        />
      </div>
    </ElCard>

    <!-- 详情对话框 -->
    <AuditDetailDialog v-model="detailVisible" :log="selectedLog" />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { auditApi } from '@/api/audit'
import type { AuditLog } from '@/types/nlgsm'
import AuditSearch from './modules/audit-search.vue'
import AuditTable from './modules/audit-table.vue'
import AuditDetailDialog from './modules/audit-detail-dialog.vue'

defineOptions({ name: 'AuditLogs' })

const loading = ref(false)
const list = ref<AuditLog[]>([])
const actionTypes = ref<string[]>([])
const targetTypes = ref<string[]>([])
const detailVisible = ref(false)
const selectedLog = ref<AuditLog | null>(null)

const searchParams = reactive({
  action: '',
  actor: '',
  targetType: '',
  dateRange: [] as string[]
})

const pagination = reactive({
  page: 1,
  pageSize: 20,
  total: 0
})

const fetchList = async () => {
  loading.value = true
  try {
    const result = await auditApi.getList({
      page: pagination.page,
      pageSize: pagination.pageSize,
      action: searchParams.action || undefined,
      actor: searchParams.actor || undefined,
      targetType: searchParams.targetType || undefined,
      startDate: searchParams.dateRange[0] || undefined,
      endDate: searchParams.dateRange[1] || undefined
    })
    list.value = result.items
    pagination.total = result.total
  } catch (e) {
    console.error('Failed to fetch audit logs:', e)
  } finally {
    loading.value = false
  }
}

const fetchOptions = async () => {
  try {
    const [actions, targets] = await Promise.all([
      auditApi.getActionTypes(),
      auditApi.getTargetTypes()
    ])
    actionTypes.value = actions
    targetTypes.value = targets
  } catch (e) {
    console.error('Failed to fetch options:', e)
  }
}

const handleSearch = () => {
  pagination.page = 1
  fetchList()
}

const handleReset = () => {
  searchParams.action = ''
  searchParams.actor = ''
  searchParams.targetType = ''
  searchParams.dateRange = []
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

const handleView = (row: AuditLog) => {
  selectedLog.value = row
  detailVisible.value = true
}

const handleVerify = async (row: AuditLog) => {
  try {
    const result = await auditApi.verifyIntegrity(row.id)
    if (result.valid) {
      ElMessage.success('完整性验证通过')
    } else {
      ElMessage.error('完整性验证失败：哈希不匹配')
    }
  } catch (e) {
    ElMessage.error('验证失败')
  }
}

const handleExport = async () => {
  try {
    const blob = await auditApi.exportLogs({
      action: searchParams.action || undefined,
      actor: searchParams.actor || undefined,
      targetType: searchParams.targetType || undefined,
      startDate: searchParams.dateRange[0] || undefined,
      endDate: searchParams.dateRange[1] || undefined
    })

    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `audit_logs_${new Date().toISOString().slice(0, 10)}.csv`
    a.click()
    window.URL.revokeObjectURL(url)

    ElMessage.success('导出成功')
  } catch (e) {
    ElMessage.error('导出失败')
  }
}

onMounted(() => {
  fetchList()
  fetchOptions()
})
</script>

<style scoped lang="scss">
.audit-page {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.pagination-wrapper {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}
</style>
