<!-- 数据备份页面 -->
<template>
  <div class="backup-page art-full-height">
    <ElRow :gutter="16">
      <!-- 统计信息 -->
      <ElCol :span="24">
        <ElCard shadow="never" class="stats-card">
          <div class="stats-row">
            <div class="stat-item">
              <div class="stat-icon">
                <i class="ri-database-2-line" />
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ backupStats?.total_backups || 0 }}</div>
                <div class="stat-label">总备份数</div>
              </div>
            </div>
            <div class="stat-item success">
              <div class="stat-icon">
                <i class="ri-checkbox-circle-line" />
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ backupStats?.completed_backups || 0 }}</div>
                <div class="stat-label">已完成</div>
              </div>
            </div>
            <div class="stat-item danger">
              <div class="stat-icon">
                <i class="ri-close-circle-line" />
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ backupStats?.failed_backups || 0 }}</div>
                <div class="stat-label">失败</div>
              </div>
            </div>
            <div class="stat-item info">
              <div class="stat-icon">
                <i class="ri-hard-drive-2-line" />
              </div>
              <div class="stat-info">
                <div class="stat-value">{{ backupStats?.total_storage_display || '0 B' }}</div>
                <div class="stat-label">存储空间</div>
              </div>
            </div>
            <div class="stat-item">
              <div class="stat-info">
                <div class="stat-value text-sm">
                  {{ backupStats?.latest_backup ? formatDateTime(backupStats.latest_backup.completed_at) : '无' }}
                </div>
                <div class="stat-label">最近备份</div>
              </div>
            </div>
          </div>
        </ElCard>
      </ElCol>
    </ElRow>

    <ElRow :gutter="16" class="mt-4">
      <!-- 备份列表 -->
      <ElCol :span="16">
        <ElCard shadow="never" class="art-table-card">
          <template #header>
            <div class="card-header">
              <span>备份列表</span>
              <div class="actions">
                <ElButton type="primary" @click="showCreateDialog = true">
                  <template #icon><i class="ri-add-line" /></template>
                  创建备份
                </ElButton>
                <ElButton :icon="RefreshRight" circle @click="loadBackups" />
              </div>
            </div>
          </template>

          <ElTable
            v-loading="loading"
            :data="backups"
            border
            stripe
          >
            <ElTableColumn prop="backup_name" label="备份名称" min-width="160" />
            <ElTableColumn prop="backup_type" label="类型" width="100">
              <template #default="{ row }">
                <ElTag :type="row.backup_type === 'full' ? 'primary' : 'info'" size="small">
                  {{ row.backup_type === 'full' ? '完整' : '部分' }}
                </ElTag>
              </template>
            </ElTableColumn>
            <ElTableColumn prop="status" label="状态" width="100">
              <template #default="{ row }">
                <ElTag :type="getStatusType(row.status)" size="small">
                  {{ getStatusLabel(row.status) }}
                </ElTag>
              </template>
            </ElTableColumn>
            <ElTableColumn prop="file_size_display" label="大小" width="100" />
            <ElTableColumn prop="progress" label="进度" width="100">
              <template #default="{ row }">
                <ElProgress 
                  v-if="row.status === 'running'"
                  :percentage="row.progress" 
                  :stroke-width="6"
                  :show-text="false"
                />
                <span v-else>{{ row.status === 'completed' ? '100%' : '-' }}</span>
              </template>
            </ElTableColumn>
            <ElTableColumn prop="created_at" label="创建时间" width="160">
              <template #default="{ row }">
                {{ formatDateTime(row.created_at) }}
              </template>
            </ElTableColumn>
            <ElTableColumn label="操作" width="140" fixed="right">
              <template #default="{ row }">
                <ElButton 
                  type="primary" 
                  link 
                  size="small"
                  :disabled="row.status !== 'completed'"
                  @click="handleRestore(row)"
                >
                  恢复
                </ElButton>
                <ElButton 
                  type="primary" 
                  link 
                  size="small"
                  @click="showBackupDetail(row)"
                >
                  详情
                </ElButton>
                <ElButton 
                  type="danger" 
                  link 
                  size="small"
                  @click="handleDelete(row)"
                >
                  删除
                </ElButton>
              </template>
            </ElTableColumn>
          </ElTable>

          <div class="pagination-container">
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
      </ElCol>

      <!-- 可备份表信息 -->
      <ElCol :span="8">
        <ElCard shadow="never">
          <template #header>
            <div class="card-header">
              <span>数据表信息</span>
              <ElButton :icon="RefreshRight" circle size="small" @click="loadTableInfo" />
            </div>
          </template>

          <div v-loading="tableInfoLoading" class="table-info-list">
            <div 
              v-for="table in tableInfo" 
              :key="table.table_name"
              class="table-info-item"
            >
              <div class="table-name">
                <i class="ri-table-line" />
                <span>{{ table.table_name }}</span>
              </div>
              <div class="record-count">
                {{ table.record_count.toLocaleString() }} 条
              </div>
            </div>
          </div>
        </ElCard>
      </ElCol>
    </ElRow>

    <!-- 创建备份弹窗 -->
    <ElDialog v-model="showCreateDialog" title="创建备份" width="500px">
      <ElForm :model="createForm" label-width="100px">
        <ElFormItem label="备份名称" required>
          <ElInput v-model="createForm.backup_name" placeholder="请输入备份名称" />
        </ElFormItem>
        <ElFormItem label="备份类型">
          <ElRadioGroup v-model="createForm.backup_type">
            <ElRadio value="full">完整备份</ElRadio>
            <ElRadio value="tables">指定表</ElRadio>
          </ElRadioGroup>
        </ElFormItem>
        <ElFormItem v-if="createForm.backup_type === 'tables'" label="选择表">
          <ElSelect 
            v-model="createForm.tables" 
            multiple 
            placeholder="请选择要备份的表"
            style="width: 100%"
          >
            <ElOption
              v-for="table in tableInfo"
              :key="table.table_name"
              :label="table.table_name"
              :value="table.table_name"
            />
          </ElSelect>
        </ElFormItem>
        <ElFormItem label="压缩">
          <ElSwitch v-model="createForm.compress" />
        </ElFormItem>
        <ElFormItem label="备份说明">
          <ElInput 
            v-model="createForm.description" 
            type="textarea"
            :rows="3"
            placeholder="可选，备份说明"
          />
        </ElFormItem>
      </ElForm>
      <template #footer>
        <ElButton @click="showCreateDialog = false">取消</ElButton>
        <ElButton type="primary" :loading="creating" @click="handleCreate">
          创建
        </ElButton>
      </template>
    </ElDialog>

    <!-- 备份详情弹窗 -->
    <ElDialog v-model="detailVisible" title="备份详情" width="600px">
      <ElDescriptions v-if="currentBackup" :column="1" border>
        <ElDescriptionsItem label="备份名称">{{ currentBackup.backup_name }}</ElDescriptionsItem>
        <ElDescriptionsItem label="备份类型">
          <ElTag :type="currentBackup.backup_type === 'full' ? 'primary' : 'info'">
            {{ currentBackup.backup_type === 'full' ? '完整备份' : '指定表备份' }}
          </ElTag>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="状态">
          <ElTag :type="getStatusType(currentBackup.status)">
            {{ getStatusLabel(currentBackup.status) }}
          </ElTag>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="文件大小">{{ currentBackup.file_size_display }}</ElDescriptionsItem>
        <ElDescriptionsItem label="是否压缩">
          {{ currentBackup.compressed ? '是' : '否' }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="开始时间">
          {{ formatDateTime(currentBackup.started_at) }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="完成时间">
          {{ formatDateTime(currentBackup.completed_at) }}
        </ElDescriptionsItem>
        <ElDescriptionsItem label="耗时">
          {{ currentBackup.duration_seconds ? `${currentBackup.duration_seconds}秒` : '-' }}
        </ElDescriptionsItem>
        <ElDescriptionsItem v-if="currentBackup.tables?.length" label="备份表">
          <div class="backup-tables">
            <ElTag v-for="t in currentBackup.tables" :key="t" size="small" class="mr-1 mb-1">
              {{ t }}
            </ElTag>
          </div>
        </ElDescriptionsItem>
        <ElDescriptionsItem v-if="currentBackup.record_counts" label="记录数">
          <div class="record-counts">
            <div v-for="(count, table) in currentBackup.record_counts" :key="table" class="count-item">
              <span class="table">{{ table }}</span>
              <span class="count">{{ count }}</span>
            </div>
          </div>
        </ElDescriptionsItem>
        <ElDescriptionsItem v-if="currentBackup.error_message" label="错误信息">
          <span class="error-text">{{ currentBackup.error_message }}</span>
        </ElDescriptionsItem>
        <ElDescriptionsItem label="备份说明">
          {{ currentBackup.description || '-' }}
        </ElDescriptionsItem>
      </ElDescriptions>
    </ElDialog>

    <!-- 恢复确认弹窗 -->
    <ElDialog v-model="restoreVisible" title="恢复备份" width="500px">
      <ElAlert type="warning" :closable="false" class="mb-4">
        <template #title>
          <strong>警告：</strong>恢复操作将覆盖当前数据库中的数据，此操作不可撤销！
        </template>
      </ElAlert>
      
      <ElForm :model="restoreForm" label-width="100px">
        <ElFormItem label="恢复范围">
          <ElRadioGroup v-model="restoreMode">
            <ElRadio value="all">全部恢复</ElRadio>
            <ElRadio value="select">选择表</ElRadio>
          </ElRadioGroup>
        </ElFormItem>
        <ElFormItem v-if="restoreMode === 'select'" label="选择表">
          <ElSelect 
            v-model="restoreForm.tables" 
            multiple 
            placeholder="请选择要恢复的表"
            style="width: 100%"
          >
            <ElOption
              v-for="t in currentBackup?.tables"
              :key="t"
              :label="t"
              :value="t"
            />
          </ElSelect>
        </ElFormItem>
      </ElForm>
      
      <template #footer>
        <ElButton @click="restoreVisible = false">取消</ElButton>
        <ElButton type="danger" :loading="restoring" @click="confirmRestore">
          确认恢复
        </ElButton>
      </template>
    </ElDialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { RefreshRight } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { backupsApi } from '@/api/backups'
import type { DataBackup, BackupStatistics, TableInfo } from '@/types/system'
import { BackupStatus, BackupStatusLabels } from '@/types/system'

defineOptions({ name: 'Backup' })

// 数据
const backups = ref<DataBackup[]>([])
const backupStats = ref<BackupStatistics | null>(null)
const tableInfo = ref<TableInfo[]>([])
const pagination = ref({
  page: 1,
  pageSize: 10,
  total: 0
})

// 状态
const loading = ref(false)
const tableInfoLoading = ref(false)
const creating = ref(false)
const restoring = ref(false)

// 弹窗
const showCreateDialog = ref(false)
const detailVisible = ref(false)
const restoreVisible = ref(false)
const currentBackup = ref<DataBackup | null>(null)
const restoreMode = ref<'all' | 'select'>('all')

// 表单
const createForm = ref({
  backup_name: '',
  backup_type: 'full' as 'full' | 'tables',
  tables: [] as string[],
  compress: true,
  description: ''
})

const restoreForm = ref({
  tables: [] as string[]
})

// 轮询
let pollInterval: ReturnType<typeof setInterval> | null = null

// 加载备份列表
const loadBackups = async () => {
  loading.value = true
  try {
    const res = await backupsApi.getList({
      page: pagination.value.page,
      page_size: pagination.value.pageSize
    })
    backups.value = res.data?.items || []
    pagination.value.total = res.data?.total || 0
  } catch (error) {
    console.error('加载备份列表失败:', error)
  } finally {
    loading.value = false
  }
}

// 加载统计
const loadStats = async () => {
  try {
    const res = await backupsApi.getStats()
    backupStats.value = res.data
  } catch (error) {
    console.error('加载统计失败:', error)
  }
}

// 加载表信息
const loadTableInfo = async () => {
  tableInfoLoading.value = true
  try {
    const res = await backupsApi.getTableInfo()
    tableInfo.value = res.data || []
  } catch (error) {
    console.error('加载表信息失败:', error)
  } finally {
    tableInfoLoading.value = false
  }
}

// 创建备份
const handleCreate = async () => {
  if (!createForm.value.backup_name) {
    ElMessage.warning('请输入备份名称')
    return
  }
  
  if (createForm.value.backup_type === 'tables' && createForm.value.tables.length === 0) {
    ElMessage.warning('请选择要备份的表')
    return
  }
  
  creating.value = true
  try {
    await backupsApi.create({
      backup_name: createForm.value.backup_name,
      backup_type: createForm.value.backup_type,
      tables: createForm.value.backup_type === 'tables' ? createForm.value.tables : undefined,
      compress: createForm.value.compress,
      description: createForm.value.description || undefined
    })
    
    ElMessage.success('备份任务已创建')
    showCreateDialog.value = false
    resetCreateForm()
    loadBackups()
    loadStats()
    
    // 启动轮询
    startPolling()
  } catch (error) {
    ElMessage.error('创建备份失败')
    console.error('创建备份失败:', error)
  } finally {
    creating.value = false
  }
}

// 显示详情
const showBackupDetail = (backup: DataBackup) => {
  currentBackup.value = backup
  detailVisible.value = true
}

// 恢复备份
const handleRestore = (backup: DataBackup) => {
  currentBackup.value = backup
  restoreMode.value = 'all'
  restoreForm.value.tables = []
  restoreVisible.value = true
}

// 确认恢复
const confirmRestore = async () => {
  if (!currentBackup.value) return
  
  restoring.value = true
  try {
    const res = await backupsApi.restore(currentBackup.value.id, {
      tables: restoreMode.value === 'select' ? restoreForm.value.tables : undefined
    })
    
    // 显示恢复结果
    const results = res.data?.results || {}
    const success = Object.values(results).filter((r: any) => r.success).length
    const failed = Object.values(results).filter((r: any) => !r.success).length
    
    if (failed === 0) {
      ElMessage.success(`恢复成功，共恢复 ${success} 个表`)
    } else {
      ElMessage.warning(`恢复完成，成功 ${success} 个，失败 ${failed} 个`)
    }
    
    restoreVisible.value = false
  } catch (error) {
    ElMessage.error('恢复失败')
    console.error('恢复失败:', error)
  } finally {
    restoring.value = false
  }
}

// 删除备份
const handleDelete = (backup: DataBackup) => {
  ElMessageBox.confirm(
    `确定要删除备份"${backup.backup_name}"吗？此操作不可恢复！`,
    '删除确认',
    {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    }
  ).then(async () => {
    try {
      await backupsApi.remove(backup.id)
      ElMessage.success('删除成功')
      loadBackups()
      loadStats()
    } catch (error) {
      ElMessage.error('删除失败')
      console.error('删除失败:', error)
    }
  })
}

// 分页
const handleSizeChange = () => {
  pagination.value.page = 1
  loadBackups()
}

const handlePageChange = () => {
  loadBackups()
}

// 重置表单
const resetCreateForm = () => {
  createForm.value = {
    backup_name: '',
    backup_type: 'full',
    tables: [],
    compress: true,
    description: ''
  }
}

// 轮询检查运行中的备份
const startPolling = () => {
  if (pollInterval) return
  
  pollInterval = setInterval(() => {
    const hasRunning = backups.value.some(b => b.status === 'running' || b.status === 'pending')
    if (hasRunning) {
      loadBackups()
      loadStats()
    } else {
      stopPolling()
    }
  }, 3000)
}

const stopPolling = () => {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

// 工具函数
const getStatusType = (status: string): 'info' | 'warning' | 'success' | 'danger' => {
  const map: Record<string, 'info' | 'warning' | 'success' | 'danger'> = {
    pending: 'info',
    running: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return map[status] || 'info'
}

const getStatusLabel = (status: string): string => {
  return BackupStatusLabels[status as BackupStatus] || status
}

const formatDateTime = (dateStr: string | undefined): string => {
  if (!dateStr) return '-'
  return new Date(dateStr).toLocaleString('zh-CN')
}

onMounted(() => {
  loadBackups()
  loadStats()
  loadTableInfo()
  
  // 检查是否有运行中的任务
  const hasRunning = backups.value.some(b => b.status === 'running' || b.status === 'pending')
  if (hasRunning) {
    startPolling()
  }
})

onUnmounted(() => {
  stopPolling()
})
</script>

<style scoped lang="scss">
.backup-page {
  padding: 16px;
}

.stats-card {
  .stats-row {
    display: flex;
    justify-content: space-around;
    
    .stat-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 16px 24px;
      
      .stat-icon {
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: var(--el-fill-color-light);
        border-radius: 12px;
        
        i {
          font-size: 24px;
          color: var(--el-text-color-secondary);
        }
      }
      
      .stat-info {
        .stat-value {
          font-size: 24px;
          font-weight: 600;
          margin-bottom: 4px;
        }
        
        .stat-label {
          font-size: 13px;
          color: var(--el-text-color-secondary);
        }
      }
      
      &.success {
        .stat-icon {
          background-color: var(--el-color-success-light-9);
          i { color: var(--el-color-success); }
        }
        .stat-value { color: var(--el-color-success); }
      }
      
      &.danger {
        .stat-icon {
          background-color: var(--el-color-danger-light-9);
          i { color: var(--el-color-danger); }
        }
        .stat-value { color: var(--el-color-danger); }
      }
      
      &.info {
        .stat-icon {
          background-color: var(--el-color-primary-light-9);
          i { color: var(--el-color-primary); }
        }
        .stat-value { color: var(--el-color-primary); }
      }
    }
  }
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  
  .actions {
    display: flex;
    gap: 8px;
  }
}

.pagination-container {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}

.table-info-list {
  max-height: 500px;
  overflow-y: auto;
  
  .table-info-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    border-bottom: 1px solid var(--el-border-color-lighter);
    
    &:last-child {
      border-bottom: none;
    }
    
    .table-name {
      display: flex;
      align-items: center;
      gap: 8px;
      
      i {
        color: var(--el-text-color-secondary);
      }
    }
    
    .record-count {
      color: var(--el-text-color-secondary);
      font-size: 13px;
    }
  }
}

.backup-tables {
  max-height: 100px;
  overflow-y: auto;
}

.record-counts {
  max-height: 150px;
  overflow-y: auto;
  
  .count-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px dashed var(--el-border-color-lighter);
    
    &:last-child {
      border-bottom: none;
    }
    
    .table {
      color: var(--el-text-color-regular);
    }
    
    .count {
      color: var(--el-color-primary);
      font-weight: 500;
    }
  }
}

.error-text {
  color: var(--el-color-danger);
}

.text-sm {
  font-size: 16px !important;
}
</style>
