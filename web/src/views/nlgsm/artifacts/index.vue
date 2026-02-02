<!-- 工件管理页面 -->
<template>
  <div class="artifacts-page">
    <!-- 搜索 -->
    <ArtifactSearch
      v-model:search="searchParams.search"
      v-model:is-approved="searchParams.isApproved"
      @search="handleSearch"
      @reset="handleReset"
    />

    <!-- 列表 -->
    <ElCard shadow="hover">
      <template #header>
        <div class="card-header">
          <span>工件列表</span>
          <ElButton :icon="Refresh" size="small" @click="fetchList">刷新</ElButton>
        </div>
      </template>

      <ArtifactTable
        :data="list"
        :loading="loading"
        @view="handleView"
        @rollback="handleRollback"
      />

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

    <!-- 回滚对话框 -->
    <RollbackDialog
      v-model="rollbackVisible"
      :artifact="selectedArtifact"
      @confirm="handleRollbackConfirm"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { artifactsApi } from '@/api/artifacts'
import type { Artifact } from '@/types/nlgsm'
import ArtifactSearch from './modules/artifact-search.vue'
import ArtifactTable from './modules/artifact-table.vue'
import RollbackDialog from './modules/rollback-dialog.vue'

defineOptions({ name: 'Artifacts' })

const router = useRouter()
const loading = ref(false)
const list = ref<Artifact[]>([])
const rollbackVisible = ref(false)
const selectedArtifact = ref<Artifact | null>(null)

const searchParams = reactive({
  search: '',
  isApproved: ''
})

const pagination = reactive({
  page: 1,
  pageSize: 20,
  total: 0
})

const fetchList = async () => {
  loading.value = true
  try {
    const result = await artifactsApi.getList({
      page: pagination.page,
      pageSize: pagination.pageSize,
      search: searchParams.search || undefined,
      isApproved: searchParams.isApproved === '' ? undefined : searchParams.isApproved === 'true'
    })
    list.value = result.items
    pagination.total = result.total
  } catch (e) {
    console.error('Failed to fetch artifacts:', e)
  } finally {
    loading.value = false
  }
}

const handleSearch = () => {
  pagination.page = 1
  fetchList()
}

const handleReset = () => {
  searchParams.search = ''
  searchParams.isApproved = ''
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

const handleView = (row: Artifact) => {
  router.push({ name: 'ArtifactDetail', params: { id: row.id } })
}

const handleRollback = (row: Artifact) => {
  selectedArtifact.value = row
  rollbackVisible.value = true
}

const handleRollbackConfirm = async (targetVersion: number, reason: string) => {
  if (!selectedArtifact.value) return

  try {
    await artifactsApi.rollback(selectedArtifact.value.id, { targetVersion, reason })
    ElMessage.success('回滚成功')
    rollbackVisible.value = false
    fetchList()
  } catch (e) {
    console.error('Rollback failed:', e)
  }
}

onMounted(() => {
  fetchList()
})
</script>

<style scoped lang="scss">
.artifacts-page {
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
