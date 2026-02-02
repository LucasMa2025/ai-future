<!-- 权限分配页面 - 统一管理角色和用户的权限 -->
<template>
  <div class="permission-page art-full-height">
    <ElCard shadow="never">
      <!-- 切换角色/用户模式 -->
      <div class="permission-header">
        <ElRadioGroup v-model="assignMode" @change="handleModeChange">
          <ElRadioButton value="role">按角色分配</ElRadioButton>
          <ElRadioButton value="user">按用户分配</ElRadioButton>
        </ElRadioGroup>
        
        <div class="target-selector">
          <!-- 角色选择 -->
          <ElSelect
            v-if="assignMode === 'role'"
            v-model="selectedRoleId"
            placeholder="请选择角色"
            style="width: 240px"
            @change="handleTargetChange"
          >
            <ElOption
              v-for="role in roles"
              :key="role.id"
              :label="role.display_name"
              :value="role.id"
            >
              <div class="flex-y-center justify-between">
                <span>{{ role.display_name }}</span>
                <ElTag v-if="role.is_system" type="info" size="small">系统</ElTag>
              </div>
            </ElOption>
          </ElSelect>
          
          <!-- 用户选择 -->
          <ElSelect
            v-else
            v-model="selectedUserId"
            placeholder="请选择用户"
            filterable
            style="width: 240px"
            @change="handleTargetChange"
          >
            <ElOption
              v-for="user in users"
              :key="user.id"
              :label="user.username"
              :value="user.id"
            >
              <div class="flex-y-center gap-2">
                <ElAvatar :size="24" :src="user.avatar">
                  {{ user.username?.charAt(0).toUpperCase() }}
                </ElAvatar>
                <span>{{ user.username }}</span>
                <span class="text-gray-400">({{ user.full_name || user.email }})</span>
              </div>
            </ElOption>
          </ElSelect>
        </div>
        
        <ElButton 
          type="primary" 
          :loading="saving"
          :disabled="!hasTarget"
          @click="handleSave"
        >
          <template #icon><i class="ri-save-line" /></template>
          保存权限
        </ElButton>
      </div>
      
      <!-- 权限树 -->
      <div class="permission-content" v-loading="loading">
        <ElEmpty v-if="!hasTarget" description="请先选择角色或用户" />
        
        <div v-else class="permission-tree-container">
          <!-- 功能模块列表 -->
          <div class="module-list">
            <div
              v-for="module in permissionTree"
              :key="module.id"
              class="module-item"
              :class="{ active: activeModuleId === module.id }"
              @click="activeModuleId = module.id"
            >
              <i :class="module.icon || 'ri-folder-line'" />
              <span>{{ module.name }}</span>
              <ElBadge 
                :value="getModuleCheckedCount(module)" 
                :max="99"
                :hidden="getModuleCheckedCount(module) === 0"
              />
            </div>
          </div>
          
          <!-- 权限详情 -->
          <div class="permission-detail">
            <template v-if="activeModule">
              <div class="module-header">
                <h3>{{ activeModule.name }}</h3>
                <ElCheckbox
                  :model-value="isModuleAllChecked(activeModule)"
                  :indeterminate="isModulePartialChecked(activeModule)"
                  @change="toggleModuleAll(activeModule, $event)"
                >
                  全选
                </ElCheckbox>
              </div>
              
              <div class="feature-groups">
                <div 
                  v-for="group in activeModule.children" 
                  :key="group.id"
                  class="feature-group"
                >
                  <div class="group-header">
                    <ElCheckbox
                      :model-value="isGroupAllChecked(group)"
                      :indeterminate="isGroupPartialChecked(group)"
                      @change="toggleGroupAll(group, $event)"
                    >
                      {{ group.name }}
                    </ElCheckbox>
                  </div>
                  
                  <div class="permission-items">
                    <template v-if="group.permissions && group.permissions.length > 0">
                      <ElCheckbox
                        v-for="perm in group.permissions"
                        :key="perm.id"
                        v-model="checkedPermissions[perm.id]"
                        class="permission-checkbox"
                      >
                        {{ perm.name }}
                        <ElTag size="small" class="ml-1">{{ perm.action }}</ElTag>
                      </ElCheckbox>
                    </template>
                    
                    <!-- 子功能 -->
                    <div v-if="group.children && group.children.length > 0" class="sub-features">
                      <div 
                        v-for="feature in group.children" 
                        :key="feature.id"
                        class="sub-feature"
                      >
                        <div class="feature-name">
                          <ElTag v-if="feature.method" :color="getMethodColor(feature.method)" size="small">
                            {{ feature.method }}
                          </ElTag>
                          <span>{{ feature.name }}</span>
                        </div>
                        <div class="feature-permissions">
                          <ElCheckbox
                            v-for="perm in feature.permissions"
                            :key="perm.id"
                            v-model="checkedPermissions[perm.id]"
                          >
                            {{ perm.name }}
                          </ElCheckbox>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </ElCard>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import { permissionsApi } from '@/api/permissions'
import { rolesApi } from '@/api/roles'
import { usersApi } from '@/api/users'
import type { SystemFunction, Role, User } from '@/types/system'
import { HttpMethodColors, HttpMethod } from '@/types/system'

defineOptions({ name: 'Permission' })

// 分配模式
const assignMode = ref<'role' | 'user'>('role')

// 目标选择
const selectedRoleId = ref<number | null>(null)
const selectedUserId = ref<string | null>(null)

// 数据
const roles = ref<Role[]>([])
const users = ref<User[]>([])
const permissionTree = ref<SystemFunction[]>([])
const checkedPermissions = ref<Record<number, boolean>>({})

// 状态
const loading = ref(false)
const saving = ref(false)
const activeModuleId = ref<number | null>(null)

// 计算属性
const hasTarget = computed(() => {
  return assignMode.value === 'role' 
    ? selectedRoleId.value !== null 
    : selectedUserId.value !== null
})

const activeModule = computed(() => {
  return permissionTree.value.find(m => m.id === activeModuleId.value)
})

// 加载角色列表
const loadRoles = async () => {
  try {
    const res = await rolesApi.getList()
    roles.value = res.data || []
  } catch (error) {
    console.error('加载角色失败:', error)
  }
}

// 加载用户列表
const loadUsers = async () => {
  try {
    const res = await usersApi.getList({ page_size: 100 })
    users.value = res.data?.items || []
  } catch (error) {
    console.error('加载用户失败:', error)
  }
}

// 加载权限树
const loadPermissionTree = async () => {
  loading.value = true
  try {
    const params: { role_id?: number; user_id?: string } = {}
    if (assignMode.value === 'role' && selectedRoleId.value) {
      params.role_id = selectedRoleId.value
    } else if (assignMode.value === 'user' && selectedUserId.value) {
      params.user_id = selectedUserId.value
    }
    
    const res = await permissionsApi.getTree(params.role_id, params.user_id)
    permissionTree.value = res.data || []
    
    // 初始化选中状态
    initCheckedPermissions(permissionTree.value)
    
    // 默认选中第一个模块
    if (permissionTree.value.length > 0 && !activeModuleId.value) {
      activeModuleId.value = permissionTree.value[0].id
    }
  } catch (error) {
    console.error('加载权限树失败:', error)
  } finally {
    loading.value = false
  }
}

// 初始化选中状态
const initCheckedPermissions = (tree: SystemFunction[]) => {
  checkedPermissions.value = {}
  
  const traverse = (nodes: SystemFunction[]) => {
    for (const node of nodes) {
      if (node.permissions) {
        for (const perm of node.permissions) {
          checkedPermissions.value[perm.id] = perm.checked
        }
      }
      if (node.children) {
        traverse(node.children)
      }
    }
  }
  
  traverse(tree)
}

// 获取模块选中数量
const getModuleCheckedCount = (module: SystemFunction): number => {
  let count = 0
  
  const traverse = (nodes: SystemFunction[]) => {
    for (const node of nodes) {
      if (node.permissions) {
        count += node.permissions.filter(p => checkedPermissions.value[p.id]).length
      }
      if (node.children) {
        traverse(node.children)
      }
    }
  }
  
  traverse([module])
  return count
}

// 模块是否全选
const isModuleAllChecked = (module: SystemFunction): boolean => {
  const permissions = collectPermissions([module])
  return permissions.length > 0 && permissions.every(p => checkedPermissions.value[p.id])
}

// 模块是否部分选中
const isModulePartialChecked = (module: SystemFunction): boolean => {
  const permissions = collectPermissions([module])
  const checkedCount = permissions.filter(p => checkedPermissions.value[p.id]).length
  return checkedCount > 0 && checkedCount < permissions.length
}

// 切换模块全选
const toggleModuleAll = (module: SystemFunction, checked: boolean | string | number) => {
  const permissions = collectPermissions([module])
  permissions.forEach(p => {
    checkedPermissions.value[p.id] = !!checked
  })
}

// 功能组是否全选
const isGroupAllChecked = (group: SystemFunction): boolean => {
  const permissions = collectPermissions([group])
  return permissions.length > 0 && permissions.every(p => checkedPermissions.value[p.id])
}

// 功能组是否部分选中
const isGroupPartialChecked = (group: SystemFunction): boolean => {
  const permissions = collectPermissions([group])
  const checkedCount = permissions.filter(p => checkedPermissions.value[p.id]).length
  return checkedCount > 0 && checkedCount < permissions.length
}

// 切换功能组全选
const toggleGroupAll = (group: SystemFunction, checked: boolean | string | number) => {
  const permissions = collectPermissions([group])
  permissions.forEach(p => {
    checkedPermissions.value[p.id] = !!checked
  })
}

// 收集所有权限
const collectPermissions = (nodes: SystemFunction[]): { id: number }[] => {
  const result: { id: number }[] = []
  
  const traverse = (nodes: SystemFunction[]) => {
    for (const node of nodes) {
      if (node.permissions) {
        result.push(...node.permissions)
      }
      if (node.children) {
        traverse(node.children)
      }
    }
  }
  
  traverse(nodes)
  return result
}

// 获取HTTP方法颜色
const getMethodColor = (method: string): string => {
  return HttpMethodColors[method as HttpMethod] || '#909399'
}

// 处理模式切换
const handleModeChange = () => {
  selectedRoleId.value = null
  selectedUserId.value = null
  permissionTree.value = []
  checkedPermissions.value = {}
  activeModuleId.value = null
}

// 处理目标切换
const handleTargetChange = () => {
  if (hasTarget.value) {
    loadPermissionTree()
  }
}

// 保存权限
const handleSave = async () => {
  saving.value = true
  try {
    const permissionIds = Object.entries(checkedPermissions.value)
      .filter(([_, checked]) => checked)
      .map(([id]) => parseInt(id))
    
    if (assignMode.value === 'role' && selectedRoleId.value) {
      await permissionsApi.setRolePermissions(selectedRoleId.value, permissionIds)
      ElMessage.success('角色权限保存成功')
    } else if (assignMode.value === 'user' && selectedUserId.value) {
      const permissions = permissionIds.map(id => ({ permission_id: id, grant_type: 'allow' }))
      await permissionsApi.setUserPermissions(selectedUserId.value, permissions)
      ElMessage.success('用户权限保存成功')
    }
  } catch (error) {
    ElMessage.error('保存权限失败')
    console.error('保存权限失败:', error)
  } finally {
    saving.value = false
  }
}

// 初始化
onMounted(async () => {
  await Promise.all([loadRoles(), loadUsers()])
})
</script>

<style scoped lang="scss">
.permission-page {
  padding: 16px;
}

.permission-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--el-border-color-light);
  
  .target-selector {
    flex: 1;
  }
}

.permission-content {
  min-height: 500px;
}

.permission-tree-container {
  display: flex;
  gap: 20px;
  height: calc(100vh - 300px);
  min-height: 500px;
}

.module-list {
  width: 200px;
  flex-shrink: 0;
  border-right: 1px solid var(--el-border-color-light);
  padding-right: 16px;
  overflow-y: auto;
  
  .module-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    margin-bottom: 4px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
    
    i {
      font-size: 18px;
      color: var(--el-text-color-secondary);
    }
    
    span {
      flex: 1;
    }
    
    &:hover {
      background-color: var(--el-fill-color-light);
    }
    
    &.active {
      background-color: var(--el-color-primary-light-9);
      color: var(--el-color-primary);
      
      i {
        color: var(--el-color-primary);
      }
    }
  }
}

.permission-detail {
  flex: 1;
  overflow-y: auto;
  padding-left: 16px;
  
  .module-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    
    h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }
  }
}

.feature-groups {
  .feature-group {
    margin-bottom: 24px;
    padding: 16px;
    background-color: var(--el-fill-color-lighter);
    border-radius: 8px;
    
    .group-header {
      margin-bottom: 12px;
      font-weight: 500;
    }
    
    .permission-items {
      padding-left: 24px;
    }
    
    .permission-checkbox {
      margin-right: 16px;
      margin-bottom: 8px;
    }
  }
}

.sub-features {
  margin-top: 12px;
  
  .sub-feature {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 8px 0;
    border-bottom: 1px dashed var(--el-border-color-lighter);
    
    &:last-child {
      border-bottom: none;
    }
    
    .feature-name {
      width: 200px;
      flex-shrink: 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    
    .feature-permissions {
      flex: 1;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
  }
}
</style>
