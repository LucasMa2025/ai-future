<!-- 系统配置页面 -->
<template>
  <div class="config-page art-full-height">
    <ElRow :gutter="16">
      <!-- 配置分组导航 -->
      <ElCol :span="5">
        <ElCard shadow="never" class="group-card">
          <template #header>
            <span>配置分组</span>
          </template>
          
          <div class="group-list">
            <div
              v-for="group in configGroups"
              :key="group.key"
              class="group-item"
              :class="{ active: activeGroup === group.key }"
              @click="activeGroup = group.key"
            >
              <div class="group-icon">
                <i :class="group.icon" />
              </div>
              <div class="group-info">
                <div class="group-name">{{ group.name }}</div>
                <div class="group-count">{{ group.config_count }} 项配置</div>
              </div>
            </div>
          </div>
        </ElCard>
      </ElCol>

      <!-- 配置项列表 -->
      <ElCol :span="19">
        <ElCard shadow="never">
          <template #header>
            <div class="card-header">
              <span>{{ activeGroupName }} 配置</span>
              <div class="actions">
                <ElButton 
                  type="primary" 
                  :loading="saving"
                  @click="handleSave"
                >
                  <template #icon><i class="ri-save-line" /></template>
                  保存更改
                </ElButton>
                <ElButton @click="handleReset">重置</ElButton>
              </div>
            </div>
          </template>

          <div v-loading="loading" class="config-list">
            <ElForm label-position="top">
              <div 
                v-for="config in filteredConfigs" 
                :key="config.config_key"
                class="config-item"
              >
                <div class="config-header">
                  <div class="config-title">
                    <span class="config-name">{{ config.display_name || config.config_key }}</span>
                    <ElTag v-if="config.is_readonly" type="info" size="small">只读</ElTag>
                    <ElTag v-if="config.is_secret" type="warning" size="small">敏感</ElTag>
                  </div>
                  <div class="config-key">{{ config.config_key }}</div>
                </div>
                
                <div class="config-description" v-if="config.description">
                  {{ config.description }}
                </div>
                
                <div class="config-value">
                  <!-- 布尔类型 -->
                  <ElSwitch
                    v-if="config.value_type === 'boolean'"
                    v-model="configValues[config.config_key]"
                    :disabled="config.is_readonly"
                  />
                  
                  <!-- 数字类型 -->
                  <ElInputNumber
                    v-else-if="config.value_type === 'number'"
                    v-model="configValues[config.config_key]"
                    :disabled="config.is_readonly"
                    :min="0"
                    style="width: 200px"
                  />
                  
                  <!-- JSON 类型 -->
                  <ElInput
                    v-else-if="config.value_type === 'json'"
                    v-model="configValues[config.config_key]"
                    type="textarea"
                    :rows="3"
                    :disabled="config.is_readonly"
                    placeholder="JSON 格式"
                  />
                  
                  <!-- 密码/敏感类型 -->
                  <ElInput
                    v-else-if="config.is_secret"
                    v-model="configValues[config.config_key]"
                    type="password"
                    show-password
                    :disabled="config.is_readonly"
                    placeholder="******"
                    style="width: 300px"
                  />
                  
                  <!-- 默认字符串类型 -->
                  <ElInput
                    v-else
                    v-model="configValues[config.config_key]"
                    :disabled="config.is_readonly"
                    style="width: 300px"
                  />
                  
                  <!-- 默认值提示 -->
                  <span v-if="config.default_value" class="default-hint">
                    默认: {{ config.default_value }}
                  </span>
                </div>
              </div>
            </ElForm>
            
            <ElEmpty v-if="filteredConfigs.length === 0" description="暂无配置项" />
          </div>
        </ElCard>
      </ElCol>
    </ElRow>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { systemConfigsApi } from '@/api/system-configs'
import type { SystemConfig, ConfigGroup } from '@/types/system'

defineOptions({ name: 'SystemConfig' })

// 数据
const configGroups = ref<ConfigGroup[]>([])
const configs = ref<SystemConfig[]>([])
const configValues = ref<Record<string, any>>({})
const originalValues = ref<Record<string, any>>({})

// 状态
const loading = ref(false)
const saving = ref(false)
const activeGroup = ref('system')

// 计算属性
const activeGroupName = computed(() => {
  const group = configGroups.value.find(g => g.key === activeGroup.value)
  return group?.name || '系统'
})

const filteredConfigs = computed(() => {
  return configs.value.filter(c => c.config_group === activeGroup.value)
})

// 加载配置分组
const loadGroups = async () => {
  try {
    const res = await systemConfigsApi.getGroups()
    configGroups.value = res.data || []
    
    if (configGroups.value.length > 0 && !activeGroup.value) {
      activeGroup.value = configGroups.value[0].key
    }
  } catch (error) {
    console.error('加载配置分组失败:', error)
  }
}

// 加载配置项
const loadConfigs = async () => {
  loading.value = true
  try {
    const res = await systemConfigsApi.getList()
    configs.value = res.data || []
    
    // 初始化配置值
    initConfigValues()
  } catch (error) {
    console.error('加载配置失败:', error)
  } finally {
    loading.value = false
  }
}

// 初始化配置值
const initConfigValues = () => {
  configValues.value = {}
  originalValues.value = {}
  
  for (const config of configs.value) {
    const value = parseConfigValue(config.config_value, config.value_type)
    configValues.value[config.config_key] = value
    originalValues.value[config.config_key] = value
  }
}

// 解析配置值
const parseConfigValue = (value: string, type: string): any => {
  if (value === null || value === undefined) return ''
  
  switch (type) {
    case 'boolean':
      return value === 'true' || value === '1'
    case 'number':
      return parseFloat(value) || 0
    case 'json':
      try {
        return JSON.stringify(JSON.parse(value), null, 2)
      } catch {
        return value
      }
    default:
      return value
  }
}

// 保存配置
const handleSave = async () => {
  // 收集变更的配置
  const changedConfigs: Record<string, any> = {}
  
  for (const key in configValues.value) {
    const config = configs.value.find(c => c.config_key === key)
    if (!config || config.is_readonly) continue
    
    const newValue = configValues.value[key]
    const oldValue = originalValues.value[key]
    
    if (newValue !== oldValue) {
      changedConfigs[key] = newValue
    }
  }
  
  if (Object.keys(changedConfigs).length === 0) {
    ElMessage.info('没有需要保存的更改')
    return
  }
  
  saving.value = true
  try {
    await systemConfigsApi.batchUpdate({ configs: changedConfigs })
    ElMessage.success(`已保存 ${Object.keys(changedConfigs).length} 项配置`)
    
    // 更新原始值
    Object.assign(originalValues.value, changedConfigs)
    
    // 重新加载
    await loadConfigs()
  } catch (error) {
    ElMessage.error('保存配置失败')
    console.error('保存配置失败:', error)
  } finally {
    saving.value = false
  }
}

// 重置配置
const handleReset = () => {
  for (const key in originalValues.value) {
    configValues.value[key] = originalValues.value[key]
  }
  ElMessage.info('已重置为保存前的值')
}

// 初始化
onMounted(async () => {
  await loadGroups()
  await loadConfigs()
})
</script>

<style scoped lang="scss">
.config-page {
  padding: 16px;
}

.group-card {
  height: 100%;
  
  .group-list {
    .group-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      margin-bottom: 8px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
      
      &:hover {
        background-color: var(--el-fill-color-light);
      }
      
      &.active {
        background-color: var(--el-color-primary-light-9);
        
        .group-icon {
          background-color: var(--el-color-primary);
          
          i {
            color: #fff;
          }
        }
        
        .group-name {
          color: var(--el-color-primary);
          font-weight: 500;
        }
      }
      
      .group-icon {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background-color: var(--el-fill-color-light);
        border-radius: 8px;
        
        i {
          font-size: 20px;
          color: var(--el-text-color-secondary);
        }
      }
      
      .group-info {
        flex: 1;
        
        .group-name {
          font-size: 14px;
          margin-bottom: 2px;
        }
        
        .group-count {
          font-size: 12px;
          color: var(--el-text-color-secondary);
        }
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

.config-list {
  .config-item {
    padding: 20px;
    margin-bottom: 16px;
    background-color: var(--el-fill-color-lighter);
    border-radius: 8px;
    
    &:last-child {
      margin-bottom: 0;
    }
    
    .config-header {
      margin-bottom: 8px;
      
      .config-title {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
        
        .config-name {
          font-size: 15px;
          font-weight: 500;
        }
      }
      
      .config-key {
        font-size: 12px;
        color: var(--el-text-color-placeholder);
        font-family: monospace;
      }
    }
    
    .config-description {
      font-size: 13px;
      color: var(--el-text-color-secondary);
      margin-bottom: 12px;
    }
    
    .config-value {
      display: flex;
      align-items: center;
      gap: 12px;
      
      .default-hint {
        font-size: 12px;
        color: var(--el-text-color-placeholder);
      }
    }
  }
}
</style>
