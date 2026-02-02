import { AppRouteRecord } from '@/types/router'

export const systemRoutes: AppRouteRecord = {
  path: '/system',
  name: 'System',
  component: '/index/index',
  meta: {
    title: 'menus.system.title',
    icon: 'ri:settings-3-line',
    roles: ['R_SUPER', 'R_ADMIN']
  },
  children: [
    {
      path: 'user',
      name: 'User',
      component: '/system/user',
      meta: {
        title: 'menus.system.user',
        icon: 'ri:user-3-line',
        keepAlive: true,
        roles: ['R_SUPER', 'R_ADMIN']
      }
    },
    {
      path: 'role',
      name: 'Role',
      component: '/system/role',
      meta: {
        title: 'menus.system.role',
        icon: 'ri:user-settings-line',
        keepAlive: true,
        roles: ['R_SUPER']
      }
    },
    {
      path: 'permission',
      name: 'Permission',
      component: '/system/permission',
      meta: {
        title: 'menus.system.permission',
        icon: 'ri:shield-keyhole-line',
        keepAlive: true,
        roles: ['R_SUPER']
      }
    },
    {
      path: 'operation-log',
      name: 'OperationLog',
      component: '/system/operation-log',
      meta: {
        title: 'menus.system.operationLog',
        icon: 'ri:file-list-3-line',
        keepAlive: true,
        roles: ['R_SUPER', 'R_ADMIN']
      }
    },
    {
      path: 'backup',
      name: 'Backup',
      component: '/system/backup',
      meta: {
        title: 'menus.system.backup',
        icon: 'ri:database-2-line',
        keepAlive: true,
        roles: ['R_SUPER']
      }
    },
    {
      path: 'config',
      name: 'SystemConfig',
      component: '/system/config',
      meta: {
        title: 'menus.system.config',
        icon: 'ri:settings-4-line',
        keepAlive: true,
        roles: ['R_SUPER']
      }
    },
    {
      path: 'user-center',
      name: 'UserCenter',
      component: '/system/user-center',
      meta: {
        title: 'menus.system.userCenter',
        isHide: true,
        keepAlive: true,
        isHideTab: true
      }
    },
    {
      path: 'menu',
      name: 'Menus',
      component: '/system/menu',
      meta: {
        title: 'menus.system.menu',
        icon: 'ri:menu-2-line',
        keepAlive: true,
        roles: ['R_SUPER'],
        authList: [
          { title: '新增', authMark: 'add' },
          { title: '编辑', authMark: 'edit' },
          { title: '删除', authMark: 'delete' }
        ]
      }
    }
  ]
}
