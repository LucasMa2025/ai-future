/**
 * NLGSM 治理系统路由配置
 */
import { AppRouteRecord } from '@/types/router'

export const nlgsmRoutes: AppRouteRecord = {
  path: '/nlgsm',
  name: 'NLGSM',
  component: '/index/index',
  meta: {
    title: 'menus.nlgsm.title',
    icon: 'ri:shield-check-line',
    roles: ['R_SUPER', 'R_ADMIN', 'R_AUDITOR']
  },
  children: [
    {
      path: 'dashboard',
      name: 'NLGSMDashboard',
      component: '/nlgsm/dashboard',
      meta: {
        title: 'menus.nlgsm.dashboard',
        icon: 'ri:dashboard-3-line',
        keepAlive: true,
        fixedTab: true
      }
    },
    {
      path: 'state-machine',
      name: 'StateMachine',
      component: '/nlgsm/state-machine',
      meta: {
        title: 'menus.nlgsm.stateMachine',
        icon: 'ri:flow-chart',
        keepAlive: true
      }
    },
    {
      path: 'learning-units',
      name: 'LearningUnits',
      component: '/nlgsm/learning-units',
      meta: {
        title: 'menus.nlgsm.learningUnits',
        icon: 'ri:book-2-line',
        keepAlive: true
      }
    },
    {
      path: 'learning-units/:id',
      name: 'LearningUnitDetail',
      component: '/nlgsm/learning-units/detail',
      meta: {
        title: 'menus.nlgsm.learningUnitDetail',
        isHide: true,
        isHideTab: false
      }
    },
    {
      path: 'approvals',
      name: 'Approvals',
      component: '/nlgsm/approvals',
      meta: {
        title: 'menus.nlgsm.approvals',
        icon: 'ri:checkbox-circle-line',
        keepAlive: true
      }
    },
    {
      path: 'artifacts',
      name: 'Artifacts',
      component: '/nlgsm/artifacts',
      meta: {
        title: 'menus.nlgsm.artifacts',
        icon: 'ri:archive-line',
        keepAlive: true
      }
    },
    {
      path: 'artifacts/:id',
      name: 'ArtifactDetail',
      component: '/nlgsm/artifacts/detail',
      meta: {
        title: 'menus.nlgsm.artifactDetail',
        isHide: true,
        isHideTab: false
      }
    },
    {
      path: 'audit',
      name: 'AuditLogs',
      component: '/nlgsm/audit',
      meta: {
        title: 'menus.nlgsm.audit',
        icon: 'ri:file-list-3-line',
        keepAlive: true
      }
    },
    {
      path: 'anomaly',
      name: 'AnomalyMonitor',
      component: '/nlgsm/anomaly',
      meta: {
        title: 'menus.nlgsm.anomaly',
        icon: 'ri:alarm-warning-line',
        keepAlive: true
      }
    },
    {
      path: 'learning',
      name: 'LearningControl',
      component: '/nlgsm/learning',
      meta: {
        title: 'menus.nlgsm.learning',
        icon: 'ri:play-circle-line',
        keepAlive: true
      }
    },
    {
      path: 'learning/visualization',
      name: 'LearningVisualization',
      component: '/nlgsm/learning/visualization',
      meta: {
        title: 'menus.nlgsm.visualization',
        isHide: true,
        isHideTab: false
      }
    }
  ]
}
