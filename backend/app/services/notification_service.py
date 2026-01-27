"""
通知服务

实现:
1. 统一通知接口
2. WebSocket + Email 双路发送
3. 用户通知偏好管理
"""
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy.orm import Session

from ..config import settings
from ..models.notification import Notification, UserNotificationSetting
from ..models.user import User
from ..core.enums import NotificationType, NotificationChannel
from .websocket_service import websocket_manager
from .email_service import email_service

logger = logging.getLogger(__name__)


class NotificationService:
    """
    通知服务
    
    提供统一的通知发送接口，支持 WebSocket 和 Email 双路方案
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    async def send(
        self,
        user_id: UUID,
        notification_type: NotificationType,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        channel: NotificationChannel = NotificationChannel.BOTH,
        priority: str = "normal",
        related_type: Optional[str] = None,
        related_id: Optional[str] = None,
    ) -> Notification:
        """
        发送通知
        
        Args:
            user_id: 接收用户ID
            notification_type: 通知类型
            title: 标题
            message: 消息内容
            metadata: 附加数据
            channel: 发送渠道
            priority: 优先级
            related_type: 关联对象类型
            related_id: 关联对象ID
            
        Returns:
            创建的通知对象
        """
        # 获取用户
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            logger.warning(f"User not found: {user_id}")
            return None
        
        # 获取用户通知偏好
        preferences = self._get_user_preferences(user_id, notification_type)
        
        # 根据偏好调整渠道
        effective_channel = self._apply_preferences(channel, preferences)
        
        # 创建通知记录
        notification = Notification(
            user_id=user_id,
            type=notification_type.value,
            category=self._get_category(notification_type),
            channel=effective_channel.value,
            title=title,
            message=message,
            metadata=metadata or {},
            related_type=related_type,
            related_id=related_id,
            priority=priority,
        )
        
        self.db.add(notification)
        self.db.commit()
        self.db.refresh(notification)
        
        # 发送通知
        await self._dispatch(notification, user, effective_channel)
        
        return notification
    
    async def send_to_users(
        self,
        user_ids: List[UUID],
        notification_type: NotificationType,
        title: str,
        message: str,
        **kwargs
    ) -> List[Notification]:
        """发送通知给多个用户"""
        notifications = []
        for user_id in user_ids:
            notification = await self.send(
                user_id=user_id,
                notification_type=notification_type,
                title=title,
                message=message,
                **kwargs
            )
            if notification:
                notifications.append(notification)
        
        return notifications
    
    async def send_to_roles(
        self,
        role_names: List[str],
        notification_type: NotificationType,
        title: str,
        message: str,
        **kwargs
    ) -> List[Notification]:
        """发送通知给指定角色的所有用户"""
        from ..models.user import Role
        
        # 获取角色对应的用户
        users = self.db.query(User).join(User.roles).filter(
            Role.name.in_(role_names),
            User.is_active == True
        ).distinct().all()
        
        user_ids = [user.id for user in users]
        return await self.send_to_users(user_ids, notification_type, title, message, **kwargs)
    
    async def broadcast(
        self,
        notification_type: NotificationType,
        title: str,
        message: str,
        **kwargs
    ) -> int:
        """广播通知给所有用户"""
        users = self.db.query(User).filter(User.is_active == True).all()
        notifications = await self.send_to_users(
            [user.id for user in users],
            notification_type,
            title,
            message,
            **kwargs
        )
        return len(notifications)
    
    # ==================== 预定义通知方法 ====================
    
    async def send_approval_required(
        self,
        user_id: UUID,
        target_type: str,
        target_id: str,
        risk_level: str,
        deadline: Optional[datetime] = None
    ) -> Notification:
        """发送审批请求通知"""
        return await self.send(
            user_id=user_id,
            notification_type=NotificationType.APPROVAL_REQUIRED,
            title=f"新的审批请求 - {target_type}",
            message=f"您有一个新的 {target_type} 审批请求需要处理，风险等级: {risk_level}",
            metadata={
                "target_type": target_type,
                "target_id": target_id,
                "risk_level": risk_level,
                "deadline": deadline.isoformat() if deadline else None,
            },
            priority="high" if risk_level in ("high", "critical") else "normal",
            related_type=target_type,
            related_id=target_id,
        )
    
    async def send_state_changed(
        self,
        user_id: UUID,
        from_state: str,
        to_state: str,
        trigger_event: str
    ) -> Notification:
        """发送状态变更通知"""
        return await self.send(
            user_id=user_id,
            notification_type=NotificationType.STATE_CHANGED,
            title="系统状态变更",
            message=f"系统状态从 {from_state} 变更为 {to_state}",
            metadata={
                "from_state": from_state,
                "to_state": to_state,
                "trigger_event": trigger_event,
            },
            related_type="state",
            related_id=to_state,
        )
    
    async def send_anomaly_detected(
        self,
        user_id: UUID,
        severity: str,
        anomaly_type: str,
        details: Dict[str, Any]
    ) -> Notification:
        """发送异常检测通知"""
        priority_map = {
            "low": "low",
            "medium": "normal",
            "high": "high",
            "critical": "urgent"
        }
        
        return await self.send(
            user_id=user_id,
            notification_type=NotificationType.ANOMALY_DETECTED,
            title=f"异常检测 - {severity.upper()}",
            message=f"检测到 {anomaly_type} 异常，严重性: {severity}",
            metadata={
                "severity": severity,
                "anomaly_type": anomaly_type,
                **details,
            },
            priority=priority_map.get(severity, "normal"),
            related_type="anomaly",
        )
    
    async def send_safe_halt_triggered(
        self,
        user_ids: List[UUID],
        reason: str,
        details: Dict[str, Any]
    ) -> List[Notification]:
        """发送安全停机通知"""
        return await self.send_to_users(
            user_ids=user_ids,
            notification_type=NotificationType.SAFE_HALT_TRIGGERED,
            title="⚠️ 安全停机已触发",
            message=f"系统已进入安全停机状态。原因: {reason}",
            metadata={"reason": reason, **details},
            priority="urgent",
            channel=NotificationChannel.BOTH,  # 强制双路
        )
    
    # ==================== 通知管理方法 ====================
    
    def get_user_notifications(
        self,
        user_id: UUID,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> List[Notification]:
        """获取用户通知列表"""
        query = self.db.query(Notification).filter(
            Notification.user_id == user_id
        )
        
        if unread_only:
            query = query.filter(Notification.is_read == False)
        
        return query.order_by(
            Notification.created_at.desc()
        ).offset(offset).limit(limit).all()
    
    def mark_as_read(
        self,
        notification_id: UUID,
        user_id: UUID
    ) -> bool:
        """标记通知为已读"""
        notification = self.db.query(Notification).filter(
            Notification.id == notification_id,
            Notification.user_id == user_id
        ).first()
        
        if not notification:
            return False
        
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        self.db.commit()
        
        return True
    
    def mark_all_as_read(self, user_id: UUID) -> int:
        """标记所有通知为已读"""
        count = self.db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.is_read == False
        ).update({
            "is_read": True,
            "read_at": datetime.utcnow()
        })
        
        self.db.commit()
        return count
    
    def get_unread_count(self, user_id: UUID) -> int:
        """获取未读通知数量"""
        return self.db.query(Notification).filter(
            Notification.user_id == user_id,
            Notification.is_read == False
        ).count()
    
    # ==================== 私有方法 ====================
    
    async def _dispatch(
        self,
        notification: Notification,
        user: User,
        channel: NotificationChannel
    ):
        """分发通知"""
        ws_message = {
            "type": "notification",
            "data": {
                "id": str(notification.id),
                "notification_type": notification.type,
                "category": notification.category,
                "title": notification.title,
                "message": notification.message,
                "metadata": notification.metadata,
                "priority": notification.priority,
                "created_at": notification.created_at.isoformat(),
            }
        }
        
        # WebSocket 发送
        if channel in (NotificationChannel.WEBSOCKET, NotificationChannel.BOTH):
            ws_sent = await websocket_manager.send_to_user(
                str(user.id),
                ws_message
            )
            
            if ws_sent:
                notification.ws_sent = True
                notification.ws_sent_at = datetime.utcnow()
                self.db.commit()
        
        # Email 发送
        if channel in (NotificationChannel.EMAIL, NotificationChannel.BOTH):
            if user.email and settings.EMAIL_ENABLED:
                try:
                    email_sent = await email_service.send_notification(
                        to_email=user.email,
                        notification_type=notification.type,
                        title=notification.title,
                        message=notification.message,
                        metadata=notification.metadata
                    )
                    
                    notification.email_sent = email_sent
                    notification.email_sent_at = datetime.utcnow() if email_sent else None
                    self.db.commit()
                    
                except Exception as e:
                    notification.email_error = str(e)
                    self.db.commit()
                    logger.error(f"Email send failed: {e}")
    
    def _get_user_preferences(
        self,
        user_id: UUID,
        notification_type: NotificationType
    ) -> Optional[UserNotificationSetting]:
        """获取用户通知偏好"""
        return self.db.query(UserNotificationSetting).filter(
            UserNotificationSetting.user_id == user_id,
            UserNotificationSetting.notification_type == notification_type.value
        ).first()
    
    def _apply_preferences(
        self,
        channel: NotificationChannel,
        preferences: Optional[UserNotificationSetting]
    ) -> NotificationChannel:
        """应用用户偏好"""
        if not preferences:
            return channel
        
        # 检查是否静音
        if preferences.muted:
            if preferences.muted_until and preferences.muted_until > datetime.utcnow():
                return NotificationChannel.WEBSOCKET  # 静音期只发 WebSocket
        
        # 根据偏好调整渠道
        if channel == NotificationChannel.BOTH:
            if not preferences.websocket_enabled and preferences.email_enabled:
                return NotificationChannel.EMAIL
            elif preferences.websocket_enabled and not preferences.email_enabled:
                return NotificationChannel.WEBSOCKET
        
        return channel
    
    def _get_category(self, notification_type: NotificationType) -> str:
        """获取通知分类"""
        category_map = {
            NotificationType.APPROVAL_REQUIRED: "approval",
            NotificationType.APPROVAL_COMPLETED: "approval",
            NotificationType.STATE_CHANGED: "state",
            NotificationType.ANOMALY_DETECTED: "anomaly",
            NotificationType.SAFE_HALT_TRIGGERED: "anomaly",
            NotificationType.SYSTEM_ALERT: "system",
        }
        return category_map.get(notification_type, "system")

