"""
邮件服务

实现:
1. 异步邮件发送
2. 模板支持
3. 重试机制
"""
import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import aiosmtplib

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    """邮件消息"""
    to: List[str]
    subject: str
    body: str
    html_body: Optional[str] = None
    cc: Optional[List[str]] = None
    bcc: Optional[List[str]] = None
    reply_to: Optional[str] = None


class EmailService:
    """
    邮件服务
    
    提供异步邮件发送功能
    """
    
    def __init__(self):
        self.host = settings.SMTP_HOST
        self.port = settings.SMTP_PORT
        self.username = settings.SMTP_USER
        self.password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM_EMAIL
        self.from_name = settings.SMTP_FROM_NAME
        self.use_tls = settings.SMTP_TLS
        self.enabled = settings.EMAIL_ENABLED
    
    async def send(
        self,
        to: List[str],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        发送邮件
        
        Args:
            to: 收件人列表
            subject: 主题
            body: 纯文本内容
            html_body: HTML 内容（可选）
            
        Returns:
            是否发送成功
        """
        if not self.enabled:
            logger.warning("Email service is disabled")
            return False
        
        message = EmailMessage(
            to=to,
            subject=subject,
            body=body,
            html_body=html_body,
            **kwargs
        )
        
        return await self._send_with_retry(message)
    
    async def send_notification(
        self,
        to_email: str,
        notification_type: str,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        发送通知邮件
        
        使用预定义模板
        """
        html_body = self._render_notification_template(
            notification_type=notification_type,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        return await self.send(
            to=[to_email],
            subject=f"[NLGSM] {title}",
            body=message,
            html_body=html_body
        )
    
    async def send_approval_request(
        self,
        to_email: str,
        approver_name: str,
        target_type: str,
        target_id: str,
        risk_level: str,
        deadline: Optional[str] = None
    ) -> bool:
        """发送审批请求邮件"""
        subject = f"[NLGSM] 审批请求 - {target_type}"
        
        body = f"""
您好 {approver_name},

您有一个新的审批请求需要处理:

类型: {target_type}
ID: {target_id}
风险等级: {risk_level}
截止时间: {deadline or '无'}

请登录 NLGSM 系统进行审批。

---
此邮件由 NLGSM 系统自动发送，请勿回复。
        """.strip()
        
        html_body = self._render_approval_template(
            approver_name=approver_name,
            target_type=target_type,
            target_id=target_id,
            risk_level=risk_level,
            deadline=deadline
        )
        
        return await self.send(
            to=[to_email],
            subject=subject,
            body=body,
            html_body=html_body
        )
    
    async def send_alert(
        self,
        to_emails: List[str],
        alert_type: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """发送告警邮件"""
        subject = f"[NLGSM 告警] [{severity.upper()}] {alert_type}"
        
        body = f"""
NLGSM 系统告警

类型: {alert_type}
严重性: {severity}
消息: {message}

详情:
{self._format_details(details or {})}

请立即处理。

---
此邮件由 NLGSM 系统自动发送。
        """.strip()
        
        return await self.send(
            to=to_emails,
            subject=subject,
            body=body
        )
    
    async def _send_with_retry(
        self,
        message: EmailMessage,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> bool:
        """带重试的发送"""
        for attempt in range(max_retries):
            try:
                await self._send_email(message)
                logger.info(f"Email sent to {message.to}")
                return True
            except Exception as e:
                logger.warning(
                    f"Email send attempt {attempt + 1} failed: {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
        
        logger.error(f"Failed to send email to {message.to} after {max_retries} attempts")
        return False
    
    async def _send_email(self, message: EmailMessage):
        """实际发送邮件"""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = message.subject
        msg["From"] = f"{self.from_name} <{self.from_email}>"
        msg["To"] = ", ".join(message.to)
        
        if message.cc:
            msg["Cc"] = ", ".join(message.cc)
        if message.reply_to:
            msg["Reply-To"] = message.reply_to
        
        # 添加纯文本部分
        msg.attach(MIMEText(message.body, "plain", "utf-8"))
        
        # 添加 HTML 部分（如果有）
        if message.html_body:
            msg.attach(MIMEText(message.html_body, "html", "utf-8"))
        
        # 发送
        await aiosmtplib.send(
            msg,
            hostname=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            start_tls=self.use_tls,
        )
    
    def _render_notification_template(
        self,
        notification_type: str,
        title: str,
        message: str,
        metadata: Dict[str, Any]
    ) -> str:
        """渲染通知模板"""
        # 根据类型选择颜色
        color_map = {
            "approval_required": "#f59e0b",
            "state_changed": "#3b82f6",
            "anomaly_detected": "#ef4444",
            "safe_halt_triggered": "#dc2626",
            "approval_completed": "#10b981",
            "system_alert": "#6366f1",
        }
        color = color_map.get(notification_type, "#6b7280")
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: {color}; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9fafb; padding: 20px; border-radius: 0 0 8px 8px; }}
        .footer {{ color: #6b7280; font-size: 12px; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 style="margin: 0;">{title}</h2>
        </div>
        <div class="content">
            <p>{message}</p>
            {self._render_metadata(metadata)}
        </div>
        <div class="footer">
            此邮件由 NLGSM 系统自动发送
        </div>
    </div>
</body>
</html>
        """
    
    def _render_approval_template(
        self,
        approver_name: str,
        target_type: str,
        target_id: str,
        risk_level: str,
        deadline: Optional[str]
    ) -> str:
        """渲染审批请求模板"""
        risk_colors = {
            "low": "#10b981",
            "medium": "#f59e0b",
            "high": "#f97316",
            "critical": "#ef4444"
        }
        risk_color = risk_colors.get(risk_level, "#6b7280")
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #3b82f6; color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
        .content {{ background: #f9fafb; padding: 20px; border-radius: 0 0 8px 8px; }}
        .info-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e5e7eb; }}
        .risk-badge {{ background: {risk_color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; }}
        .btn {{ display: inline-block; padding: 12px 24px; background: #3b82f6; color: white; text-decoration: none; border-radius: 6px; }}
        .footer {{ color: #6b7280; font-size: 12px; margin-top: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2 style="margin: 0;">审批请求</h2>
        </div>
        <div class="content">
            <p>您好 {approver_name},</p>
            <p>您有一个新的审批请求需要处理:</p>
            
            <div class="info-row">
                <span>类型</span>
                <span>{target_type}</span>
            </div>
            <div class="info-row">
                <span>ID</span>
                <span>{target_id}</span>
            </div>
            <div class="info-row">
                <span>风险等级</span>
                <span class="risk-badge">{risk_level.upper()}</span>
            </div>
            <div class="info-row">
                <span>截止时间</span>
                <span>{deadline or '无限制'}</span>
            </div>
            
            <p style="text-align: center; margin-top: 20px;">
                <a href="#" class="btn">前往审批</a>
            </p>
        </div>
        <div class="footer">
            此邮件由 NLGSM 系统自动发送，请勿回复
        </div>
    </div>
</body>
</html>
        """
    
    def _render_metadata(self, metadata: Dict[str, Any]) -> str:
        """渲染元数据"""
        if not metadata:
            return ""
        
        rows = []
        for key, value in metadata.items():
            rows.append(f"<p><strong>{key}:</strong> {value}</p>")
        
        return "".join(rows)
    
    def _format_details(self, details: Dict[str, Any]) -> str:
        """格式化详情"""
        if not details:
            return "无"
        
        lines = []
        for key, value in details.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


# 全局邮件服务实例
email_service = EmailService()

