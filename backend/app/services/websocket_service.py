"""
WebSocket 服务

实现:
1. WebSocket 连接管理
2. 用户连接状态维护
3. 消息广播和定向推送
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """连接信息"""
    websocket: WebSocket
    user_id: str
    username: str
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    subscriptions: Set[str] = field(default_factory=set)


class WebSocketManager:
    """
    WebSocket 连接管理器
    
    管理所有活跃的 WebSocket 连接
    """
    
    def __init__(self):
        # 用户ID -> 连接列表 (一个用户可以有多个连接)
        self._connections: Dict[str, list[ConnectionInfo]] = {}
        # 订阅主题 -> 用户ID集合
        self._subscriptions: Dict[str, Set[str]] = {}
        # 心跳任务
        self._heartbeat_task: Optional[asyncio.Task] = None
        # 锁
        self._lock = asyncio.Lock()
    
    async def start(self):
        """启动管理器"""
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket manager started")
    
    async def stop(self):
        """停止管理器"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        async with self._lock:
            for user_id, connections in self._connections.items():
                for conn in connections:
                    try:
                        await conn.websocket.close()
                    except:
                        pass
            self._connections.clear()
            self._subscriptions.clear()
        
        logger.info("WebSocket manager stopped")
    
    async def connect(
        self,
        websocket: WebSocket,
        user_id: str,
        username: str
    ) -> ConnectionInfo:
        """
        建立连接
        """
        await websocket.accept()
        
        conn_info = ConnectionInfo(
            websocket=websocket,
            user_id=user_id,
            username=username,
        )
        
        async with self._lock:
            if user_id not in self._connections:
                self._connections[user_id] = []
            self._connections[user_id].append(conn_info)
        
        logger.info(f"WebSocket connected: user={username}, id={user_id}")
        
        # 发送欢迎消息
        await self.send_to_user(user_id, {
            "type": "connected",
            "message": "Connected to NLGSM notification service",
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        return conn_info
    
    async def disconnect(self, user_id: str, websocket: WebSocket):
        """断开连接"""
        async with self._lock:
            if user_id in self._connections:
                self._connections[user_id] = [
                    conn for conn in self._connections[user_id]
                    if conn.websocket != websocket
                ]
                
                if not self._connections[user_id]:
                    del self._connections[user_id]
                    
                    # 清理订阅
                    for topic, subscribers in self._subscriptions.items():
                        subscribers.discard(user_id)
        
        logger.info(f"WebSocket disconnected: user_id={user_id}")
    
    async def subscribe(self, user_id: str, topic: str):
        """订阅主题"""
        async with self._lock:
            if topic not in self._subscriptions:
                self._subscriptions[topic] = set()
            self._subscriptions[topic].add(user_id)
            
            # 更新连接的订阅列表
            if user_id in self._connections:
                for conn in self._connections[user_id]:
                    conn.subscriptions.add(topic)
        
        logger.debug(f"User {user_id} subscribed to {topic}")
    
    async def unsubscribe(self, user_id: str, topic: str):
        """取消订阅"""
        async with self._lock:
            if topic in self._subscriptions:
                self._subscriptions[topic].discard(user_id)
                
            if user_id in self._connections:
                for conn in self._connections[user_id]:
                    conn.subscriptions.discard(topic)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]) -> bool:
        """
        发送消息给指定用户
        
        Returns:
            是否发送成功（至少发送到一个连接）
        """
        connections = self._connections.get(user_id, [])
        if not connections:
            return False
        
        sent = False
        failed_connections = []
        
        for conn in connections:
            try:
                if conn.websocket.client_state == WebSocketState.CONNECTED:
                    await conn.websocket.send_json(message)
                    sent = True
                else:
                    failed_connections.append(conn)
            except Exception as e:
                logger.warning(f"Failed to send to {user_id}: {e}")
                failed_connections.append(conn)
        
        # 清理失败的连接
        if failed_connections:
            async with self._lock:
                if user_id in self._connections:
                    self._connections[user_id] = [
                        c for c in self._connections[user_id]
                        if c not in failed_connections
                    ]
        
        return sent
    
    async def send_to_topic(self, topic: str, message: Dict[str, Any]) -> int:
        """
        发送消息给主题订阅者
        
        Returns:
            成功发送的用户数
        """
        subscribers = self._subscriptions.get(topic, set())
        if not subscribers:
            return 0
        
        count = 0
        for user_id in subscribers:
            if await self.send_to_user(user_id, message):
                count += 1
        
        return count
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """
        广播消息给所有用户
        
        Returns:
            成功发送的用户数
        """
        count = 0
        for user_id in list(self._connections.keys()):
            if await self.send_to_user(user_id, message):
                count += 1
        
        return count
    
    def get_online_users(self) -> list[str]:
        """获取在线用户列表"""
        return list(self._connections.keys())
    
    def get_connection_count(self) -> int:
        """获取连接总数"""
        return sum(len(conns) for conns in self._connections.values())
    
    def is_user_online(self, user_id: str) -> bool:
        """检查用户是否在线"""
        return user_id in self._connections and len(self._connections[user_id]) > 0
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while True:
            try:
                await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
                await self._send_heartbeats()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
    
    async def _send_heartbeats(self):
        """发送心跳"""
        message = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        for user_id in list(self._connections.keys()):
            await self.send_to_user(user_id, message)
    
    async def handle_message(
        self,
        user_id: str,
        websocket: WebSocket,
        message: Dict[str, Any]
    ):
        """处理来自客户端的消息"""
        msg_type = message.get("type")
        
        if msg_type == "subscribe":
            topic = message.get("topic")
            if topic:
                await self.subscribe(user_id, topic)
                await self.send_to_user(user_id, {
                    "type": "subscribed",
                    "topic": topic,
                })
        
        elif msg_type == "unsubscribe":
            topic = message.get("topic")
            if topic:
                await self.unsubscribe(user_id, topic)
                await self.send_to_user(user_id, {
                    "type": "unsubscribed",
                    "topic": topic,
                })
        
        elif msg_type == "pong":
            # 更新心跳时间
            async with self._lock:
                if user_id in self._connections:
                    for conn in self._connections[user_id]:
                        if conn.websocket == websocket:
                            conn.last_heartbeat = datetime.utcnow()


# 全局 WebSocket 管理器实例
websocket_manager = WebSocketManager()

