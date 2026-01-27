"""
WebSocket 路由
"""
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from jose import JWTError
from sqlalchemy.orm import Session

from ..db.session import get_db
from ..services.websocket_service import websocket_manager
from ..middleware.auth import JWTAuthMiddleware
from ..models.user import User

logger = logging.getLogger(__name__)

websocket_router = APIRouter()


@websocket_router.websocket("/notifications")
async def websocket_notifications(
    websocket: WebSocket,
    token: str = Query(..., description="JWT Token"),
    db: Session = Depends(get_db),
):
    """
    通知 WebSocket 端点
    
    连接时需要提供有效的 JWT Token
    """
    user_id = None
    username = None
    
    try:
        # 验证 Token
        payload = JWTAuthMiddleware.decode_token(token)
        user_id = payload.sub
        
        # 获取用户
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            await websocket.close(code=4001, reason="User not found")
            return
        
        if not user.is_active:
            await websocket.close(code=4002, reason="User is inactive")
            return
        
        username = user.username
        
        # 建立连接
        await websocket_manager.connect(websocket, user_id, username)
        
        # 自动订阅用户相关主题
        await websocket_manager.subscribe(user_id, f"user:{user_id}")
        
        # 订阅角色相关主题
        for role in user.roles:
            await websocket_manager.subscribe(user_id, f"role:{role.name}")
        
        # 处理消息
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                await websocket_manager.handle_message(user_id, websocket, message)
            except json.JSONDecodeError:
                await websocket_manager.send_to_user(user_id, {
                    "type": "error",
                    "message": "Invalid JSON format"
                })
    
    except JWTError as e:
        logger.warning(f"WebSocket auth failed: {e}")
        await websocket.close(code=4000, reason="Invalid token")
    
    except WebSocketDisconnect:
        if user_id:
            await websocket_manager.disconnect(user_id, websocket)
            logger.info(f"WebSocket disconnected: {username}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if user_id:
            await websocket_manager.disconnect(user_id, websocket)


@websocket_router.websocket("/state-updates")
async def websocket_state_updates(
    websocket: WebSocket,
    token: str = Query(..., description="JWT Token"),
    db: Session = Depends(get_db),
):
    """
    状态更新 WebSocket 端点
    
    订阅系统状态变化的实时通知
    """
    user_id = None
    
    try:
        # 验证 Token
        payload = JWTAuthMiddleware.decode_token(token)
        user_id = payload.sub
        
        # 获取用户
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            await websocket.close(code=4001, reason="Unauthorized")
            return
        
        # 建立连接
        await websocket_manager.connect(websocket, user_id, user.username)
        
        # 订阅状态更新主题
        await websocket_manager.subscribe(user_id, "state:updates")
        await websocket_manager.subscribe(user_id, "anomaly:alerts")
        
        # 处理消息
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            await websocket_manager.handle_message(user_id, websocket, message)
    
    except JWTError:
        await websocket.close(code=4000, reason="Invalid token")
    
    except WebSocketDisconnect:
        if user_id:
            await websocket_manager.disconnect(user_id, websocket)
    
    except Exception as e:
        logger.error(f"State WebSocket error: {e}")
        if user_id:
            await websocket_manager.disconnect(user_id, websocket)

