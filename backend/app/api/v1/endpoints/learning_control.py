"""
学习控制 API 端点 v4.0

提供自学习系统的 REST API 控制接口：
- 暂停/恢复/停止学习
- 调整学习方向
- 检查点管理
- 学习进度查询
- 可视化数据
"""
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime

from ....core.exceptions import BusinessError, NotFoundError
from ....db.session import get_db
from ....services.learning_control_service import LearningControlService
from ....api.v1.endpoints.auth import get_current_user
from ....models.user import User

router = APIRouter()


# ============================================================
# 请求/响应模型
# ============================================================

class StartLearningRequest(BaseModel):
    """启动学习请求"""
    goal: str = Field(..., description="学习目标")
    scope: Optional[dict] = Field(None, description="学习范围配置")


class PauseLearningRequest(BaseModel):
    """暂停学习请求"""
    reason: str = Field(..., description="暂停原因")


class StopLearningRequest(BaseModel):
    """停止学习请求"""
    reason: str = Field(..., description="停止原因")
    save_progress: bool = Field(True, description="是否保存进度")


class RedirectLearningRequest(BaseModel):
    """调整学习方向请求"""
    new_direction: str = Field(..., description="新的学习方向/目标")
    reason: str = Field(..., description="调整原因")
    new_scope: Optional[dict] = Field(None, description="新的学习范围（可选）")


class CreateCheckpointRequest(BaseModel):
    """创建检查点请求"""
    reason: str = Field("manual", description="创建原因")
    metadata: Optional[dict] = Field(None, description="额外元数据")


class RollbackRequest(BaseModel):
    """回滚请求"""
    checkpoint_id: str = Field(..., description="目标检查点 ID")
    reason: str = Field(..., description="回滚原因")


class UpdateProgressRequest(BaseModel):
    """更新进度请求"""
    completed_steps: Optional[int] = Field(None, description="已完成步数")
    total_steps: Optional[int] = Field(None, description="总步数")
    current_depth: Optional[int] = Field(None, description="当前探索深度")


class LearningControlResponse(BaseModel):
    """学习控制响应"""
    success: bool
    message: str = ""
    data: Optional[dict] = None


# ============================================================
# 辅助函数
# ============================================================

def get_learning_control_service(db=Depends(get_db)) -> LearningControlService:
    """获取学习控制服务实例"""
    return LearningControlService(db)


# ============================================================
# API 端点
# ============================================================

# ==================== 会话管理 ====================

@router.post("/session/start", response_model=LearningControlResponse)
async def start_learning_session(
    request: StartLearningRequest,
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    启动学习会话
    
    从 FROZEN 状态启动新的学习会话。
    """
    try:
        result = service.start_learning_session(
            goal=request.goal,
            scope=request.scope,
            actor=current_user.username,
        )
        return LearningControlResponse(
            success=result["success"],
            message="Learning session started" if result["success"] else "Failed to start",
            data=result,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/session/current")
async def get_current_session(
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    获取当前学习会话
    """
    session = service.get_current_session()
    if session:
        return {"success": True, "data": session}
    return {"success": False, "message": "No active session", "data": None}


@router.get("/session/history")
async def get_session_history(
    limit: int = Query(20, ge=1, le=100),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    获取会话历史
    """
    history = service.get_session_history(limit=limit)
    return {"success": True, "data": history, "count": len(history)}


# ==================== 学习控制 ====================

@router.post("/pause", response_model=LearningControlResponse)
async def pause_learning(
    request: PauseLearningRequest,
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    暂停学习
    
    将学习状态从 LEARNING 转换为 PAUSED。
    暂停期间会创建检查点，可随时恢复。
    """
    try:
        result = service.pause_learning(
            reason=request.reason,
            actor=current_user.username,
        )
        return LearningControlResponse(
            success=result["success"],
            message=f"Learning paused: {request.reason}",
            data=result,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/resume", response_model=LearningControlResponse)
async def resume_learning(
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    恢复学习
    
    将学习状态从 PAUSED 恢复为 LEARNING。
    """
    try:
        result = service.resume_learning(actor=current_user.username)
        return LearningControlResponse(
            success=result["success"],
            message="Learning resumed",
            data=result,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop", response_model=LearningControlResponse)
async def stop_learning(
    request: StopLearningRequest,
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    停止学习
    
    终止当前学习会话，状态转换为 FROZEN。
    可选择是否保存当前进度到检查点。
    """
    try:
        result = service.stop_learning(
            reason=request.reason,
            actor=current_user.username,
            save_progress=request.save_progress,
        )
        return LearningControlResponse(
            success=result["success"],
            message=f"Learning stopped: {request.reason}",
            data=result,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/redirect", response_model=LearningControlResponse)
async def redirect_learning(
    request: RedirectLearningRequest,
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    调整学习方向
    
    修改当前学习的目标或范围。
    调整前会创建检查点，触发验证流程。
    """
    try:
        result = service.redirect_learning(
            new_direction=request.new_direction,
            reason=request.reason,
            actor=current_user.username,
            new_scope=request.new_scope,
        )
        return LearningControlResponse(
            success=result["success"],
            message=f"Learning redirected: {request.new_direction}",
            data=result,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== 检查点管理 ====================

@router.post("/checkpoint", response_model=LearningControlResponse)
async def create_checkpoint(
    request: CreateCheckpointRequest,
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    创建检查点
    
    手动创建当前学习状态的检查点。
    """
    try:
        checkpoint = service.create_checkpoint(
            reason=request.reason,
            actor=current_user.username,
            metadata=request.metadata,
        )
        return LearningControlResponse(
            success=True,
            message="Checkpoint created",
            data=checkpoint,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/checkpoints")
async def get_checkpoints(
    session_id: Optional[str] = Query(None, description="过滤指定会话的检查点"),
    limit: int = Query(50, ge=1, le=200),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    获取检查点列表
    """
    checkpoints = service.get_checkpoints(session_id=session_id, limit=limit)
    return {"success": True, "data": checkpoints, "count": len(checkpoints)}


@router.post("/rollback", response_model=LearningControlResponse)
async def rollback_to_checkpoint(
    request: RollbackRequest,
    current_user: User = Depends(get_current_user),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    回滚到检查点
    
    将学习状态恢复到指定检查点。
    回滚前会创建当前状态的检查点。
    """
    try:
        result = service.rollback_to_checkpoint(
            checkpoint_id=request.checkpoint_id,
            reason=request.reason,
            actor=current_user.username,
        )
        return LearningControlResponse(
            success=result["success"],
            message=f"Rolled back to checkpoint: {request.checkpoint_id}",
            data=result,
        )
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== 进度监控 ====================

@router.get("/progress")
async def get_learning_progress(
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    获取学习进度
    
    返回当前学习会话的进度信息。
    """
    progress = service.get_learning_progress()
    return {"success": True, "data": progress}


@router.post("/progress", response_model=LearningControlResponse)
async def update_learning_progress(
    request: UpdateProgressRequest,
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    更新学习进度
    
    由自学习系统调用，更新当前进度。
    """
    try:
        progress = service.update_progress(
            completed_steps=request.completed_steps,
            total_steps=request.total_steps,
            current_depth=request.current_depth,
        )
        return LearningControlResponse(
            success=True,
            message="Progress updated",
            data=progress,
        )
    except BusinessError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==================== 可视化 ====================

@router.get("/visualization")
async def get_visualization_data(
    session_id: Optional[str] = Query(None, description="会话 ID（默认当前会话）"),
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    获取学习过程可视化数据
    
    返回用于前端可视化的数据，包括：
    - 状态流转图
    - 时间线事件
    - 进度曲线
    - 检查点标记
    - 方向调整标记
    - 统计摘要
    """
    data = service.get_visualization_data(session_id=session_id)
    return {"success": True, "data": data}


# ==================== 状态查询 ====================

@router.get("/status")
async def get_learning_status(
    service: LearningControlService = Depends(get_learning_control_service),
):
    """
    获取完整的学习状态
    
    包括当前会话、进度、检查点等所有信息。
    """
    session = service.get_current_session()
    progress = service.get_learning_progress()
    
    return {
        "success": True,
        "data": {
            "has_active_session": session is not None,
            "session": session,
            "progress": progress,
            "available_actions": _get_available_actions(session),
        }
    }


def _get_available_actions(session: Optional[dict]) -> List[str]:
    """根据当前状态返回可用操作"""
    if not session:
        return ["start"]
    
    state = session.get("state")
    
    if state == "learning":
        return ["pause", "stop", "redirect", "checkpoint"]
    elif state == "paused":
        return ["resume", "stop", "redirect", "rollback"]
    elif state == "frozen":
        return ["start"]
    else:
        return []
