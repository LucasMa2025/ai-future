"""
NLGSM 后端应用主入口
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .api.v1 import api_router
from .api.websocket import websocket_router
from .db.session import engine
from .db.redis import redis_manager
from .db.init_db import init_db
from .db.session import SessionLocal
from .models.base import Base
from .middleware.audit import AuditMiddleware
from .middleware.logging import LoggingMiddleware
from .services.websocket_service import websocket_manager
from .core.exceptions import NLGSMException

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # ==================== 启动 ====================
    logger.info("Starting NLGSM Backend...")
    
    # 创建数据库表
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")
    
    # 初始化数据库（预置数据）
    db = SessionLocal()
    try:
        init_db(db)
    finally:
        db.close()
    
    # 初始化 Redis
    await redis_manager.init()
    logger.info("Redis connection established")
    
    # 启动 WebSocket 管理器
    await websocket_manager.start()
    logger.info("WebSocket manager started")
    
    logger.info("NLGSM Backend started successfully")
    
    yield
    
    # ==================== 关闭 ====================
    logger.info("Shutting down NLGSM Backend...")
    
    # 停止 WebSocket 管理器
    await websocket_manager.stop()
    
    # 关闭 Redis
    await redis_manager.close()
    
    logger.info("NLGSM Backend shutdown complete")


# 创建应用
app = FastAPI(
    title=settings.APP_NAME,
    description="Nested Learning Governance State Machine - 治理后端系统",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ==================== 中间件 ====================

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 日志中间件
app.add_middleware(LoggingMiddleware)

# 审计中间件
app.add_middleware(AuditMiddleware)


# ==================== 异常处理 ====================

@app.exception_handler(NLGSMException)
async def nlgsm_exception_handler(request: Request, exc: NLGSMException):
    """自定义异常处理"""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "code": exc.code,
            "details": exc.details,
        }
    )


# ==================== 路由 ====================

# API 路由
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# WebSocket 路由
app.include_router(websocket_router, prefix="/ws")


# ==================== 健康检查 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "app_name": settings.APP_NAME,
    }


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "NLGSM API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
    }


# ==================== 开发入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )

