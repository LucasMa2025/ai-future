"""
NLGSM 系统配置
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    """系统配置"""
    
    # ==================== Application ====================
    APP_NAME: str = "NLGSM"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    
    # ==================== Security ====================
    SECRET_KEY: str = Field(default="nlgsm-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"
    
    # ==================== Database ====================
    DATABASE_URL: str = Field(
        default="postgresql://nlgsm:nlgsm123@localhost:5432/nlgsm_db"
    )
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_ECHO: bool = False
    
    # ==================== Redis ====================
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TOKEN_DB: int = 1  # Token 缓存使用的 DB
    REDIS_TOKEN_EXPIRE: int = 60 * 60 * 24  # Token 缓存过期时间 (秒)
    
    # ==================== Email ====================
    SMTP_HOST: str = "smtp.example.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "nlgsm@example.com"
    SMTP_FROM_NAME: str = "NLGSM System"
    SMTP_TLS: bool = True
    EMAIL_ENABLED: bool = False
    
    # ==================== WebSocket ====================
    WS_HEARTBEAT_INTERVAL: int = 30  # 心跳间隔 (秒)
    
    # ==================== Audit ====================
    AUDIT_ENABLED: bool = True
    AUDIT_LOG_RETENTION_DAYS: int = 365
    
    # ==================== NLGSM State Machine ====================
    INITIAL_STATE: str = "frozen"
    
    # ==================== Approval ====================
    MULTI_SIG_REQUIRED_CRITICAL: int = 3
    MULTI_SIG_REQUIRED_HIGH: int = 2
    APPROVAL_TIMEOUT_HOURS: int = 24
    
    # ==================== CORS ====================
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # ==================== AGA Portal ====================
    # AGA Portal 为独立部署的知识管理服务，通过 HTTP API 通信
    # 治理系统和 AGA Portal 可能部署在不同服务器
    AGA_PORTAL_ENABLED: bool = True
    AGA_PORTAL_URL: str = Field(default="http://localhost:8081")  # AGA Portal API 地址
    AGA_PORTAL_TIMEOUT: float = 30.0  # HTTP 请求超时 (秒)
    AGA_PORTAL_API_KEY: Optional[str] = None  # API 认证密钥 (可选)
    AGA_PORTAL_NAMESPACE: str = "default"  # 默认命名空间
    AGA_PORTAL_RETRY_ATTEMPTS: int = 3  # 失败重试次数
    AGA_PORTAL_RETRY_DELAY: float = 1.0  # 重试间隔 (秒)
    
    # AGA 知识内化配置
    AGA_INITIAL_LIFECYCLE: str = "probationary"  # 新知识初始状态
    AGA_ENCODING_HIDDEN_DIM: int = 4096  # 向量维度 (需与 AGA Portal 配置一致)
    AGA_ENCODING_BOTTLENECK_DIM: int = 64  # 瓶颈层维度
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()

