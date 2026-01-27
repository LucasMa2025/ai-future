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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()


settings = get_settings()

