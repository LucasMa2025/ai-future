"""
通用 Schema
"""
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """分页参数"""
    skip: int = Field(default=0, ge=0, description="跳过记录数")
    limit: int = Field(default=50, ge=1, le=200, description="返回记录数")


class MessageResponse(BaseModel):
    """消息响应"""
    message: str
    success: bool = True
    data: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None


class PaginatedResponse(BaseModel):
    """分页响应基类"""
    total: int
    skip: int
    limit: int
    
    class Config:
        from_attributes = True

