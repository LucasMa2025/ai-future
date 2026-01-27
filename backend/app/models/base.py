"""
基础模型定义
"""
from datetime import datetime
from typing import Any
from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import DeclarativeBase, declared_attr


class Base(DeclarativeBase):
    """SQLAlchemy 声明式基类"""
    id: Any
    __name__: str
    
    @declared_attr.directive
    def __tablename__(cls) -> str:
        """自动生成表名（类名小写）"""
        return cls.__name__.lower()


class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="创建时间"
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="更新时间"
    )

