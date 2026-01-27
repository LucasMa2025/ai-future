"""
Redis 连接管理
"""
from typing import Optional
import redis.asyncio as redis
from redis.asyncio import Redis

from ..config import settings


class RedisManager:
    """
    Redis 连接管理器
    
    管理 Redis 连接池，支持多个数据库
    """
    
    def __init__(self):
        self._pools: dict[int, Redis] = {}
        self._default_pool: Optional[Redis] = None
    
    async def init(self):
        """初始化默认连接池"""
        self._default_pool = await self._create_pool(0)
        # Token 缓存使用独立的 DB
        self._pools[settings.REDIS_TOKEN_DB] = await self._create_pool(
            settings.REDIS_TOKEN_DB
        )
    
    async def _create_pool(self, db: int) -> Redis:
        """创建连接池"""
        # 解析 URL
        url = settings.REDIS_URL
        if url.endswith(f"/{db}"):
            pass
        elif "/" in url.split(":")[-1]:
            # 替换 DB 编号
            url = "/".join(url.rsplit("/", 1)[:-1]) + f"/{db}"
        else:
            url = f"{url}/{db}"
        
        return redis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=20,
        )
    
    async def close(self):
        """关闭所有连接"""
        if self._default_pool:
            await self._default_pool.close()
        for pool in self._pools.values():
            await pool.close()
    
    def get_client(self, db: Optional[int] = None) -> Redis:
        """获取 Redis 客户端"""
        if db is not None and db in self._pools:
            return self._pools[db]
        return self._default_pool
    
    @property
    def token_client(self) -> Redis:
        """获取 Token 缓存客户端"""
        return self._pools.get(settings.REDIS_TOKEN_DB, self._default_pool)


# 全局 Redis 管理器
redis_manager = RedisManager()


async def get_redis() -> Redis:
    """
    获取默认 Redis 客户端
    
    用于 FastAPI 依赖注入
    """
    return redis_manager.get_client()


async def get_token_redis() -> Redis:
    """
    获取 Token Redis 客户端
    
    用于 JWT Token 缓存
    """
    return redis_manager.token_client

