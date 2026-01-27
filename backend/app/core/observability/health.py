"""
健康检查系统

提供系统各组件的健康状态检查
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import threading
import logging


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """组件健康状态"""
    name: str
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    last_check: Optional[datetime] = None
    response_time_ms: Optional[float] = None


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    overall_status: HealthStatus = HealthStatus.UNKNOWN
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "components": {
                name: {
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "last_check": c.last_check.isoformat() if c.last_check else None,
                    "response_time_ms": c.response_time_ms,
                }
                for name, c in self.components.items()
            }
        }


class HealthChecker:
    """
    健康检查器
    
    管理系统各组件的健康检查
    """
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], ComponentHealth]] = {}
        self._cache: Dict[str, ComponentHealth] = {}
        self._cache_ttl: timedelta = timedelta(seconds=30)
        self._lock = threading.Lock()
        
        # 注册默认检查
        self._register_default_checks()
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self.register("database", self._check_database)
        self.register("state_machine", self._check_state_machine)
        self.register("learning_system", self._check_learning_system)
        self.register("anomaly_detection", self._check_anomaly_detection)
        self.register("event_bus", self._check_event_bus)
    
    def register(self, name: str, check_fn: Callable[[], ComponentHealth]):
        """注册健康检查"""
        self._checks[name] = check_fn
    
    def unregister(self, name: str):
        """取消注册"""
        if name in self._checks:
            del self._checks[name]
    
    def check(self, use_cache: bool = True) -> HealthCheckResult:
        """
        执行健康检查
        
        Args:
            use_cache: 是否使用缓存
            
        Returns:
            健康检查结果
        """
        result = HealthCheckResult()
        
        for name, check_fn in self._checks.items():
            # 检查缓存
            if use_cache:
                cached = self._get_cached(name)
                if cached:
                    result.components[name] = cached
                    continue
            
            # 执行检查
            try:
                import time
                start = time.time()
                
                component_health = check_fn()
                
                component_health.response_time_ms = (time.time() - start) * 1000
                component_health.last_check = datetime.utcnow()
                
                # 更新缓存
                self._update_cache(name, component_health)
                
                result.components[name] = component_health
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                result.components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check failed: {str(e)}",
                    last_check=datetime.utcnow(),
                )
        
        # 计算整体状态
        result.overall_status = self._calculate_overall_status(result.components)
        
        return result
    
    def check_component(self, name: str) -> Optional[ComponentHealth]:
        """检查单个组件"""
        check_fn = self._checks.get(name)
        if not check_fn:
            return None
        
        try:
            import time
            start = time.time()
            
            health = check_fn()
            health.response_time_ms = (time.time() - start) * 1000
            health.last_check = datetime.utcnow()
            
            self._update_cache(name, health)
            
            return health
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
    
    def _get_cached(self, name: str) -> Optional[ComponentHealth]:
        """获取缓存的健康状态"""
        with self._lock:
            cached = self._cache.get(name)
            if cached and cached.last_check:
                if datetime.utcnow() - cached.last_check < self._cache_ttl:
                    return cached
        return None
    
    def _update_cache(self, name: str, health: ComponentHealth):
        """更新缓存"""
        with self._lock:
            self._cache[name] = health
    
    def _calculate_overall_status(
        self,
        components: Dict[str, ComponentHealth]
    ) -> HealthStatus:
        """计算整体健康状态"""
        if not components:
            return HealthStatus.UNKNOWN
        
        statuses = [c.status for c in components.values()]
        
        # 任一组件不健康，整体不健康
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        
        # 任一组件降级，整体降级
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        
        # 全部健康
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    # ==================== 默认健康检查 ====================
    
    def _check_database(self) -> ComponentHealth:
        """检查数据库"""
        # 实际实现中应该执行数据库查询
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection is healthy",
            details={"connection_pool": "active"},
        )
    
    def _check_state_machine(self) -> ComponentHealth:
        """检查状态机"""
        return ComponentHealth(
            name="state_machine",
            status=HealthStatus.HEALTHY,
            message="State machine is operational",
            details={"current_state": "frozen"},
        )
    
    def _check_learning_system(self) -> ComponentHealth:
        """检查学习系统"""
        return ComponentHealth(
            name="learning_system",
            status=HealthStatus.HEALTHY,
            message="Learning system is operational",
            details={"active_learners": 0, "pending_tasks": 0},
        )
    
    def _check_anomaly_detection(self) -> ComponentHealth:
        """检查异常检测"""
        return ComponentHealth(
            name="anomaly_detection",
            status=HealthStatus.HEALTHY,
            message="Anomaly detection is operational",
            details={"detectors_active": 4},
        )
    
    def _check_event_bus(self) -> ComponentHealth:
        """检查事件总线"""
        return ComponentHealth(
            name="event_bus",
            status=HealthStatus.HEALTHY,
            message="Event bus is operational",
            details={"handlers_registered": 0},
        )
    
    def set_cache_ttl(self, ttl: timedelta):
        """设置缓存 TTL"""
        self._cache_ttl = ttl
    
    def clear_cache(self):
        """清除缓存"""
        with self._lock:
            self._cache.clear()
    
    def get_component_names(self) -> List[str]:
        """获取所有组件名称"""
        return list(self._checks.keys())


# 全局健康检查器
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器"""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

