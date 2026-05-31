"""
全局熔断器注册表 — 按服务名管理 CircuitBreaker 实例

使用方式：
    from app.core.circuit_manager import circuit_manager
    cb = circuit_manager.get("Qdrant")
    result = await cb.call(some_async_func, arg1, arg2)
"""
from app.core.circuit_breaker import CircuitBreaker


class CircuitManager:
    """按服务名管理熔断器实例，同名服务共享同一个熔断器"""

    # 各服务的熔断器默认配置
    _defaults = {
        "Qdrant": {"failure_threshold": 5, "recovery_timeout": 30},
        "ES": {"failure_threshold": 5, "recovery_timeout": 30},
        "Embedding": {"failure_threshold": 5, "recovery_timeout": 30},
        "LLM-primary": {"failure_threshold": 3, "recovery_timeout": 60},
        "LLM-fallback": {"failure_threshold": 3, "recovery_timeout": 60},
    }

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str, **kwargs) -> CircuitBreaker:
        """获取或创建指定服务的熔断器。同名返回同一实例。"""
        if name not in self._breakers:
            defaults = self._defaults.get(name, {})
            defaults.update(kwargs)
            self._breakers[name] = CircuitBreaker(name, **defaults)
        return self._breakers[name]

    def status(self) -> dict[str, dict]:
        """返回所有熔断器的状态摘要"""
        return {name: cb.state for name, cb in self._breakers.items()}

    def reset_all(self):
        """手动重置所有熔断器"""
        for cb in self._breakers.values():
            cb.reset()


# 全局单例
circuit_manager = CircuitManager()
