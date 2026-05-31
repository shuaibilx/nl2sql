"""
异步熔断器 — 三状态机：closed → open → half_open → closed

状态转换：
  closed  --[连续失败 N 次]--> open
  open    --[冷却 T 秒后]--> half_open
  half_open --[成功]--> closed
  half_open --[失败]--> open
"""
import asyncio
import time
from dataclasses import dataclass
from typing import Literal

from app.core.log import logger


class CircuitOpenError(Exception):
    """熔断器打开时抛出，调用方应捕获此异常走降级逻辑"""
    pass


@dataclass
class CircuitState:
    status: Literal["closed", "open", "half_open"] = "closed"
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 30.0, half_open_max: int = 1):
        """
        Args:
            name: 服务标识（如 "Qdrant", "ES", "Embedding", "LLM"）
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断冷却秒数，到期后进入 half_open
            half_open_max: half_open 状态允许的连续成功次数，达标后恢复 closed
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max
        self._state = CircuitState()
        self._lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        """快速检查熔断器是否处于 open 状态（不加锁，仅用于快速路径判断）"""
        if self._state.status == "open":
            if time.time() - self._state.last_failure_time > self.recovery_timeout:
                return False  # 冷却期已过，允许尝试
            return True
        return False

    @property
    def state(self) -> dict:
        return {
            "name": self.name,
            "status": self._state.status,
            "failure_count": self._state.failure_count,
            "success_count": self._state.success_count,
        }

    async def call(self, func, *args, **kwargs):
        """带熔断保护的调用入口"""
        async with self._lock:
            if self._state.status == "open":
                elapsed = time.time() - self._state.last_failure_time
                if elapsed > self.recovery_timeout:
                    self._state.status = "half_open"
                    self._state.success_count = 0
                    logger.info(f"[{self.name}] 熔断器 冷却{elapsed:.0f}s后，进入半开状态")
                else:
                    remaining = self.recovery_timeout - elapsed
                    raise CircuitOpenError(
                        f"[{self.name}] 熔断中，{remaining:.0f}s后重试"
                    )

        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except CircuitOpenError:
            raise  # 熔断异常不计入失败
        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self):
        async with self._lock:
            if self._state.status == "half_open":
                self._state.success_count += 1
                if self._state.success_count >= self.half_open_max:
                    self._state.status = "closed"
                    self._state.failure_count = 0
                    logger.info(f"[{self.name}] 熔断器 恢复正常 (closed)")
            else:
                self._state.failure_count = 0  # closed 状态下重置连续失败计数

    async def _on_failure(self):
        async with self._lock:
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()
            if self._state.status == "half_open":
                self._state.status = "open"
                logger.warning(f"[{self.name}] 半开探测失败，重新熔断 (open)")
            elif self._state.failure_count >= self.failure_threshold:
                self._state.status = "open"
                logger.warning(
                    f"[{self.name}] 连续失败{self._state.failure_count}次，熔断 (open)，"
                    f"{self.recovery_timeout}s后恢复"
                )

    def reset(self):
        """手动重置熔断器"""
        self._state = CircuitState()
        logger.info(f"[{self.name}] 熔断器 手动重置 (closed)")
