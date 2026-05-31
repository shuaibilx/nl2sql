"""
统一缓存管理器 — 替代分散的 @ttl_cache 和 dict 缓存

所有缓存统一走 cache_manager.get() / cache_manager.set()
未来迁移到 Redis 只需替换 Backend，调用方不变

使用方式：
    from app.core.cache_manager import CacheManager, cache_key_hash
    cache = CacheManager(maxsize=1024, ttl=3600, name="embedding")
    cache.set("苹果", embedding_vector)
    result = cache.get("苹果")
"""
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from app.core.log import logger


@dataclass
class CacheEntry:
    value: Any
    created_at: float
    ttl: int

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    @property
    def age(self) -> float:
        return time.time() - self.created_at


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    expired: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class CacheManager:
    """
    进程内 LRU + TTL 缓存管理器

    Args:
        maxsize: 最大缓存条目数（LRU 淘汰）
        ttl: 默认缓存过期时间（秒）
        name: 缓存名称（用于日志和监控）
    """

    def __init__(self, maxsize: int = 256, ttl: int = 3600, name: str = ""):
        self.name = name
        self.maxsize = maxsize
        self.ttl = ttl
        self._store: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self.stats = CacheStats()

    def get(self, key: str) -> Any | None:
        """获取缓存，返回 None 表示未命中或已过期"""
        entry = self._store.get(key)
        if entry is None:
            self.stats.misses += 1
            self._log_stats()
            return None
        if entry.is_expired:
            self.stats.expired += 1
            self._log_stats()
            return None  # 过期不返回，但保留在 _store 供 get_stale() 使用
        # 命中，更新 LRU 顺序
        self.stats.hits += 1
        self._touch(key)
        self._log_stats()
        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None):
        """写入缓存"""
        if key in self._store:
            self._remove(key)
        while len(self._store) >= self.maxsize:
            self._evict_oldest()
        self._store[key] = CacheEntry(
            value=value, created_at=time.time(), ttl=ttl or self.ttl
        )
        self._access_order.append(key)

    def delete(self, key: str):
        """删除指定缓存"""
        self._remove(key)

    def clear(self):
        """清空所有缓存"""
        self._store.clear()
        self._access_order.clear()

    def get_stale(self, key: str) -> Any | None:
        """获取过期但仍存在的缓存（用于 Stale Cache 降级）"""
        entry = self._store.get(key)
        if entry is not None:
            return entry.value
        return None

    def _touch(self, key: str):
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove(self, key: str):
        self._store.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_oldest(self):
        if self._access_order:
            oldest = self._access_order.pop(0)
            self._store.pop(oldest, None)
            self.stats.evictions += 1

    def _log_stats(self):
        total = self.stats.hits + self.stats.misses
        if total > 0 and total % 100 == 0:
            logger.info(f"[Cache:{self.name}] hit_rate={self.stats.hit_rate:.1%} "
                        f"size={self.size} hits={self.stats.hits} misses={self.stats.misses}")

    @property
    def size(self) -> int:
        return len(self._store)


def cache_key_hash(*args) -> str:
    """将任意参数哈希为短字符串作为缓存键"""
    raw = str(args)
    return hashlib.md5(raw.encode()).hexdigest()
