import asyncio
import fnmatch
import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from app.conf.app_config import CacheConfig, RedisConfig
from app.core.cache_context import get_cache_scope
from app.core.cache_metrics import (
    cache_backend_up,
    cache_eviction_ratio,
    cache_evictions_total,
    cache_expired_ratio,
    cache_hit_ratio,
    cache_requests_total,
    cache_sets_total,
)
from app.core.log import logger


@dataclass
class CacheEntry:
    value: Any
    created_at: float
    ttl: int

    @property
    def age(self) -> float:
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        return self.age > self.ttl

    def within_stale_ttl(self, stale_ttl: int) -> bool:
        return self.age <= self.ttl + stale_ttl


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    expired: int = 0
    stale_hits: int = 0
    stale_misses: int = 0
    sets: int = 0
    evictions: int = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses + self.expired

    @property
    def hit_rate(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def expired_rate(self) -> float:
        return self.expired / self.requests if self.requests else 0.0

    @property
    def eviction_rate(self) -> float:
        return self.evictions / self.sets if self.sets else 0.0


class CacheBackend(Protocol):
    async def get_entry(self, key: str) -> CacheEntry | None:
        ...

    async def set_entry(self, key: str, entry: CacheEntry, max_age: int) -> int:
        ...

    async def delete(self, key: str) -> None:
        ...

    async def clear_prefix(self, prefix: str) -> None:
        ...

    async def clear_pattern(self, pattern: str) -> None:
        ...

    async def close(self) -> None:
        ...


class MemoryCacheBackend:
    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self._store: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

    async def get_entry(self, key: str) -> CacheEntry | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            self._touch(key)
            return entry

    async def set_entry(self, key: str, entry: CacheEntry, max_age: int) -> int:
        evictions = 0
        async with self._lock:
            if key in self._store:
                self._remove(key)
            while len(self._store) >= self.maxsize:
                evictions += self._evict_oldest()
            self._store[key] = entry
            self._access_order.append(key)
        return evictions

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._remove(key)

    async def clear_prefix(self, prefix: str) -> None:
        async with self._lock:
            for key in list(self._store.keys()):
                if key.startswith(prefix):
                    self._remove(key)

    async def clear_pattern(self, pattern: str) -> None:
        async with self._lock:
            for key in list(self._store.keys()):
                if fnmatch.fnmatch(key, pattern):
                    self._remove(key)

    async def close(self) -> None:
        return None

    def _touch(self, key: str) -> None:
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _remove(self, key: str) -> None:
        self._store.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def _evict_oldest(self) -> int:
        if not self._access_order:
            return 0
        oldest = self._access_order.pop(0)
        self._store.pop(oldest, None)
        return 1


class RedisCacheBackend:
    def __init__(self, config: RedisConfig):
        from redis import asyncio as redis_async

        self.client = redis_async.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password or None,
            socket_timeout=config.socket_timeout,
            decode_responses=False,
        )

    async def ping(self) -> None:
        await self.client.ping()

    async def get_entry(self, key: str) -> CacheEntry | None:
        raw = await self.client.get(key)
        if raw is None:
            return None
        return pickle.loads(raw)

    async def set_entry(self, key: str, entry: CacheEntry, max_age: int) -> int:
        await self.client.set(key, pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL), ex=max_age)
        return 0

    async def delete(self, key: str) -> None:
        await self.client.delete(key)

    async def clear_prefix(self, prefix: str) -> None:
        async for key in self.client.scan_iter(match=f"{prefix}*"):
            await self.client.delete(key)

    async def clear_pattern(self, pattern: str) -> None:
        async for key in self.client.scan_iter(match=pattern):
            await self.client.delete(key)

    async def close(self) -> None:
        await self.client.aclose()


class ScopedCache:
    def __init__(self, name: str, ttl: int, version: str, registry: "CacheRegistry"):
        self.name = name
        self.ttl = ttl
        self.version = version
        self.registry = registry
        self.stats = CacheStats()

    async def get(self, raw_key: Any) -> Any | None:
        key = self._key(raw_key)
        entry = await self.registry.backend.get_entry(key)
        if entry is None:
            self._record_request("miss")
            return None
        if entry.is_expired:
            self._record_request("expired")
            return None
        self._record_request("hit")
        return entry.value

    async def set(self, raw_key: Any, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self.ttl
        key = self._key(raw_key)
        entry = CacheEntry(value=value, created_at=time.time(), ttl=ttl)
        evictions = await self.registry.backend.set_entry(
            key,
            entry,
            max_age=ttl + self.registry.stale_ttl_seconds,
        )
        self.stats.sets += 1
        cache_sets_total.labels(cache=self.name).inc()
        if evictions:
            self.stats.evictions += evictions
            cache_evictions_total.labels(cache=self.name).inc(evictions)
        self._update_ratios()

    async def get_stale(self, raw_key: Any) -> Any | None:
        key = self._key(raw_key)
        entry = await self.registry.backend.get_entry(key)
        if entry is not None and entry.is_expired and entry.within_stale_ttl(self.registry.stale_ttl_seconds):
            self.stats.stale_hits += 1
            cache_requests_total.labels(cache=self.name, result="stale_hit").inc()
            self._update_ratios()
            return entry.value
        self.stats.stale_misses += 1
        cache_requests_total.labels(cache=self.name, result="stale_miss").inc()
        self._update_ratios()
        return None

    async def delete(self, raw_key: Any) -> None:
        await self.registry.backend.delete(self._key(raw_key))

    async def clear(self) -> None:
        await self.registry.backend.clear_pattern(self._pattern())

    def _record_request(self, result: str) -> None:
        if result == "hit":
            self.stats.hits += 1
        elif result == "miss":
            self.stats.misses += 1
        elif result == "expired":
            self.stats.expired += 1
        cache_requests_total.labels(cache=self.name, result=result).inc()
        self._update_ratios()

    def _update_ratios(self) -> None:
        cache_hit_ratio.labels(cache=self.name).set(self.stats.hit_rate)
        cache_expired_ratio.labels(cache=self.name).set(self.stats.expired_rate)
        cache_eviction_ratio.labels(cache=self.name).set(self.stats.eviction_rate)

    def _prefix(self) -> str:
        config = self.registry.config
        return f"{config.key_prefix}:{config.env}:"

    def _pattern(self) -> str:
        config = self.registry.config
        return f"{config.key_prefix}:{config.env}:*:*:*:{self.name}:*"

    def _key(self, raw_key: Any) -> str:
        scope = get_cache_scope()
        digest = cache_key_hash(raw_key)
        config = self.registry.config
        tenant_id = _key_part(scope.tenant_id)
        user_id = _key_part(scope.user_id)
        project_id = _key_part(scope.project_id)
        return (
            f"{config.key_prefix}:{config.env}:"
            f"{tenant_id}:{user_id}:{project_id}:"
            f"{self.name}:{self.version}:{digest}"
        )


class CacheRegistry:
    def __init__(self):
        self.config = CacheConfig(
            backend="memory",
            env="dev",
            key_prefix="nl2sql",
            fail_fast=False,
            stale_ttl_seconds=3600,
            prompt_version="v1",
            schema_version="v1",
            embedding_model_version="v1",
            index_version="v1",
        )
        self.stale_ttl_seconds = self.config.stale_ttl_seconds
        self.backend: CacheBackend = MemoryCacheBackend()
        self.backend_name = "memory"
        self._create_caches()

    async def init(self, cache_config: CacheConfig, redis_config: RedisConfig) -> None:
        self.config = cache_config
        self.stale_ttl_seconds = cache_config.stale_ttl_seconds
        await self.backend.close()
        self.backend = MemoryCacheBackend()
        self.backend_name = "memory"

        backend = cache_config.backend.lower()
        if backend == "redis":
            redis_backend = RedisCacheBackend(redis_config)
            try:
                await redis_backend.ping()
                self.backend = redis_backend
                self.backend_name = "redis"
                cache_backend_up.labels(backend="redis").set(1)
                cache_backend_up.labels(backend="memory").set(0)
                logger.info("Redis cache backend initialized")
            except Exception as exc:
                cache_backend_up.labels(backend="redis").set(0)
                await redis_backend.close()
                if cache_config.fail_fast:
                    raise RuntimeError(f"Redis cache backend is unavailable: {exc}") from exc
                logger.warning(f"Redis cache backend unavailable, falling back to memory: {exc}")
                cache_backend_up.labels(backend="memory").set(1)
        else:
            cache_backend_up.labels(backend="memory").set(1)
            cache_backend_up.labels(backend="redis").set(0)

        self._create_caches()

    async def close(self) -> None:
        await self.backend.close()

    async def clear_names(self, names: list[str]) -> None:
        for name in names:
            cache = getattr(self, name, None)
            if isinstance(cache, ScopedCache):
                await cache.clear()

    async def clear_all(self) -> None:
        await self.backend.clear_prefix(f"{self.config.key_prefix}:{self.config.env}:")

    def all_stats(self) -> dict[str, dict]:
        result = {}
        for name, cache in self.__dict__.items():
            if isinstance(cache, ScopedCache):
                stats = cache.stats
                result[name] = {
                    "backend": self.backend_name,
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "expired": stats.expired,
                    "stale_hits": stats.stale_hits,
                    "stale_misses": stats.stale_misses,
                    "sets": stats.sets,
                    "evictions": stats.evictions,
                    "hit_rate": f"{stats.hit_rate:.1%}",
                    "expired_rate": f"{stats.expired_rate:.1%}",
                    "eviction_rate": f"{stats.eviction_rate:.1%}",
                }
        return result

    def _create_caches(self) -> None:
        config = self.config
        prompt_schema_version = f"prompt:{config.prompt_version}:schema:{config.schema_version}"
        self.embedding = ScopedCache("embedding", 3600, config.embedding_model_version, self)
        self.llm_expand = ScopedCache("llm_expand", 1800, prompt_schema_version, self)
        self.llm_cleanup = ScopedCache("llm_cleanup", 3600, prompt_schema_version, self)
        self.qdrant_column = ScopedCache("qdrant_column", 300, config.index_version, self)
        self.qdrant_metric = ScopedCache("qdrant_metric", 300, config.index_version, self)
        self.es_value = ScopedCache("es_value", 300, config.index_version, self)
        self.generate_sql = ScopedCache("generate_sql", 3600, prompt_schema_version, self)
        self.meta_mysql = ScopedCache("meta_mysql", 3600, config.schema_version, self)


def _key_part(value: str) -> str:
    return quote(value, safe="")


def cache_key_hash(*args: Any) -> str:
    try:
        raw = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        raw = repr(args)
    return hashlib.sha256(raw.encode()).hexdigest()
