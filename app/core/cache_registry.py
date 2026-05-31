"""
全局缓存注册表 — 所有缓存实例统一管理

使用方式：
    from app.core.cache_registry import caches
    caches.embedding.get("苹果")
    caches.qdrant_column.set(key, results)
    stats = caches.all_stats()  # 获取所有缓存统计
"""
from app.core.cache_manager import CacheManager


class CacheRegistry:
    """按缓存用途管理所有 CacheManager 实例"""

    def __init__(self):
        # 阶段1：核心缓存
        self.embedding = CacheManager(maxsize=1024, ttl=3600, name="embedding")
        self.llm_expand = CacheManager(maxsize=256, ttl=1800, name="llm_expand")
        self.llm_cleanup = CacheManager(maxsize=512, ttl=3600, name="llm_cleanup")
        # 阶段2：搜索缓存
        self.qdrant_column = CacheManager(maxsize=512, ttl=300, name="qdrant_column")
        self.qdrant_metric = CacheManager(maxsize=512, ttl=300, name="qdrant_metric")
        self.es_value = CacheManager(maxsize=512, ttl=300, name="es_value")
        # 阶段3：高级缓存
        self.generate_sql = CacheManager(maxsize=256, ttl=3600, name="generate_sql")
        self.meta_mysql = CacheManager(maxsize=1024, ttl=3600, name="meta_mysql")

    def all_stats(self) -> dict[str, dict]:
        """返回所有缓存的统计信息"""
        result = {}
        for name, cache in self.__dict__.items():
            if isinstance(cache, CacheManager):
                result[name] = {
                    "size": cache.size,
                    "maxsize": cache.maxsize,
                    "hits": cache.stats.hits,
                    "misses": cache.stats.misses,
                    "hit_rate": f"{cache.stats.hit_rate:.1%}",
                    "expired": cache.stats.expired,
                    "evictions": cache.stats.evictions,
                }
        return result

    def clear_all(self):
        """清空所有缓存"""
        for cache in self.__dict__.values():
            if isinstance(cache, CacheManager):
                cache.clear()


# 全局单例
caches = CacheRegistry()
