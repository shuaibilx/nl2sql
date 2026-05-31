"""
[改进] 通用 TTL-LRU 缓存工具 — 进程内缓存，零外部依赖

适用场景：
  1. Embedding 缓存：相同关键词跨查询重复出现，省 HuggingFace API 调用
  2. LLM 结果缓存：相同查询的 expand_keywords 结果不变，省 LLM 调用
  3. 向量/全文搜索缓存：相同关键词的 Qdrant/ES 搜索结果不变

使用示例：
    from app.core.cache import ttl_cache

    @ttl_cache(maxsize=512, ttl=3600)
    async def expensive_call(key: str) -> list:
        ...
"""
import functools
import time
from typing import Callable


def ttl_cache(maxsize: int = 256, ttl: int = 3600):
    """[改进] 异步 TTL-LRU 缓存装饰器

    Args:
        maxsize: 最大缓存条目数（LRU 淘汰）
        ttl: 缓存过期时间（秒），默认 1 小时
    """
    def decorator(func: Callable):
        cache: dict[str, tuple[float, any]] = {}
        access_order: list[str] = []  # LRU 访问顺序

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # 构造缓存键：将所有参数序列化为字符串
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()

            # 命中缓存且未过期
            if key in cache:
                ts, result = cache[key]
                if now - ts < ttl:
                    # 更新 LRU 顺序
                    if key in access_order:
                        access_order.remove(key)
                    access_order.append(key)
                    return result
                else:
                    # 过期，删除
                    del cache[key]
                    access_order.remove(key)

            # 缓存未命中，调用原函数
            result = await func(*args, **kwargs)

            # 写入缓存（LRU 淘汰）
            while len(cache) >= maxsize and access_order:
                oldest_key = access_order.pop(0)
                cache.pop(oldest_key, None)

            cache[key] = (now, result)
            access_order.append(key)
            return result

        wrapper.cache_clear = lambda: (cache.clear(), access_order.clear())
        return wrapper
    return decorator
