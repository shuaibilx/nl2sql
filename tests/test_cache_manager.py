import pytest

from app.conf.app_config import CacheConfig, RedisConfig
from app.core import cache_manager as cache_manager_module
from app.core.cache_context import CacheScope, use_cache_scope
from app.core.cache_manager import CacheRegistry, MemoryCacheBackend
from app.core.cache_metrics import render_metrics


def make_cache_config(**overrides):
    values = {
        "backend": "memory",
        "env": "test",
        "key_prefix": "nl2sql_test",
        "fail_fast": False,
        "stale_ttl_seconds": 3600,
        "semantic_enabled": False,
        "semantic_threshold": 0.94,
        "semantic_max_entries": 1024,
        "semantic_ttl_seconds": 3600,
        "ttl_embedding_seconds": 3600,
        "ttl_llm_cleanup_seconds": 3600,
        "ttl_llm_expand_seconds": 1800,
        "ttl_qdrant_column_seconds": 300,
        "ttl_qdrant_metric_seconds": 300,
        "ttl_es_value_seconds": 300,
        "ttl_generate_sql_seconds": 3600,
        "ttl_meta_mysql_seconds": 3600,
        "prompt_version": "v1",
        "schema_version": "v1",
        "embedding_model_version": "emb-v1",
        "index_version": "idx-v1",
    }
    values.update(overrides)
    return CacheConfig(**values)


def make_redis_config():
    return RedisConfig(host="localhost", port=6379, db=0, password="", socket_timeout=0.1)


@pytest.mark.asyncio
async def test_memory_cache_hit_miss_expired_stale_and_eviction():
    registry = CacheRegistry()
    await registry.init(make_cache_config(stale_ttl_seconds=60), make_redis_config())
    registry.backend = MemoryCacheBackend(maxsize=1)
    registry._create_caches()

    await registry.llm_cleanup.set("a", "value-a", ttl=30)
    assert await registry.llm_cleanup.get("a") == "value-a"
    assert await registry.llm_cleanup.get("missing") is None

    key = registry.llm_cleanup._key("a")
    entry = await registry.backend.get_entry(key)
    entry.created_at -= 31
    assert await registry.llm_cleanup.get("a") is None
    assert await registry.llm_cleanup.get_stale("a") == "value-a"

    await registry.llm_cleanup.set("b", "value-b", ttl=30)
    assert registry.llm_cleanup.stats.evictions == 1
    await registry.close()


@pytest.mark.asyncio
async def test_cache_ttl_values_come_from_config():
    registry = CacheRegistry()
    await registry.init(
        make_cache_config(
            ttl_embedding_seconds=11,
            ttl_llm_cleanup_seconds=12,
            ttl_llm_expand_seconds=13,
            ttl_qdrant_column_seconds=14,
            ttl_qdrant_metric_seconds=15,
            ttl_es_value_seconds=16,
            ttl_generate_sql_seconds=17,
            ttl_meta_mysql_seconds=18,
            semantic_ttl_seconds=19,
        ),
        make_redis_config(),
    )

    assert registry.embedding.ttl == 11
    assert registry.llm_cleanup.ttl == 12
    assert registry.llm_expand.ttl == 13
    assert registry.qdrant_column.ttl == 14
    assert registry.qdrant_metric.ttl == 15
    assert registry.es_value.ttl == 16
    assert registry.generate_sql.ttl == 17
    assert registry.meta_mysql.ttl == 18
    assert registry.semantic_generate_sql.ttl == 19

    await registry.close()


@pytest.mark.asyncio
async def test_cache_key_isolated_by_scope_and_version():
    registry = CacheRegistry()
    await registry.init(make_cache_config(), make_redis_config())

    with use_cache_scope(CacheScope(tenant_id="tenant-a", user_id="user-a", project_id="project-a")):
        await registry.llm_cleanup.set("same-query", "tenant-a-value")
        assert await registry.llm_cleanup.get("same-query") == "tenant-a-value"

    with use_cache_scope(CacheScope(tenant_id="tenant-b", user_id="user-a", project_id="project-a")):
        assert await registry.llm_cleanup.get("same-query") is None

    old_cache = registry.llm_cleanup
    registry.config.prompt_version = "v2"
    registry._create_caches()
    with use_cache_scope(CacheScope(tenant_id="tenant-a", user_id="user-a", project_id="project-a")):
        assert await registry.llm_cleanup.get("same-query") is None
        assert await old_cache.get("same-query") == "tenant-a-value"

    await registry.close()


@pytest.mark.asyncio
async def test_clear_names_only_clears_target_cache_namespace():
    registry = CacheRegistry()
    await registry.init(make_cache_config(), make_redis_config())

    await registry.meta_mysql.set("metadata", "meta-value")
    await registry.generate_sql.set("sql", "sql-value")

    await registry.clear_names(["meta_mysql"])

    assert await registry.meta_mysql.get("metadata") is None
    assert await registry.generate_sql.get("sql") == "sql-value"
    await registry.close()


@pytest.mark.asyncio
async def test_semantic_cache_matches_same_scope_context_and_threshold():
    registry = CacheRegistry()
    await registry.init(make_cache_config(semantic_enabled=True, semantic_threshold=0.9), make_redis_config())
    context = {"tables": ["sales"], "metrics": ["amount"]}

    await registry.semantic_generate_sql.set("浙江销售额", [1.0, 0.0], context, "select sum(amount)")
    hit = await registry.semantic_generate_sql.get("浙江销售总额", [0.98, 0.1], context)
    assert hit is not None
    assert hit.value == "select sum(amount)"
    assert hit.similarity >= 0.9

    assert await registry.semantic_generate_sql.get("浙江订单数", [0.0, 1.0], context) is None
    assert await registry.semantic_generate_sql.get("浙江销售总额", [0.98, 0.1], {"tables": ["orders"]}) is None

    with use_cache_scope(CacheScope(tenant_id="tenant-b", user_id="user-a", project_id="project-a")):
        assert await registry.semantic_generate_sql.get("浙江销售总额", [0.98, 0.1], context) is None

    await registry.close()


@pytest.mark.asyncio
async def test_semantic_cache_max_entries_and_namespace_clear():
    registry = CacheRegistry()
    await registry.init(
        make_cache_config(semantic_enabled=True, semantic_threshold=0.8, semantic_max_entries=1),
        make_redis_config(),
    )
    context = {"tables": ["sales"]}

    await registry.semantic_generate_sql.set("query-a", [1.0, 0.0], context, "sql-a")
    await registry.semantic_generate_sql.set("query-b", [0.0, 1.0], context, "sql-b")

    assert await registry.semantic_generate_sql.get("query-a-like", [1.0, 0.0], context) is None
    hit = await registry.semantic_generate_sql.get("query-b-like", [0.0, 1.0], context)
    assert hit is not None
    assert hit.value == "sql-b"

    await registry.clear_names(["semantic_generate_sql"])
    assert await registry.semantic_generate_sql.get("query-b-like", [0.0, 1.0], context) is None
    await registry.close()


@pytest.mark.asyncio
async def test_redis_unavailable_falls_back_or_fails_fast(monkeypatch):
    class BrokenRedisBackend:
        def __init__(self, config):
            pass

        async def ping(self):
            raise OSError("redis down")

        async def close(self):
            pass

    monkeypatch.setattr(cache_manager_module, "RedisCacheBackend", BrokenRedisBackend)

    registry = CacheRegistry()
    await registry.init(make_cache_config(backend="redis", fail_fast=False), make_redis_config())
    assert registry.backend_name == "memory"

    with pytest.raises(RuntimeError, match="Redis cache backend is unavailable"):
        await registry.init(make_cache_config(backend="redis", fail_fast=True), make_redis_config())


@pytest.mark.asyncio
async def test_cache_metrics_are_exported():
    registry = CacheRegistry()
    await registry.init(make_cache_config(), make_redis_config())

    await registry.llm_cleanup.set("query", "cleaned")
    assert await registry.llm_cleanup.get("query") == "cleaned"
    assert await registry.llm_cleanup.get("miss") is None

    metrics = render_metrics().decode()
    assert "nl2sql_cache_requests_total" in metrics
    assert "nl2sql_cache_hit_ratio" in metrics
    assert 'cache="llm_cleanup"' in metrics
    await registry.close()
