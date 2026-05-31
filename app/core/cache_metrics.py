from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest


cache_requests_total = Counter(
    "nl2sql_cache_requests_total",
    "Cache request count by cache name and result.",
    ["cache", "result"],
)
cache_sets_total = Counter(
    "nl2sql_cache_sets_total",
    "Cache write count by cache name.",
    ["cache"],
)
cache_evictions_total = Counter(
    "nl2sql_cache_evictions_total",
    "Cache eviction count by cache name.",
    ["cache"],
)
cache_hit_ratio = Gauge(
    "nl2sql_cache_hit_ratio",
    "Cache hit ratio by cache name.",
    ["cache"],
)
cache_expired_ratio = Gauge(
    "nl2sql_cache_expired_ratio",
    "Cache expired ratio by cache name.",
    ["cache"],
)
cache_eviction_ratio = Gauge(
    "nl2sql_cache_eviction_ratio",
    "Cache eviction ratio by cache name.",
    ["cache"],
)
cache_backend_up = Gauge(
    "nl2sql_cache_backend_up",
    "Whether a cache backend is available.",
    ["backend"],
)


def render_metrics() -> bytes:
    return generate_latest()
