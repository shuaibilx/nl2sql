from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


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
checkpoint_backend_up = Gauge(
    "nl2sql_checkpoint_backend_up",
    "Whether the checkpoint backend is available.",
    ["backend"],
)
checkpoint_operations_total = Counter(
    "nl2sql_checkpoint_operations_total",
    "LangGraph checkpoint operation count.",
    ["backend", "operation", "result"],
)
checkpoint_operation_seconds = Histogram(
    "nl2sql_checkpoint_operation_seconds",
    "LangGraph checkpoint operation latency in seconds.",
    ["backend", "operation"],
)
checkpoint_cleanup_deleted_total = Counter(
    "nl2sql_checkpoint_cleanup_deleted_total",
    "Deleted expired checkpoint session count.",
)
checkpoint_active_sessions = Gauge(
    "nl2sql_checkpoint_active_sessions",
    "Active checkpoint sessions tracked by metadata.",
)


def render_metrics() -> bytes:
    return generate_latest()
