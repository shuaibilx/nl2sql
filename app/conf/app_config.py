from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf


@dataclass
class File:
    enable: bool
    level: str
    path: str
    rotation: str
    retention: str


@dataclass
class Console:
    enable: bool
    level: str


@dataclass
class LoggingConfig:
    file: File
    console: Console


@dataclass
class DBConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


@dataclass
class QdrantConfig:
    host: str
    port: int
    embedding_size: int


@dataclass
class EmbeddingConfig:
    host: str
    port: int
    model: str


@dataclass
class ESConfig:
    host: str
    port: int
    index_name: str


@dataclass
class LLMConfig:
    model_name: str
    api_key: str
    base_url: str


@dataclass
class CacheConfig:
    backend: str
    env: str
    key_prefix: str
    fail_fast: bool
    stale_ttl_seconds: int
    semantic_enabled: bool
    semantic_threshold: float
    semantic_max_entries: int
    semantic_ttl_seconds: int
    ttl_embedding_seconds: int
    ttl_llm_cleanup_seconds: int
    ttl_llm_expand_seconds: int
    ttl_qdrant_column_seconds: int
    ttl_qdrant_metric_seconds: int
    ttl_es_value_seconds: int
    ttl_generate_sql_seconds: int
    ttl_meta_mysql_seconds: int
    prompt_version: str
    schema_version: str
    embedding_model_version: str
    index_version: str


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    password: str
    socket_timeout: float


@dataclass
class CheckpointConfig:
    backend: str
    postgres_dsn: str
    pool_min_size: int
    pool_max_size: int
    setup_on_start: bool
    retention_days: int
    sqlite_path: str
    strict_msgpack: bool


@dataclass
class RecallConfig:
    column_score_threshold: float
    metric_score_threshold: float
    value_score_threshold: float


@dataclass
class MonitoringConfig:
    prometheus_enabled: bool


@dataclass
class AppConfig:
    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig
    llm_fallback: LLMConfig
    cache: CacheConfig
    redis: RedisConfig
    checkpoint: CheckpointConfig
    recall: RecallConfig
    monitoring: MonitoringConfig


project_root = Path(__file__).parents[2]
load_dotenv(project_root / ".env")

config_file = project_root / "conf" / "app_config.yaml"
context = OmegaConf.load(config_file)
schema = OmegaConf.structured(AppConfig)
merged_config = OmegaConf.merge(schema, context)
OmegaConf.resolve(merged_config)
app_config: AppConfig = OmegaConf.to_object(merged_config)


def _validate_config(config: AppConfig) -> None:
    missing = []
    if not config.db_meta.password:
        missing.append("DB_META_PASSWORD")
    if not config.db_dw.password:
        missing.append("DB_DW_PASSWORD")

    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing required configuration: {names}. Set them in .env or environment variables.")


_validate_config(app_config)


if __name__ == "__main__":
    print(app_config.es.host)
