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
class AppConfig:
    logging: LoggingConfig
    db_meta: DBConfig
    db_dw: DBConfig
    qdrant: QdrantConfig
    embedding: EmbeddingConfig
    es: ESConfig
    llm: LLMConfig
    llm_fallback: LLMConfig


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
