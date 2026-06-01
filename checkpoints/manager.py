from typing import Any

from app.conf.app_config import CheckpointConfig, app_config
from app.core.cache_context import CacheScope
from app.core.cache_metrics import checkpoint_active_sessions, checkpoint_backend_up
from checkpoints.postgres_saver import PostgresCheckpointBackend
from checkpoints.sqlite_saver import SQLiteCheckpointBackend


checkpointer: Any | None = None
_backend: PostgresCheckpointBackend | SQLiteCheckpointBackend | None = None
_backend_name: str | None = None


async def init_checkpointer(config: CheckpointConfig | None = None) -> None:
    """Create the LangGraph checkpointer for the configured backend."""
    global checkpointer, _backend, _backend_name

    config = config or app_config.checkpoint
    backend_name = config.backend.lower()
    await close_checkpointer()

    try:
        if backend_name == "postgres":
            backend = await PostgresCheckpointBackend.create(config)
        elif backend_name == "sqlite":
            backend = await SQLiteCheckpointBackend.create(config)
        else:
            raise RuntimeError(f"Unsupported checkpoint backend: {config.backend}")
    except Exception:
        checkpoint_backend_up.labels(backend_name).set(0)
        await close_checkpointer()
        raise

    _backend = backend
    _backend_name = backend_name
    checkpointer = backend.checkpointer


async def touch_checkpoint_session(
    *,
    external_session_id: str,
    thread_id: str,
    scope: CacheScope,
    retention_days: int | None = None,
) -> None:
    if isinstance(_backend, PostgresCheckpointBackend):
        await _backend.touch_session(
            external_session_id=external_session_id,
            thread_id=thread_id,
            scope=scope,
            retention_days=retention_days,
        )


async def refresh_active_checkpoint_sessions() -> int:
    if isinstance(_backend, PostgresCheckpointBackend):
        return await _backend.refresh_active_sessions()
    checkpoint_active_sessions.set(0)
    return 0


async def cleanup_expired_checkpoints(limit: int | None = None) -> int:
    if isinstance(_backend, PostgresCheckpointBackend):
        return await _backend.cleanup_expired(limit=limit)
    return 0


async def close_checkpointer() -> None:
    """Close checkpoint resources during application shutdown."""
    global checkpointer, _backend, _backend_name

    if _backend is not None:
        await _backend.close()
    elif _backend_name:
        checkpoint_backend_up.labels(_backend_name).set(0)

    checkpointer = None
    _backend = None
    _backend_name = None
