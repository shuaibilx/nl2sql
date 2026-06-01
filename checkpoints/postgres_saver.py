import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.conf.app_config import CheckpointConfig, app_config
from app.core.cache_context import CacheScope
from app.core.cache_metrics import (
    checkpoint_active_sessions,
    checkpoint_backend_up,
    checkpoint_cleanup_deleted_total,
)
from app.core.log import logger
from checkpoints.base import CHECKPOINT_ALLOWED_MSGPACK_MODULES, InstrumentedCheckpointer


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class PostgresCheckpointBackend:
    backend_name = "postgres"

    def __init__(self, pool: AsyncConnectionPool, saver: Any):
        self.pool = pool
        self.raw_saver = saver
        self.checkpointer: Any = InstrumentedCheckpointer(self.backend_name, saver)

    @classmethod
    async def create(cls, config: CheckpointConfig) -> "PostgresCheckpointBackend":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        os.environ["LANGGRAPH_STRICT_MSGPACK"] = str(config.strict_msgpack).lower()
        serde = JsonPlusSerializer(
            pickle_fallback=False,
            allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES
            if config.strict_msgpack
            else True,
        )
        pool = AsyncConnectionPool(
            config.postgres_dsn,
            min_size=config.pool_min_size,
            max_size=config.pool_max_size,
            kwargs={
                "autocommit": True,
                "row_factory": dict_row,
                "prepare_threshold": 0,
            },
            open=False,
        )
        try:
            await pool.open()
            await pool.wait()
            saver = AsyncPostgresSaver(pool, serde=serde)
            if config.setup_on_start:
                await saver.setup()
                await _setup_checkpoint_session_table(pool)
            backend = cls(pool, saver)
            checkpoint_backend_up.labels(cls.backend_name).set(1)
            await backend.refresh_active_sessions()
            logger.info("LangGraph checkpointer initialized with PostgreSQL backend")
            return backend
        except Exception:
            await pool.close()
            raise

    async def touch_session(
        self,
        *,
        external_session_id: str,
        thread_id: str,
        scope: CacheScope,
        retention_days: int | None = None,
    ) -> None:
        now = datetime.now(UTC)
        expires_at = now + timedelta(
            days=retention_days or app_config.checkpoint.retention_days
        )
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO checkpoint_sessions (
                    thread_id,
                    external_session_id,
                    tenant_id,
                    user_id,
                    project_id,
                    created_at,
                    last_seen_at,
                    expires_at,
                    status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'active')
                ON CONFLICT (thread_id) DO UPDATE SET
                    external_session_id = EXCLUDED.external_session_id,
                    tenant_id = EXCLUDED.tenant_id,
                    user_id = EXCLUDED.user_id,
                    project_id = EXCLUDED.project_id,
                    last_seen_at = EXCLUDED.last_seen_at,
                    expires_at = EXCLUDED.expires_at,
                    status = 'active'
                """,
                (
                    thread_id,
                    external_session_id,
                    scope.tenant_id,
                    scope.user_id,
                    scope.project_id,
                    now,
                    now,
                    expires_at,
                ),
            )
        await self.refresh_active_sessions()

    async def refresh_active_sessions(self) -> int:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) AS count FROM checkpoint_sessions WHERE expires_at >= now()"
            )
            row = await cursor.fetchone()
        count = int(row["count"] if row else 0)
        checkpoint_active_sessions.set(count)
        return count

    async def cleanup_expired(self, limit: int | None = None) -> int:
        query = "SELECT thread_id FROM checkpoint_sessions WHERE expires_at < now()"
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT %s"
            params = (limit,)

        async with self.pool.connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

        thread_ids = [row["thread_id"] for row in rows]
        for thread_id in thread_ids:
            await self.raw_saver.adelete_thread(thread_id)

        if thread_ids:
            async with self.pool.connection() as conn:
                for thread_id in thread_ids:
                    await conn.execute(
                        "DELETE FROM checkpoint_sessions WHERE thread_id = %s",
                        (thread_id,),
                    )
            checkpoint_cleanup_deleted_total.inc(len(thread_ids))
        await self.refresh_active_sessions()
        return len(thread_ids)

    async def close(self) -> None:
        await self.pool.close()
        checkpoint_backend_up.labels(self.backend_name).set(0)


async def _setup_checkpoint_session_table(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoint_sessions (
                thread_id TEXT PRIMARY KEY,
                external_session_id TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                project_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoint_sessions_expires_at
            ON checkpoint_sessions (expires_at)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_checkpoint_sessions_scope_session
            ON checkpoint_sessions (
                tenant_id,
                user_id,
                project_id,
                external_session_id
            )
            """
        )
