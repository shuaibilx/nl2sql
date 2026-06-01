from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.conf.app_config import CheckpointConfig
from app.core.cache_metrics import checkpoint_backend_up
from app.core.log import logger
from checkpoints.base import InstrumentedCheckpointer


class SQLiteCheckpointBackend:
    backend_name = "sqlite"

    def __init__(self, conn: aiosqlite.Connection, saver: AsyncSqliteSaver):
        self.conn = conn
        self.raw_saver = saver
        self.checkpointer: Any = InstrumentedCheckpointer(self.backend_name, saver)

    @classmethod
    async def create(cls, config: CheckpointConfig) -> "SQLiteCheckpointBackend":
        sqlite_path = Path(config.sqlite_path)
        if not sqlite_path.is_absolute():
            sqlite_path = Path(__file__).parents[1] / sqlite_path
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(sqlite_path)
        saver = AsyncSqliteSaver(conn)
        if config.setup_on_start:
            await saver.setup()

        backend = cls(conn, saver)
        checkpoint_backend_up.labels(cls.backend_name).set(1)
        logger.info("LangGraph checkpointer initialized with SQLite backend")
        return backend

    async def close(self) -> None:
        await self.conn.close()
        checkpoint_backend_up.labels(self.backend_name).set(0)
