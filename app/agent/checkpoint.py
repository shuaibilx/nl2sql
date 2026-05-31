from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


checkpointer: AsyncSqliteSaver | None = None
checkpoint_conn: aiosqlite.Connection | None = None


async def init_checkpointer() -> None:
    """Create the SQLite checkpointer used by LangGraph."""
    global checkpointer, checkpoint_conn

    if checkpoint_conn is not None:
        await close_checkpointer()

    checkpoint_dir = Path(__file__).parents[2] / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    db_path = checkpoint_dir / "checkpoint.db"

    checkpoint_conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(checkpoint_conn)
    await checkpointer.setup()


async def close_checkpointer() -> None:
    """Close the SQLite checkpoint connection during application shutdown."""
    global checkpointer, checkpoint_conn

    if checkpoint_conn is not None:
        await checkpoint_conn.close()
    checkpoint_conn = None
    checkpointer = None
