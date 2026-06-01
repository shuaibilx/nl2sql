import argparse
import asyncio

from app.conf.app_config import app_config
from app.core.log import logger
from checkpoints.manager import cleanup_expired_checkpoints, close_checkpointer, init_checkpointer


async def main(limit: int | None) -> None:
    await init_checkpointer(app_config.checkpoint)
    try:
        deleted = await cleanup_expired_checkpoints(limit=limit)
        logger.info(f"Deleted {deleted} expired checkpoint sessions")
    finally:
        await close_checkpointer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean expired LangGraph checkpoints.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(main(args.limit))
