from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.checkpoint import close_checkpointer, init_checkpointer
from app.agent.graph import setup_graph
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import (
    dw_mysql_client_manager,
    meta_mysql_client_manager,
)
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.conf.app_config import app_config
from app.core.cache_registry import caches


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_client_manager.init()
    qdrant_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()
    await caches.init(app_config.cache, app_config.redis)
    await init_checkpointer()
    setup_graph()
    try:
        yield
    finally:
        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()
        await close_checkpointer()
        await caches.close()
