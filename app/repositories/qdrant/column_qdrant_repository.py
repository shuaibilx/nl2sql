# 字段向量索引的读写

from dataclasses import asdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.conf.app_config import app_config
from app.core.cache_manager import cache_key_hash
from app.core.cache_registry import caches
from app.core.circuit_manager import circuit_manager
from app.core.log import logger
from app.core.retry import retry_async
from app.entities.column_info import ColumnInfo


class ColumnQdrantRepository:
    collection_name: str = 'data-agent-column'

    def __init__(self, client: AsyncQdrantClient):
        self.client = client
        self._cb = circuit_manager.get("Qdrant")

    # 建仓库
    async def ensure_collection(self):
        if not await retry_async(self.client.collection_exists, self.collection_name,
                                 operation_name="Qdrant-collection_exists",
                                 circuit_breaker=self._cb):
            await retry_async(self.client.create_collection, self.collection_name,
                              vectors_config=VectorParams(size=app_config.qdrant.embedding_size,
                                                          distance=Distance.COSINE),
                              operation_name="Qdrant-create_collection",
                              circuit_breaker=self._cb)

    # 写入数据
    async def upsert(self, ids: list[str], embeddings: list[list[float]], payloads: list[ColumnInfo],
                     batch_size: int = 20):
        zipped = list(zip(ids, embeddings, payloads))
        for i in range(0, len(zipped), batch_size):
            batch = zipped[i:i + batch_size]
            batch_points = [PointStruct(id=id, vector=embedding, payload=asdict(payload)) for id, embedding, payload in
                            batch]
            await retry_async(self.client.upsert, collection_name=self.collection_name, points=batch_points,
                              operation_name="Qdrant-upsert",
                              circuit_breaker=self._cb)

    async def search(self, embedding: list[float], score_threshold: float = 0.6, limit: int = 5) -> list[ColumnInfo]:
        cache_key = cache_key_hash(tuple(embedding), score_threshold, limit)

        # 1. 查新鲜缓存
        cached = caches.qdrant_column.get(cache_key)
        if cached is not None:
            return cached

        # 2. 缓存未命中，调用外部服务（带重试+熔断）
        try:
            result = await retry_async(self.client.query_points, collection_name=self.collection_name,
                                       query=embedding, score_threshold=score_threshold, limit=limit,
                                       operation_name="Qdrant-query_points",
                                       circuit_breaker=self._cb)
            results = [ColumnInfo(**point.payload) for point in result.points]
            caches.qdrant_column.set(cache_key, results)
            return results
        except Exception as e:
            # 3. 外部服务失败，尝试返回过期缓存（Stale Cache 降级）
            stale = caches.qdrant_column.get_stale(cache_key)
            if stale is not None:
                logger.warning(f"Qdrant 查询失败，使用过期缓存: {e}")
                return stale
            # 4. 连过期缓存都没有，降级返回空
            logger.warning(f"Qdrant 查询失败且无缓存: {e}")
            return []
