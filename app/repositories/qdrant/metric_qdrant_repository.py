from dataclasses import asdict

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from app.conf.app_config import app_config
from app.core.cache_manager import cache_key_hash
from app.core.cache_registry import caches
from app.core.circuit_manager import circuit_manager
from app.core.log import logger
from app.core.retry import retry_async
from app.entities.metric_info import MetricInfo

# 指标向量索引的读写


class MetricQdrantRepository:
    collection_name = 'data-agent-metric'

    def __init__(self, client: AsyncQdrantClient):
        self.client = client
        self._cb = circuit_manager.get("Qdrant")

    async def ensure_collection(self):
        if not await retry_async(self.client.collection_exists, self.collection_name,
                                 operation_name="Qdrant-collection_exists",
                                 circuit_breaker=self._cb):
            await retry_async(self.client.create_collection, self.collection_name,
                              vectors_config=VectorParams(size=app_config.qdrant.embedding_size,
                                                          distance=Distance.COSINE),
                              operation_name="Qdrant-create_collection",
                              circuit_breaker=self._cb)

    async def upsert(self, ids: list[str], embeddings: list[list[float]], payloads: list[MetricInfo],
                     batch_size: int = 20):
        zipped = list(zip(ids, embeddings, payloads))
        for i in range(0, len(zipped), batch_size):
            batch = zipped[i:i + batch_size]
            batch_points = [PointStruct(id=id, vector=embedding, payload=asdict(payload)) for id, embedding, payload in
                            batch]
            await retry_async(self.client.upsert, collection_name=self.collection_name, points=batch_points,
                              operation_name="Qdrant-upsert",
                              circuit_breaker=self._cb)

    async def search(self, embedding: list[float], score_threshold: float = 0.6, limit: int = 5) -> list[MetricInfo]:
        cache_key = cache_key_hash(tuple(embedding), score_threshold, limit)

        cached = await caches.qdrant_metric.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = await retry_async(self.client.query_points, collection_name=self.collection_name,
                                       query=embedding, score_threshold=score_threshold, limit=limit,
                                       operation_name="Qdrant-query_points",
                                       circuit_breaker=self._cb)
            results = [MetricInfo(**point.payload) for point in result.points]
            await caches.qdrant_metric.set(cache_key, results)
            return results
        except Exception as e:
            stale = await caches.qdrant_metric.get_stale(cache_key)
            if stale is not None:
                logger.warning(f"Qdrant 查询失败，使用过期缓存: {e}")
                return stale
            logger.warning(f"Qdrant 查询失败且无缓存: {e}")
            return []
