from dataclasses import asdict

from elasticsearch import AsyncElasticsearch

from app.core.cache_registry import caches
from app.core.circuit_manager import circuit_manager
from app.core.log import logger
from app.core.retry import retry_async
from app.entities.value_info import ValueInfo

# 数据访问层：字段取值全文索引的读写


class ValueESRepository:
    index_name = 'data-agent-value'
    index_mappings = {
        "dynamic": False,
        "properties": {
            "id": {"type": "keyword"},
            "value": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_max_word"},
            "column_id": {"type": "keyword"}
        }
    }

    def __init__(self, client: AsyncElasticsearch):
        self.client = client
        self._cb = circuit_manager.get("ES")

    async def ensure_index(self):
        if not await retry_async(self.client.indices.exists, index=self.index_name,
                                 operation_name="ES-indices.exists",
                                 circuit_breaker=self._cb):
            await retry_async(self.client.indices.create, index=self.index_name, mappings=self.index_mappings,
                              operation_name="ES-indices.create",
                              circuit_breaker=self._cb)

    async def index(self, value_infos: list[ValueInfo], batch_size=20):
        for i in range(0, len(value_infos), batch_size):
            batch = value_infos[i:i + batch_size]
            operations = []
            for value_info in batch:
                operations.append({"index": {"_index": self.index_name, "_id": value_info.id}})
                operations.append(asdict(value_info))
            await retry_async(self.client.bulk, operations=operations,
                              operation_name="ES-bulk",
                              circuit_breaker=self._cb)

    async def search(self, keyword: str, score_threshold: float = 0.6, limit: int = 5) -> list[ValueInfo]:
        cache_key = f"{keyword}:{score_threshold}:{limit}"

        cached = caches.es_value.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = await retry_async(self.client.search, index=self.index_name,
                                       query={"match": {"value": keyword}},
                                       min_score=score_threshold, size=limit,
                                       operation_name="ES-search",
                                       circuit_breaker=self._cb)
            results = [ValueInfo(**hit['_source']) for hit in result['hits']['hits']]
            caches.es_value.set(cache_key, results)
            return results
        except Exception as e:
            stale = caches.es_value.get_stale(cache_key)
            if stale is not None:
                logger.warning(f"ES 查询失败，使用过期缓存: {e}")
                return stale
            logger.warning(f"ES 查询失败且无缓存: {e}")
            return []
