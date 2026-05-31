# [改进] 统一召回节点 — 通过 recall_type 分派到不同的召回逻辑
# 配合 LangGraph Send API 实现 map-reduce：send_to_recalls() 动态派发 3 个并行分支

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.cache_registry import caches
from app.core.circuit_manager import circuit_manager
from app.core.log import logger
from app.core.retry import retry_async
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo


# [改进] Embedding 缓存：相同关键词跨查询重复出现（如"销售额"、"浙江"），
# 缓存后直接返回，省去 HuggingFace API 调用。maxsize=1024 覆盖常见关键词集，TTL=1小时
async def cached_embed_query(embedding_client, keyword: str) -> list[float]:
    cached = caches.embedding.get(keyword)
    if cached is not None:
        return cached
    cb = circuit_manager.get("Embedding")
    result = await retry_async(embedding_client.aembed_query, keyword,
                               operation_name="Embedding-aembed_query",
                               circuit_breaker=cb)
    caches.embedding.set(keyword, result)
    return result


async def recall_column(embedding_client, column_qdrant_repository, keywords):
    """[改进] 字段召回：embedding + Qdrant 向量搜索"""
    retrieved_map: dict[str, ColumnInfo] = {}
    for keyword in keywords:
        embedding = await cached_embed_query(embedding_client, keyword)
        payloads: list[ColumnInfo] = await column_qdrant_repository.search(embedding)
        for payload in payloads:
            if payload.id not in retrieved_map:
                retrieved_map[payload.id] = payload
    return list(retrieved_map.values())


async def recall_values(value_es_repository, keywords):
    """[改进] 取值召回：ES 全文搜索"""
    retrieved_map: dict[str, ValueInfo] = {}
    for keyword in keywords:
        values: list[ValueInfo] = await value_es_repository.search(keyword)
        for value in values:
            if value.id not in retrieved_map:
                retrieved_map[value.id] = value
    return list(retrieved_map.values())


async def recall_metrics(embedding_client, metric_qdrant_repository, keywords):
    """[改进] 指标召回：embedding + Qdrant 向量搜索"""
    retrieved_map: dict[str, MetricInfo] = {}
    for keyword in keywords:
        embedding = await cached_embed_query(embedding_client, keyword)
        payloads: list[MetricInfo] = await metric_qdrant_repository.search(embedding)
        for payload in payloads:
            if payload.id not in retrieved_map:
                retrieved_map[payload.id] = payload
    return list(retrieved_map.values())


# [改进] 召回类型 → (显示名称, 处理函数, state keywords 字段名)
RECALL_REGISTRY = {
    "column": ("字段", recall_column, "column_keywords"),
    "value":  ("取值", recall_values,  "value_keywords"),
    "metric": ("指标", recall_metrics, "metric_keywords"),
}


async def recall_node(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    """[改进] 统一召回节点 — 由 Send API 动态派发，每个分支只处理一种召回类型"""
    recall_type = state["recall_type"]
    label, func, keywords_field = RECALL_REGISTRY[recall_type]
    keywords = state.get(keywords_field, [])

    writer = runtime.stream_writer
    writer({"type": "progress", "step": f"召回{label}", "status": "running"})

    try:
        embedding_client = runtime.context["embedding_client"]
        column_qdrant_repository = runtime.context["column_qdrant_repository"]
        metric_qdrant_repository = runtime.context["metric_qdrant_repository"]
        value_es_repository = runtime.context["value_es_repository"]

        logger.info(f"召回{label}关键词：{keywords}")

        if recall_type == "column":
            results = await recall_column(embedding_client, column_qdrant_repository, keywords)
            writer({"type": "progress", "step": f"召回{label}", "status": "success"})
            logger.info(f"召回{label}：{[r.id for r in results]}")
            return {"retrieved_columns": results}

        elif recall_type == "value":
            results = await recall_values(value_es_repository, keywords)
            writer({"type": "progress", "step": f"召回{label}", "status": "success"})
            logger.info(f"召回{label}：{[r.id for r in results]}")
            return {"retrieved_values": results}

        elif recall_type == "metric":
            results = await recall_metrics(embedding_client, metric_qdrant_repository, keywords)
            writer({"type": "progress", "step": f"召回{label}", "status": "success"})
            logger.info(f"召回{label}：{[r.id for r in results]}")
            return {"retrieved_metrics": results}

    except Exception as e:
        # [降级] 召回失败时返回空结果，不影响其他并行分支和下游节点
        logger.warning(f"召回{label}异常，降级为空结果: {e}")
        writer({"type": "progress", "step": f"召回{label}", "status": "fallback"})
        if recall_type == "column":
            return {"retrieved_columns": []}
        elif recall_type == "value":
            return {"retrieved_values": []}
        elif recall_type == "metric":
            return {"retrieved_metrics": []}
        return {}
