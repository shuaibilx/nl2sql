from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.conf.app_config import app_config
from app.core.embedding_cache import cached_embed_query
from app.core.log import logger
from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo


async def recall_column(embedding_client, column_qdrant_repository, keywords):
    retrieved_map: dict[str, ColumnInfo] = {}
    for keyword in keywords:
        embedding = await cached_embed_query(embedding_client, keyword)
        payloads: list[ColumnInfo] = await column_qdrant_repository.search(
            embedding,
            score_threshold=app_config.recall.column_score_threshold,
        )
        for payload in payloads:
            if payload.id not in retrieved_map:
                retrieved_map[payload.id] = payload
    return list(retrieved_map.values())


async def recall_values(value_es_repository, keywords):
    retrieved_map: dict[str, ValueInfo] = {}
    for keyword in keywords:
        values: list[ValueInfo] = await value_es_repository.search(
            keyword,
            score_threshold=app_config.recall.value_score_threshold,
        )
        for value in values:
            if value.id not in retrieved_map:
                retrieved_map[value.id] = value
    return list(retrieved_map.values())


async def recall_metrics(embedding_client, metric_qdrant_repository, keywords):
    retrieved_map: dict[str, MetricInfo] = {}
    for keyword in keywords:
        embedding = await cached_embed_query(embedding_client, keyword)
        payloads: list[MetricInfo] = await metric_qdrant_repository.search(
            embedding,
            score_threshold=app_config.recall.metric_score_threshold,
        )
        for payload in payloads:
            if payload.id not in retrieved_map:
                retrieved_map[payload.id] = payload
    return list(retrieved_map.values())


RECALL_REGISTRY = {
    "column": ("字段", recall_column, "column_keywords"),
    "value": ("取值", recall_values, "value_keywords"),
    "metric": ("指标", recall_metrics, "metric_keywords"),
}


async def recall_node(state: DataAgentState, runtime: Runtime[DataAgentContext]):
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

        logger.info(f"召回{label}关键词: {keywords}")

        if recall_type == "column":
            results = await recall_column(embedding_client, column_qdrant_repository, keywords)
            writer({"type": "progress", "step": f"召回{label}", "status": "success"})
            logger.info(f"召回{label}: {[r.id for r in results]}")
            return {"retrieved_columns": results}

        if recall_type == "value":
            results = await recall_values(value_es_repository, keywords)
            writer({"type": "progress", "step": f"召回{label}", "status": "success"})
            logger.info(f"召回{label}: {[r.id for r in results]}")
            return {"retrieved_values": results}

        if recall_type == "metric":
            results = await recall_metrics(embedding_client, metric_qdrant_repository, keywords)
            writer({"type": "progress", "step": f"召回{label}", "status": "success"})
            logger.info(f"召回{label}: {[r.id for r in results]}")
            return {"retrieved_metrics": results}

    except Exception as exc:
        logger.warning(f"召回{label}异常，降级为空结果: {exc}")
        writer({"type": "progress", "step": f"召回{label}", "status": "fallback"})
        if recall_type == "column":
            return {"retrieved_columns": []}
        if recall_type == "value":
            return {"retrieved_values": []}
        if recall_type == "metric":
            return {"retrieved_metrics": []}
    return {}
