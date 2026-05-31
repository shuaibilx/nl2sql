# [改进] 简化：移除了独立的 LLM 关键词扩展调用，改用 expand_keywords 节点统一扩展的 metric_keywords

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger
from app.core.retry import retry_async  # [改进] 为 embedding 调用添加指数退避重试
from app.entities.metric_info import MetricInfo


async def recall_metric(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回指标", "status": "running"})

    # [改进] 直接使用 expand_keywords 节点扩展好的 metric_keywords，不再各自调用 LLM
    keywords = state["metric_keywords"]

    embedding_client = runtime.context['embedding_client']
    metric_qdrant_repository = runtime.context['metric_qdrant_repository']

    try:
        retrieved_metrics_map: dict[str, MetricInfo] = {}

        logger.info(f"召回指标信息关键词：{keywords}")
        for keyword in keywords:
            embedding = await retry_async(embedding_client.aembed_query, keyword,  # [改进] embedding 调用加重试
                                           operation_name="Embedding-aembed_query")
            payloads: list[MetricInfo] = await metric_qdrant_repository.search(embedding)
            for payload in payloads:
                metric_id = payload.id
                if metric_id not in retrieved_metrics_map:
                    retrieved_metrics_map[metric_id] = payload

        retrieved_metrics = list(retrieved_metrics_map.values())

        writer({"type": "progress", "step": "召回指标", "status": "success"})
        logger.info(f"召回指标信息：{list(retrieved_metrics_map.keys())}")
        return {"retrieved_metrics": retrieved_metrics}
    except Exception as e:
        writer({"type": "progress", "step": "召回指标", "status": "error"})
        logger.error(f"召回指标信息失败: {str(e)}")
        raise
