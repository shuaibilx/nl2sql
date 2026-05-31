# [改进] 简化：移除了独立的 LLM 关键词扩展调用，改用 expand_keywords 节点统一扩展的 column_keywords

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger
from app.core.retry import retry_async  # [改进] 为 embedding 调用添加指数退避重试
from app.entities.column_info import ColumnInfo


async def recall_column(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回字段", "status": "running"})

    # [改进] 直接使用 expand_keywords 节点扩展好的 column_keywords，不再各自调用 LLM
    keywords = state["column_keywords"]

    embedding_client = runtime.context["embedding_client"]
    column_qdrant_repository = runtime.context["column_qdrant_repository"]

    try:
        retrieved_columns_map: dict[str, ColumnInfo] = {}

        logger.info(f"召回字段信息关键词：{keywords}")
        for keyword in keywords:
            embedding = await retry_async(embedding_client.aembed_query, keyword,  # [改进] embedding 调用加重试
                                           operation_name="Embedding-aembed_query")
            payloads: list[ColumnInfo] = await column_qdrant_repository.search(
                embedding
            )
            for payload in payloads:
                column_id = payload.id
                if column_id not in retrieved_columns_map:
                    retrieved_columns_map[column_id] = payload

        retrieved_columns = list(retrieved_columns_map.values())

        writer({"type": "progress", "step": "召回字段", "status": "success"})
        logger.info(f"召回字段信息：{list(retrieved_columns_map.keys())}")
        return {"retrieved_columns": retrieved_columns}
    except Exception as e:
        writer({"type": "progress", "step": "召回字段", "status": "error"})
        logger.error(f"召回字段信息失败: {str(e)}")
        raise
