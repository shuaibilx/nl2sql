# [改进] 简化：移除了独立的 LLM 关键词扩展调用，改用 expand_keywords 节点统一扩展的 value_keywords

from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger
from app.entities.value_info import ValueInfo


async def recall_value(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "召回字段取值", "status": "running"})

    # [改进] 直接使用 expand_keywords 节点扩展好的 value_keywords，不再各自调用 LLM
    keywords = state["value_keywords"]

    value_es_repository = runtime.context["value_es_repository"]

    try:
        values_map: dict[str, ValueInfo] = {}
        logger.info(f"召回字段取值关键词：{keywords}")
        for keyword in keywords:
            values: list[ValueInfo] = await value_es_repository.search(keyword)
            for value in values:
                value_id = value.id
                if value_id not in values_map:
                    values_map[value_id] = value

        retrieved_values = list(values_map.values())

        writer({"type": "progress", "step": "召回字段取值", "status": "success"})
        logger.info(f"召回字段取值：{list(values_map.keys())}")

        return {'retrieved_values': retrieved_values}
    except Exception as e:
        writer({"type": "progress", "step": "召回字段取值", "status": "error"})
        logger.error(f"召回字段取值失败: {str(e)}")
        raise
