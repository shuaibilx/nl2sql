from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.nodes.merge_retrieved_info import NO_CONTEXT_MESSAGE
from app.agent.state import DataAgentState
from app.core.log import logger


async def no_context_response(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    message = state.get("no_context_message") or NO_CONTEXT_MESSAGE
    writer({"type": "progress", "step": "生成回复", "status": "no_context"})
    writer({"type": "result", "data": {"message": message, "rows": []}})
    logger.info(f"No database context for query: {state.get('query')}")
    return {"result_data": [{"message": message}], "sql": ""}
