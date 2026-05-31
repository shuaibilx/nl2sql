from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


MAX_RETRY = 3


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    sql = state["sql"]
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRY:
        message = f"SQL validation failed after {MAX_RETRY} correction attempts; refusing to execute."
        writer({"type": "progress", "step": "验证SQL", "status": "error", "detail": message})
        logger.error(f"{message} SQL: {sql}")
        raise RuntimeError(message)

    writer({"type": "progress", "step": "验证SQL", "status": "running"})

    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    try:
        await dw_mysql_repository.validate_sql(sql)
        writer({"type": "progress", "step": "验证SQL", "status": "success"})
        logger.info(f"SQL验证成功: {sql}")
        return {"error": None}
    except Exception as e:
        writer({"type": "progress", "step": "验证SQL", "status": "error"})
        logger.error(f"SQL验证失败(第{retry_count + 1}次): {sql}")
        return {"error": str(e)}
