from langgraph.runtime import Runtime
from langgraph.types import interrupt

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


MAX_RETRY = 3


async def validate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    sql = state["sql"]
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRY:
        detail = f"SQL validation failed after {MAX_RETRY} correction attempts; waiting for human confirmation."
        writer({"type": "progress", "step": "验证SQL", "status": "pending", "detail": detail, "sql": sql})
        logger.warning(f"{detail} SQL: {sql}")

        confirmed = interrupt({
            "action": "confirm_failed_sql",
            "sql": sql,
            "error": state.get("error"),
            "retry_count": retry_count,
            "hint": "SQL连续校验失败，请确认是否仍要执行该SQL。",
            "options": [
                {"label": "确认执行", "value": True, "description": "允许执行这条未通过自动校验的SQL"},
                {"label": "取消执行", "value": False, "description": "终止本次查询，不执行SQL"},
            ],
        })

        if confirmed:
            writer({"type": "progress", "step": "验证SQL", "status": "warning",
                    "detail": "用户确认执行未通过自动校验的SQL"})
            logger.warning(f"User confirmed execution of SQL after validation failures: {sql}")
            return {"error": None}

        message = "User cancelled SQL execution after validation failures."
        writer({"type": "progress", "step": "验证SQL", "status": "cancelled", "detail": message})
        logger.info(f"{message} SQL: {sql}")
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
