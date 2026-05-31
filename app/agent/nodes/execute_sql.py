from langgraph.runtime import Runtime
from langgraph.types import interrupt  # [改进] LangGraph 人机交互：在执行前暂停等待人工确认

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


async def execute_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    sql = state["sql"]
    dw_mysql_repository = runtime.context["dw_mysql_repository"]

    # [改进] 高风险操作：SQL执行前通过 interrupt() 暂停，等待用户确认后再继续
    # interrupt() 首次调用时抛出 GraphInterrupt，框架将中断信息写入 checkpoint 并暂停
    # 用户通过 Command(resume=True/False) 恢复后，interrupt() 返回对应的布尔值
    writer({"type": "progress", "step": "等待SQL确认", "status": "pending",
            "detail": "请确认以下SQL是否可以执行", "sql": sql})

    # [改进] options 字段提示客户端可用的操作及对应的 resume 值
    confirmed = interrupt({
        "action": "confirm_sql",
        "sql": sql,
        "hint": "请确认以下SQL是否可以执行，回复'确认'执行或'取消'跳过",
        "options": [
            {"label": "确认执行", "value": True, "description": "允许执行此SQL"},
            {"label": "取消执行", "value": False, "description": "跳过此SQL，不执行"},
        ]
    })

    if not confirmed:
        writer({"type": "progress", "step": "执行SQL", "status": "cancelled",
                "detail": "用户取消了SQL执行"})
        logger.info(f"用户取消了SQL执行: {sql}")
        return

    writer({"type": "progress", "step": "执行SQL", "status": "running"})

    try:
        result = await dw_mysql_repository.execute_sql(sql)

        writer({"type": "progress", "step": "执行SQL", "status": "success"})
        writer({"type": "result", "data": result})
        logger.info(f"执行SQL结果: {result}")
        return {"result_data": result}

    except Exception as e:
        writer({"type": "progress", "step": "执行SQL", "status": "error"})
        logger.error(f"执行SQL失败:{str(e)}")
        raise
