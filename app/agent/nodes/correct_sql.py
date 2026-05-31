import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm import call_with_fallback
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def correct_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer

    # [改进] 每次进入校正节点，retry_count 递增1，传回 validate_sql 判断是否达上限
    retry_count = state.get("retry_count", 0) + 1
    writer({"type": "progress", "step": "校正SQL", "status": "running",
            "detail": f"第{retry_count}次校正"})

    sql = state["sql"]
    error = state["error"]

    # [改进] 使用清洗后的查询进行 SQL 校正，保持一致性
    query = state.get("cleaned_query") or state["query"]
    table_infos = state["table_infos"]
    metric_infos = state["metric_infos"]
    date_info = state["date_info"]
    db_info = state["db_info"]

    try:
        # [改进] 补全 input_variables：原只含 query,metric_infos，现加入 table_infos/date_info/db_info/sql/error
        prompt = PromptTemplate(template=load_prompt("correct_sql"),
                                input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info", "sql", "error"])
        output_parser = StrOutputParser()

        invoke_args = {
            "query": query,
            "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
            "metric_infos": yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
            "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
            "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
            "sql": sql,
            "error": error,
        }
        result = await call_with_fallback(prompt, output_parser, invoke_args,
                                          primary_llm=llm, label="correct_sql")
        writer({"type": "progress", "step": "校正SQL", "status": "success"})
        logger.info(f"校正后的SQL(第{retry_count}次): {result}")
        # [改进] 返回 retry_count 到 state，供 validate_sql 判断
        return {"sql": result, "retry_count": retry_count}
    except Exception as e:
        writer({"type": "progress", "step": "校正SQL", "status": "error"})
        logger.error(f"校正SQL失败(第{retry_count}次):{str(e)}")
        raise
