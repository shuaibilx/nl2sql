import hashlib
import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm import llm_flash
from app.agent.llm import call_with_fallback
from app.agent.state import DataAgentState
from app.core.cache_registry import caches
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "生成SQL", "status": "running"})

    query = state.get("cleaned_query") or state["query"]
    table_infos = state["table_infos"]
    metric_infos = state["metric_infos"]
    date_info = state["date_info"]
    db_info = state["db_info"]

    # [改进] SQL 缓存：相同上下文生成的 SQL 不变，避免重复 LLM 调用（~2s）
    cache_input = yaml.dump({
        "query": query, "table_infos": table_infos,
        "metric_infos": metric_infos, "date_info": date_info, "db_info": db_info,
    }, sort_keys=True, allow_unicode=True)
    cache_key = hashlib.md5(cache_input.encode()).hexdigest()

    cached = caches.generate_sql.get(cache_key)
    if cached is not None:
        logger.info(f"SQL 缓存命中: {cached[:80]}...")
        writer({"type": "progress", "step": "生成SQL", "status": "cached"})
        return {"sql": cached}

    try:
        prompt = PromptTemplate(template=load_prompt("generate_sql"),
                                input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info"])
        output_parser = StrOutputParser()

        invoke_args = {
            "query": query,
            "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False),
            "metric_infos": yaml.dump(metric_infos, allow_unicode=True, sort_keys=False),
            "date_info": yaml.dump(date_info, allow_unicode=True, sort_keys=False),
            "db_info": yaml.dump(db_info, allow_unicode=True, sort_keys=False),
        }
        result = await call_with_fallback(prompt, output_parser, invoke_args,
                                          primary_llm=llm_flash, label="generate_sql")

        caches.generate_sql.set(cache_key, result)
        writer({"type": "progress", "step": "生成SQL", "status": "success"})
        logger.info(f"生成的SQL: {result}")
        return {"sql": result}
    except Exception as e:
        writer({"type": "progress", "step": "生成SQL", "status": "error"})
        logger.error(f"生成SQL失败: {str(e)}")
        raise
