import yaml
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm import llm_flash
from app.agent.llm import call_with_fallback
from app.agent.state import DataAgentState
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def filter_table(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "过滤表格", "status": "running"})

    query = state.get("cleaned_query") or state["query"]
    table_infos = state["table_infos"]

    try:
        # 用LLM过滤表信息
        prompt = PromptTemplate(template=load_prompt("filter_table_info"), input_variables=["query", "table_infos"])
        output_parser = JsonOutputParser()

        invoke_args = {"query": query, "table_infos": yaml.dump(table_infos, allow_unicode=True, sort_keys=False)}
        result = await call_with_fallback(prompt, output_parser, invoke_args,
                                          primary_llm=llm_flash, label="filter_table")

        # 利用模型输出过滤table_infos
        # {
        #   'fact_order':['order_amount', 'region_id'],
        #   'dim_region':['region_id', 'region_name']
        # }
        for table_info in table_infos[:]: # 从list[TableInfoState]取出单个TableInfoState
            if table_info["name"] not in result: # 如果遍历的这个表名不在result中，就从list[TableInfoStatre]中移除
                table_infos.remove(table_info)
            else:
                selected_columns = result[table_info["name"]] # 被lmm选中的字段名列表:list[column_name]
                for column_info in table_info["columns"][:]:
                    if column_info["name"] not in selected_columns:
                        table_info["columns"].remove(column_info)

        writer({"type": "progress", "step": "过滤表格", "status": "success"})
        logger.info(f"过滤后的表信息: {[table_info['name'] for table_info in table_infos]}")
        return {"table_infos": table_infos}
    except Exception as e:
        writer({"type": "progress", "step": "过滤表格", "status": "error"})
        logger.error(f"过滤表失败:{str(e)}")
        raise
