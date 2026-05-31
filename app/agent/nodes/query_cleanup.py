# [改进] 查询清洗节点：在 jieba 分词之前对用户原始查询进行纠错、去噪、规范化
# 解决错别字、口语化表达、冗余前缀、模糊表述、中英混杂等问题，提升后续检索质量

from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import llm
from app.agent.llm import llm_flash
from app.agent.state import DataAgentState
from app.core.cache_registry import caches
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


# [改进] LLM 清洗结果缓存：相同查询的清洗结果不变，maxsize=512，TTL=1小时
async def _call_llm_cleanup(query: str) -> str:
    """调用 LLM 清洗查询，结果会被缓存"""
    cached = await caches.llm_cleanup.get(query)
    if cached is not None:
        return cached
    prompt = PromptTemplate(
        template=load_prompt("query_cleanup"),
        input_variables=["query"],
    )
    chain = prompt | llm_flash
    result = await chain.ainvoke({"query": query})
    cleaned = result.content.strip()
    await caches.llm_cleanup.set(query, cleaned)
    return cleaned


async def query_cleanup(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "清洗查询", "status": "running"})

    query = state["query"]

    try:
        # [改进] 缓存的 LLM 调用：相同查询直接返回缓存结果
        cleaned_query = await _call_llm_cleanup(query)

        # 如果清洗结果与原始查询相同，说明无需清洗
        if cleaned_query == query:
            writer({"type": "progress", "step": "清洗查询", "status": "success", "detail": "查询无需清洗"})
            logger.info(f"查询无需清洗: {query}")
        else:
            writer({"type": "progress", "step": "清洗查询", "status": "success",
                    "detail": f"原始: {query} → 清洗后: {cleaned_query}"})
            logger.info(f"查询清洗: {query} → {cleaned_query}")

        return {"cleaned_query": cleaned_query}

    except Exception as e:
        # 清洗失败时降级使用原始查询，不阻断流程
        writer({"type": "progress", "step": "清洗查询", "status": "warning",
                "detail": f"清洗失败，使用原始查询: {str(e)[:100]}"})
        logger.warning(f"查询清洗失败，降级使用原始查询: {str(e)}")
        return {"cleaned_query": query}
