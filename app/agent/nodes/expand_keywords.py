# [改进] 统一关键词扩展节点：一次 LLM 调用同时扩展字段、取值、指标三个维度的关键词
# 原本 recall_column/recall_value/recall_metric 各自调用一次 LLM（共 3 次），合并后仅 1 次

from langchain_core.output_parsers import JsonOutputParser
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


# [改进] LLM 结果缓存：相同查询的关键词扩展结果不变，maxsize=256 覆盖常见查询集，TTL=30分钟
async def _call_llm_extend(query: str) -> dict:
    """调用 LLM 扩展关键词，结果会被缓存。主模型失败自动降级到备用模型"""
    cached = await caches.llm_expand.get(query)
    if cached is not None:
        return cached
    prompt = PromptTemplate(
        template=load_prompt("extend_keywords"),
        input_variables=["query"],
    )
    output_parser = JsonOutputParser()
    result = await call_with_fallback(prompt, output_parser, {"query": query},
                                      primary_llm=llm, label="expand_keywords")
    await caches.llm_expand.set(query, result)
    return result


async def expand_keywords(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "扩展关键词", "status": "running"})

    # [改进] 使用清洗后的查询进行关键词扩展，纠错后的查询扩展质量更高
    query = state.get("cleaned_query") or state["query"]
    base_keywords = state["keywords"]

    try:
        # [改进] 缓存的 LLM 调用：相同查询直接返回缓存结果，省去 LLM 调用延迟
        result = await _call_llm_extend(query)

        # [改进] 各维度分别与 base_keywords 合并去重，确保原始关键词不丢失
        column_keywords = list(set(base_keywords + result.get("column_keywords", [])))
        value_keywords = list(set(base_keywords + result.get("value_keywords", [])))
        metric_keywords = list(set(base_keywords + result.get("metric_keywords", [])))

        writer({"type": "progress", "step": "扩展关键词", "status": "success"})
        logger.info(f"扩展字段关键词：{column_keywords}")
        logger.info(f"扩展取值关键词：{value_keywords}")
        logger.info(f"扩展指标关键词：{metric_keywords}")

        return {
            "column_keywords": column_keywords,
            "value_keywords": value_keywords,
            "metric_keywords": metric_keywords,
        }
    except Exception as e:
        # [降级] LLM 扩展失败时，用原始 keywords 作为三路关键词继续流程
        logger.warning(f"扩展关键词失败，降级为原始关键词: {str(e)}")
        writer({"type": "progress", "step": "扩展关键词", "status": "fallback"})
        return {
            "column_keywords": base_keywords,
            "value_keywords": base_keywords,
            "metric_keywords": base_keywords,
        }
