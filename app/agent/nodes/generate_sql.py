import hashlib

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.llm import call_with_fallback, llm_flash
from app.agent.state import DataAgentState
from app.core.cache_registry import caches
from app.core.embedding_cache import cached_embed_query
from app.core.log import logger
from app.prompt.prompt_loader import load_prompt


async def generate_sql(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "生成SQL", "status": "running"})

    query = state.get("cleaned_query") or state["query"]
    context_input = {
        "table_infos": state["table_infos"],
        "metric_infos": state["metric_infos"],
        "date_info": state["date_info"],
        "db_info": state["db_info"],
    }
    context_fingerprint = yaml.dump(context_input, sort_keys=True, allow_unicode=True)
    cache_input = yaml.dump({"query": query, **context_input}, sort_keys=True, allow_unicode=True)
    cache_key = hashlib.md5(cache_input.encode()).hexdigest()
    query_embedding = None

    cached = await caches.generate_sql.get(cache_key)
    if cached is not None:
        logger.info(f"SQL cache hit: {cached[:80]}...")
        writer({"type": "progress", "step": "生成SQL", "status": "cached"})
        return {"sql": cached}

    if caches.config.semantic_enabled:
        try:
            embedding_client = runtime.context["embedding_client"]
            query_embedding = await cached_embed_query(embedding_client, query)
            semantic_hit = await caches.semantic_generate_sql.get(
                query=query,
                embedding=query_embedding,
                context_key=context_fingerprint,
            )
            if semantic_hit is not None:
                logger.info(
                    "Semantic SQL cache hit: "
                    f"similarity={semantic_hit.similarity:.4f}, query={semantic_hit.query}"
                )
                await caches.generate_sql.set(cache_key, semantic_hit.value)
                writer({"type": "progress", "step": "生成SQL", "status": "semantic_cached"})
                return {"sql": semantic_hit.value}
        except Exception as exc:
            logger.warning(f"Semantic SQL cache lookup skipped: {exc}")

    try:
        prompt = PromptTemplate(
            template=load_prompt("generate_sql"),
            input_variables=["query", "table_infos", "metric_infos", "date_info", "db_info"],
        )
        output_parser = StrOutputParser()

        invoke_args = {
            "query": query,
            "table_infos": yaml.dump(context_input["table_infos"], allow_unicode=True, sort_keys=False),
            "metric_infos": yaml.dump(context_input["metric_infos"], allow_unicode=True, sort_keys=False),
            "date_info": yaml.dump(context_input["date_info"], allow_unicode=True, sort_keys=False),
            "db_info": yaml.dump(context_input["db_info"], allow_unicode=True, sort_keys=False),
        }
        result = await call_with_fallback(
            prompt,
            output_parser,
            invoke_args,
            primary_llm=llm_flash,
            label="generate_sql",
        )

        await caches.generate_sql.set(cache_key, result)
        if caches.config.semantic_enabled:
            try:
                if query_embedding is None:
                    embedding_client = runtime.context["embedding_client"]
                    query_embedding = await cached_embed_query(embedding_client, query)
                await caches.semantic_generate_sql.set(
                    query=query,
                    embedding=query_embedding,
                    context_key=context_fingerprint,
                    value=result,
                )
            except Exception as exc:
                logger.warning(f"Semantic SQL cache write skipped: {exc}")

        writer({"type": "progress", "step": "生成SQL", "status": "success"})
        logger.info(f"Generated SQL: {result}")
        return {"sql": result}
    except Exception as exc:
        writer({"type": "progress", "step": "生成SQL", "status": "error"})
        logger.error(f"generate_sql failed: {exc}")
        raise
