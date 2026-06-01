from app.core.cache_registry import caches
from app.core.circuit_manager import circuit_manager
from app.core.retry import retry_async


async def cached_embed_query(embedding_client, text: str) -> list[float]:
    cached = await caches.embedding.get(text)
    if cached is not None:
        return cached

    circuit_breaker = circuit_manager.get("Embedding")
    result = await retry_async(
        embedding_client.aembed_query,
        text,
        operation_name="Embedding-aembed_query",
        circuit_breaker=circuit_breaker,
    )
    await caches.embedding.set(text, result)
    return result
