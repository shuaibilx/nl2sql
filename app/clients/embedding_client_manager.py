from typing import Optional

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from app.conf.app_config import EmbeddingConfig, app_config

# 客户端管理器
# Embedding模型客户端

class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[HuggingFaceEndpointEmbeddings] = None
        self.config = config

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = HuggingFaceEndpointEmbeddings(model=self._get_url())

    async def close(self):
        if self.client is None:
            return
        async_client = getattr(self.client, "async_client", None)
        if async_client is not None and hasattr(async_client, "close"):
            await async_client.close()
        self.client = None


embedding_client_manager = EmbeddingClientManager(app_config.embedding)
