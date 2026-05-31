import json
import uuid

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langgraph.types import Command

from app.agent.context import DataAgentContext
from app.agent.graph import get_graph
from app.agent.state import DataAgentState
from app.core.cache_context import CacheScope, use_cache_scope
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


class QueryService:
    def __init__(
        self,
        embedding_client: HuggingFaceEndpointEmbeddings,
        column_qdrant_repository: ColumnQdrantRepository,
        value_es_repository: ValueESRepository,
        metric_qdrant_repository: MetricQdrantRepository,
        meta_mysql_repository: MetaMySQLRepository,
        dw_mysql_repository: DWMySQLRepository,
    ):
        self.embedding_client = embedding_client
        self.column_qdrant_repository = column_qdrant_repository
        self.value_es_repository = value_es_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.meta_mysql_repository = meta_mysql_repository
        self.dw_mysql_repository = dw_mysql_repository

    async def query(
        self,
        query: str,
        session_id: str | None = None,
        tenant_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ):
        session_id = session_id or str(uuid.uuid4())
        scope = CacheScope.from_optional(tenant_id, user_id, project_id)
        config = {"configurable": {"thread_id": session_id}}
        context = DataAgentContext(
            embedding_client=self.embedding_client,
            column_qdrant_repository=self.column_qdrant_repository,
            value_es_repository=self.value_es_repository,
            metric_qdrant_repository=self.metric_qdrant_repository,
            meta_mysql_repository=self.meta_mysql_repository,
            dw_mysql_repository=self.dw_mysql_repository,
            cache_scope=scope,
        )
        state = DataAgentState(query=query, cleaned_query="", retry_count=0)
        async for event in self._stream(state, context, config, scope):
            yield event

    async def resume(
        self,
        session_id: str,
        confirmed: bool,
        tenant_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ):
        scope = CacheScope.from_optional(tenant_id, user_id, project_id)
        config = {"configurable": {"thread_id": session_id}}
        async for event in self._stream(Command(resume=confirmed), None, config, scope):
            yield event

    async def _stream(self, input_, context, config, scope: CacheScope):
        graph = get_graph()
        with use_cache_scope(scope):
            try:
                async for mode, chunk in graph.astream(
                    input=input_,
                    context=context,
                    config=config,
                    stream_mode=["custom", "updates"],
                ):
                    if mode == "updates" and "__interrupt__" in chunk:
                        interrupt_tuple = chunk["__interrupt__"]
                        interrupt_info = interrupt_tuple[0].value
                        yield f"data: {json.dumps({'type': 'interrupt', 'session_id': config['configurable']['thread_id'], **interrupt_info}, ensure_ascii=False, default=str)}\n\n"
                        return
                    elif mode == "custom":
                        yield f"data: {json.dumps(chunk, ensure_ascii=False, default=str)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False, default=str)}\n\n"
