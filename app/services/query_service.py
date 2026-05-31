import json
import uuid

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langgraph.types import Command  # [改进] 用于 interrupt 恢复：Command(resume=...) 继续暂停的图

from app.agent.context import DataAgentContext
from app.agent.graph import get_graph  # [改进] 延迟获取编译后的 graph，确保 checkpointer 已绑定
from app.agent.state import DataAgentState
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


class QueryService:
    def __init__(self,
                 embedding_client: HuggingFaceEndpointEmbeddings,
                 column_qdrant_repository: ColumnQdrantRepository,
                 value_es_repository: ValueESRepository,
                 metric_qdrant_repository: MetricQdrantRepository,
                 meta_mysql_repository: MetaMySQLRepository,
                 dw_mysql_repository: DWMySQLRepository):
        self.embedding_client = embedding_client
        self.column_qdrant_repository = column_qdrant_repository
        self.value_es_repository = value_es_repository
        self.metric_qdrant_repository = metric_qdrant_repository
        self.meta_mysql_repository = meta_mysql_repository
        self.dw_mysql_repository = dw_mysql_repository

    # [改进] 新增 session_id 参数：同一id多次查询可共享 checkpointer 持久化的状态，实现多轮对话
    async def query(self, query: str, session_id: str | None = None):
        session_id = session_id or str(uuid.uuid4())  # [改进] 未传则自动生成，保证总有 thread_id
        config = {"configurable": {"thread_id": session_id}}  # [改进] checkpointer 通过此配置区分不同会话

        context = DataAgentContext(
            embedding_client=self.embedding_client,
            column_qdrant_repository=self.column_qdrant_repository,
            value_es_repository=self.value_es_repository,
            metric_qdrant_repository=self.metric_qdrant_repository,
            meta_mysql_repository=self.meta_mysql_repository,
            dw_mysql_repository=self.dw_mysql_repository
        )
        # [改进] cleaned_query 初始化为空字符串，由 query_cleanup 节点填充
        state = DataAgentState(query=query, cleaned_query="", retry_count=0)
        async for event in self._stream(state, context, config):
            yield event

    # [改进] 人机交互：用户确认SQL后恢复图执行
    async def resume(self, session_id: str, confirmed: bool):
        """[改进] 用 Command(resume=confirmed) 恢复被 interrupt() 暂停的图"""
        config = {"configurable": {"thread_id": session_id}}
        # [改进] Command(resume=...) 将布尔值传回 interrupt() 调用点，true=确认执行，false=取消
        async for event in self._stream(Command(resume=confirmed), None, config):
            yield event

    async def _stream(self, input_, context, config):
        """[改进] 统一的流式处理：监听 custom 事件 + updates（含 __interrupt__），统一转为 SSE"""
        graph = get_graph()
        try:
            # [改进] 双流模式：custom=自定义进度事件，updates=状态变更（含 __interrupt__）
            async for mode, chunk in graph.astream(
                input=input_, context=context, config=config,
                stream_mode=["custom", "updates"]
            ):
                if mode == "updates" and "__interrupt__" in chunk:
                    # [改进] 检测到 interrupt，提取 SQL 确认信息发给客户端，然后暂停流
                    interrupt_tuple = chunk["__interrupt__"]
                    interrupt_info = interrupt_tuple[0].value  # {"action": "confirm_sql", "sql": "..."}
                    yield f"data: {json.dumps({'type': 'interrupt', 'session_id': config['configurable']['thread_id'], **interrupt_info}, ensure_ascii=False, default=str)}\n\n"
                    return  # [改进] 暂停 SSE 流，等待用户通过 /api/query/resume 确认
                elif mode == "custom":
                    yield f"data: {json.dumps(chunk, ensure_ascii=False, default=str)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False, default=str)}\n\n"
