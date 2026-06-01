import os
from dotenv import load_dotenv
import asyncio

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Send  # [改进] Send API：动态派发并行分支，实现 map-reduce

from app.agent.context import DataAgentContext
from app.agent.nodes.add_extra_context import add_extra_context
from app.agent.nodes.correct_sql import correct_sql
from app.agent.nodes.execute_sql import execute_sql
from app.agent.nodes.expand_keywords import expand_keywords  # [改进] 统一关键词扩展节点，替代 recall 各自调用 LLM
from app.agent.nodes.extract_keywords import extract_keywords
from app.agent.nodes.filter_metric import filter_metric
from app.agent.nodes.query_cleanup import query_cleanup  # [改进] 查询清洗节点，纠错+去噪+规范化
from app.agent.nodes.filter_table import filter_table
from app.agent.nodes.generate_sql import generate_sql
from app.agent.nodes.merge_retrieved_info import merge_retrieved_info
from app.agent.nodes.no_context_response import no_context_response
from app.agent.nodes.recall_node import recall_node  # [改进] 统一召回节点，由 Send API 按类型派发
from app.agent.nodes.validate_sql import validate_sql
from checkpoints import manager as checkpoint_module  # [改进] 导入模块以引用可变全局变量，避免 from-import 值绑定问题
from app.agent.state import DataAgentState
from app.core.log import logger
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager, dw_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository


load_dotenv()
os.environ.setdefault("LANGSMITH_PROJECT", "nl2sql")

graph_builder = StateGraph(state_schema=DataAgentState, context_schema=DataAgentContext)


# [改进] Send API 派发函数：expand_keywords 完成后动态生成 3 个并行召回分支
# 每个 Send 携带定制化的 state 子集（recall_type + 对应维度的关键词）
def send_to_recalls(state: DataAgentState):
    return [
        Send("recall_node", {"recall_type": "column", "query": state["query"], "column_keywords": state.get("column_keywords", [])}),
        Send("recall_node", {"recall_type": "value",  "query": state["query"], "value_keywords":  state.get("value_keywords", [])}),
        Send("recall_node", {"recall_type": "metric", "query": state["query"], "metric_keywords": state.get("metric_keywords", [])}),
    ]


# 添加节点
graph_builder.add_node("query_cleanup", query_cleanup)  # [改进] 查询清洗：纠错+去噪+规范化，提升后续检索质量
graph_builder.add_node("extract_keywords", extract_keywords)
graph_builder.add_node("expand_keywords", expand_keywords)  # [改进] 统一扩展三个维度关键词，1次LLM替代3次
graph_builder.add_node("recall_node", recall_node)  # [改进] 统一召回节点，由 Send 动态派发 3 个并行实例
graph_builder.add_node("merge_retrieved_info", merge_retrieved_info)
graph_builder.add_node("no_context_response", no_context_response)
graph_builder.add_node("filter_metric", filter_metric)
graph_builder.add_node("filter_table", filter_table)
graph_builder.add_node("add_extra_context", add_extra_context)
graph_builder.add_node("generate_sql", generate_sql)
graph_builder.add_node("validate_sql", validate_sql)
graph_builder.add_node("correct_sql", correct_sql)
graph_builder.add_node("execute_sql", execute_sql)

# 添加关系
# [改进] 查询清洗是第一个节点：纠错+去噪+规范化，清洗后的查询传递给后续所有节点
graph_builder.add_edge(START, "query_cleanup")
graph_builder.add_edge("query_cleanup", "extract_keywords")
graph_builder.add_edge("extract_keywords", "expand_keywords")

# [改进] map-reduce 模式：expand_keywords 完成后，通过 Send API 动态派发 3 个并行召回分支
# Send("recall_node", {...}) 将不同 state 子集发送给同一个 recall_node 节点的 3 个并行实例
# 每个实例通过 recall_type 区分，只处理对应维度的关键词
graph_builder.add_conditional_edges("expand_keywords", send_to_recalls)

# [改进] 3 个并行召回分支完成后，统一汇入 merge_retrieved_info（reduce 阶段）
graph_builder.add_edge("recall_node", "merge_retrieved_info")


def route_after_merge(state: DataAgentState):
    if state.get("no_context"):
        return "no_context_response"
    return ["filter_table", "filter_metric"]


graph_builder.add_conditional_edges("merge_retrieved_info", route_after_merge)
graph_builder.add_edge("no_context_response", END)
graph_builder.add_edge("filter_table", "add_extra_context")
graph_builder.add_edge("filter_metric", "add_extra_context")
graph_builder.add_edge("add_extra_context", "generate_sql")
graph_builder.add_edge("generate_sql", "validate_sql")

# [改进] validate_sql条件边：通过(error=None) → 执行SQL；失败但未超3次 → 校正；失败且超3次 → 降级强制执行
graph_builder.add_conditional_edges("validate_sql",
                                    lambda state: "execute_sql"
                                    if (state.get("error") is None or state.get("retry_count", 0) >= 3)
                                    else "correct_sql",
                                    {"execute_sql": "execute_sql", "correct_sql": "correct_sql"})

# [改进] correct_sql → validate_sql 形成回环，配合 retry_count 校验上限实现"校验→校正→再校验"循环
graph_builder.add_edge("correct_sql", "validate_sql")
graph_builder.add_edge("execute_sql", END)

# [改进] 延迟编译：checkpointer 在 lifespan 启动时异步初始化，此处先置 None
# 通过 get_graph() 获取编译后的图，确保调用方始终拿到绑定 checkpointer 的有效实例
graph = None


def get_graph() -> StateGraph:
    """[改进] 返回已编译的图实例。若尚未编译（lifespan 未调用 setup），则抛出异常。"""
    if graph is None:
        raise RuntimeError("Graph 尚未编译，请确保 lifespan 已调用 setup_graph()")
    return graph


def setup_graph() -> None:
    """[改进] 编译图并绑定 checkpointer，由 lifespan 在 checkpointer 初始化之后调用"""
    global graph
    # [改进] 通过模块引用访问 checkpointer（而非 from-import 值绑定），确保读取到 init_checkpointer() 赋值后的实例
    if checkpoint_module.checkpointer is None:
        raise RuntimeError("Checkpointer 尚未初始化，请先调用 init_checkpointer()")
    graph = graph_builder.compile(checkpointer=checkpoint_module.checkpointer)
    logger.info("[改进] Graph 编译完成，已绑定 LangGraph checkpointer")



if __name__ == '__main__':
    async def t1():
        from checkpoints.manager import init_checkpointer
        from app.conf.app_config import app_config
        embedding_client_manager.init()
        qdrant_client_manager.init()
        es_client_manager.init()
        meta_mysql_client_manager.init()
        dw_mysql_client_manager.init()
        # [改进] 先初始化 checkpointer，再编译 graph
        await init_checkpointer(app_config.checkpoint)
        setup_graph()

        async with meta_mysql_client_manager.session_factory() as meta_session, dw_mysql_client_manager.session_factory() as dw_session:
            meta_mysql_repository = MetaMySQLRepository(meta_session)
            dw_mysql_repository = DWMySQLRepository(dw_session)
            column_qdrant_repository = ColumnQdrantRepository(qdrant_client_manager.client)
            value_es_repository = ValueESRepository(es_client_manager.client)
            metric_qdrant_repository = MetricQdrantRepository(qdrant_client_manager.client)

            context = DataAgentContext(
                embedding_client=embedding_client_manager.client,
                column_qdrant_repository=column_qdrant_repository,
                value_es_repository=value_es_repository,
                metric_qdrant_repository=metric_qdrant_repository,
                meta_mysql_repository=meta_mysql_repository,
                dw_mysql_repository=dw_mysql_repository
            )
            state = DataAgentState(query="统计浙江的销售总额", cleaned_query="", retry_count=0)
            config = {"configurable": {"thread_id": "test-session-1"}}
            async for chunk in get_graph().astream(input=state, context=context, config=config, stream_mode="custom"):
                print(chunk)

        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()


    asyncio.run(t1())
