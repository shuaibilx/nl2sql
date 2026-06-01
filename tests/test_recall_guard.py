import pytest

from app.agent.nodes import recall_node as recall_node_module
from app.agent.nodes.merge_retrieved_info import NO_CONTEXT_MESSAGE, merge_retrieved_info
from app.agent.nodes.no_context_response import no_context_response
from app.agent.nodes.recall_node import recall_node
from app.conf.app_config import app_config


class DummyRuntime:
    def __init__(self, context=None):
        self.context = context or {}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


class DummyQdrantRepository:
    def __init__(self):
        self.score_threshold = None

    async def search(self, embedding, score_threshold=0.0, limit=5):
        self.score_threshold = score_threshold
        return []


@pytest.mark.asyncio
async def test_recall_node_uses_configured_qdrant_threshold(monkeypatch):
    repository = DummyQdrantRepository()
    runtime = DummyRuntime(
        {
            "embedding_client": object(),
            "column_qdrant_repository": repository,
            "metric_qdrant_repository": DummyQdrantRepository(),
            "value_es_repository": object(),
        }
    )

    async def fake_embed(*_):
        return [0.1]

    monkeypatch.setattr(recall_node_module, "cached_embed_query", fake_embed)

    result = await recall_node(
        {"recall_type": "column", "column_keywords": ["火星基地"]},
        runtime,
    )

    assert result == {"retrieved_columns": []}
    assert repository.score_threshold == app_config.recall.column_score_threshold


@pytest.mark.asyncio
async def test_merge_retrieved_info_stops_when_all_recall_results_are_empty():
    runtime = DummyRuntime({"meta_mysql_repository": object()})

    result = await merge_retrieved_info(
        {"retrieved_columns": [], "retrieved_values": [], "retrieved_metrics": []},
        runtime,
    )

    assert result == {"no_context": True, "no_context_message": NO_CONTEXT_MESSAGE}
    assert runtime.events[-1]["status"] == "no_context"


@pytest.mark.asyncio
async def test_no_context_response_returns_user_facing_message():
    runtime = DummyRuntime()

    result = await no_context_response({"query": "火星基地销量"}, runtime)

    assert result["sql"] == ""
    assert result["result_data"] == [{"message": NO_CONTEXT_MESSAGE}]
    assert runtime.events[-1] == {
        "type": "result",
        "data": {"message": NO_CONTEXT_MESSAGE, "rows": []},
    }
