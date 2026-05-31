import pytest

from app.agent.nodes.execute_sql import execute_sql


class DummyDWRepository:
    def __init__(self):
        self.executed_sql = None

    async def execute_sql(self, sql):
        self.executed_sql = sql
        return [{"result": 1}]


class DummyRuntime:
    def __init__(self, repository):
        self.context = {"dw_mysql_repository": repository}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_execute_sql_runs_without_interrupt():
    repository = DummyDWRepository()
    runtime = DummyRuntime(repository)

    result = await execute_sql({"sql": "select 1"}, runtime)

    assert repository.executed_sql == "select 1"
    assert result == {"result_data": [{"result": 1}]}
    assert [event["status"] for event in runtime.events if event["type"] == "progress"] == ["running", "success"]
