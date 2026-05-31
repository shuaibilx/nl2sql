import pytest

from app.agent.nodes.validate_sql import MAX_RETRY, validate_sql


class DummyRuntime:
    def __init__(self):
        self.context = {}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_validate_sql_refuses_execution_after_max_retry():
    runtime = DummyRuntime()
    state = {"sql": "select * from missing_table", "retry_count": MAX_RETRY}

    with pytest.raises(RuntimeError, match="refusing to execute"):
        await validate_sql(state, runtime)

    assert runtime.events[-1]["status"] == "error"
