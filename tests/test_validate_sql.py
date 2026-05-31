import pytest

from app.agent.nodes import validate_sql as validate_sql_module
from app.agent.nodes.validate_sql import MAX_RETRY, validate_sql


class DummyRuntime:
    def __init__(self, context=None):
        self.context = context or {}
        self.events = []

    def stream_writer(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_validate_sql_interrupts_after_max_retry_and_allows_confirmed_execution(monkeypatch):
    runtime = DummyRuntime()
    state = {
        "sql": "select * from missing_table",
        "retry_count": MAX_RETRY,
        "error": "table does not exist",
    }

    monkeypatch.setattr(validate_sql_module, "interrupt", lambda payload: True)

    result = await validate_sql(state, runtime)

    assert result == {"error": None}
    assert runtime.events[0]["status"] == "pending"
    assert runtime.events[-1]["status"] == "warning"


@pytest.mark.asyncio
async def test_validate_sql_interrupts_after_max_retry_and_cancels_execution(monkeypatch):
    runtime = DummyRuntime()
    state = {
        "sql": "select * from missing_table",
        "retry_count": MAX_RETRY,
        "error": "table does not exist",
    }

    monkeypatch.setattr(validate_sql_module, "interrupt", lambda payload: False)

    with pytest.raises(RuntimeError, match="cancelled"):
        await validate_sql(state, runtime)

    assert runtime.events[0]["status"] == "pending"
    assert runtime.events[-1]["status"] == "cancelled"
