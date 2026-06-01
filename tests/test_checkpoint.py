import json
from types import SimpleNamespace

import pytest

from app.conf.app_config import CheckpointConfig
from app.core.cache_context import CacheScope
from app.core.cache_metrics import render_metrics
from app.services import query_service as query_service_module
from app.services.query_service import QueryService
from checkpoints.base import make_checkpoint_thread_id
from checkpoints.manager import close_checkpointer, init_checkpointer


def test_checkpoint_thread_id_is_stable_and_scope_isolated():
    scope = CacheScope("tenant-a", "user-a", "project-a")
    same_scope = CacheScope("tenant-a", "user-a", "project-a")
    other_scope = CacheScope("tenant-b", "user-a", "project-a")

    thread_id = make_checkpoint_thread_id(scope, "session-1")

    assert thread_id == make_checkpoint_thread_id(same_scope, "session-1")
    assert thread_id != "session-1"
    assert thread_id != make_checkpoint_thread_id(other_scope, "session-1")


@pytest.mark.asyncio
async def test_query_service_uses_internal_thread_id_but_returns_external_session(monkeypatch):
    captured = {}

    async def fake_touch_checkpoint_session(**kwargs):
        captured.update(kwargs)

    class FakeGraph:
        async def astream(self, *, config, **kwargs):
            captured["graph_thread_id"] = config["configurable"]["thread_id"]
            yield (
                "updates",
                {
                    "__interrupt__": (
                        SimpleNamespace(value={"reason": "needs_confirmation"}),
                    )
                },
            )

    monkeypatch.setattr(
        query_service_module,
        "touch_checkpoint_session",
        fake_touch_checkpoint_session,
    )
    monkeypatch.setattr(query_service_module, "get_graph", lambda: FakeGraph())

    service = QueryService(None, None, None, None, None, None)
    events = [
        event
        async for event in service.query(
            "统计销售额",
            session_id="external-session",
            tenant_id="tenant-a",
            user_id="user-a",
            project_id="project-a",
        )
    ]

    payload = json.loads(events[0].removeprefix("data: ").strip())
    assert payload["type"] == "interrupt"
    assert payload["session_id"] == "external-session"
    assert captured["thread_id"] == captured["graph_thread_id"]
    assert captured["thread_id"] == make_checkpoint_thread_id(
        CacheScope("tenant-a", "user-a", "project-a"),
        "external-session",
    )


def test_checkpoint_metrics_are_exported():
    metrics = render_metrics().decode()
    assert "nl2sql_checkpoint_backend_up" in metrics
    assert "nl2sql_checkpoint_operations_total" in metrics
    assert "nl2sql_checkpoint_active_sessions" in metrics


def test_old_agent_checkpoint_import_path_is_removed():
    import subprocess

    result = subprocess.run(
        ["rg", "app\\.agent\\.checkpoint", "app", "tests", "eval"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1


@pytest.mark.asyncio
async def test_sqlite_checkpoint_backend_can_initialize(tmp_path):
    config = CheckpointConfig(
        backend="sqlite",
        postgres_dsn="postgresql://unused",
        pool_min_size=1,
        pool_max_size=1,
        setup_on_start=True,
        retention_days=30,
        sqlite_path=str(tmp_path / "checkpoint.db"),
        strict_msgpack=True,
    )

    await init_checkpointer(config)
    try:
        assert (tmp_path / "checkpoint.db").exists()
    finally:
        await close_checkpointer()
