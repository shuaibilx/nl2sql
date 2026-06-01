import hashlib
import time
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver

from app.core.cache_context import CacheScope
from app.core.cache_metrics import checkpoint_operation_seconds, checkpoint_operations_total


CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("app.entities.column_info", "ColumnInfo"),
    ("app.entities.column_metric", "ColumnMetric"),
    ("app.entities.metric_info", "MetricInfo"),
    ("app.entities.table_info", "TableInfo"),
    ("app.entities.value_info", "ValueInfo"),
]


class InstrumentedCheckpointer(BaseCheckpointSaver):
    def __init__(self, backend: str, saver: Any):
        super().__init__(serde=getattr(saver, "serde", None))
        self.backend = backend
        self.saver = saver

    def __getattr__(self, name: str):
        return getattr(self.saver, name)

    async def _record(self, operation: str, func, *args, **kwargs):
        started = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
        except Exception:
            checkpoint_operations_total.labels(self.backend, operation, "error").inc()
            raise
        else:
            checkpoint_operations_total.labels(self.backend, operation, "success").inc()
            return result
        finally:
            checkpoint_operation_seconds.labels(self.backend, operation).observe(
                time.perf_counter() - started
            )

    async def aget(self, *args, **kwargs):
        return await self._record("aget", self.saver.aget, *args, **kwargs)

    async def aget_tuple(self, *args, **kwargs):
        return await self._record("aget_tuple", self.saver.aget_tuple, *args, **kwargs)

    async def aput(self, *args, **kwargs):
        return await self._record("aput", self.saver.aput, *args, **kwargs)

    async def aput_writes(self, *args, **kwargs):
        return await self._record("aput_writes", self.saver.aput_writes, *args, **kwargs)

    async def adelete_thread(self, *args, **kwargs):
        return await self._record(
            "adelete_thread", self.saver.adelete_thread, *args, **kwargs
        )

    async def aget_delta_channel_history(self, *args, **kwargs):
        return await self._record(
            "aget_delta_channel_history",
            self.saver.aget_delta_channel_history,
            *args,
            **kwargs,
        )


def make_checkpoint_thread_id(scope: CacheScope, session_id: str) -> str:
    raw = f"{scope.tenant_id}:{scope.user_id}:{scope.project_id}:{session_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
