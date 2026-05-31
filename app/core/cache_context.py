from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


DEFAULT_TENANT_ID = "default_tenant"
DEFAULT_USER_ID = "default_user"
DEFAULT_PROJECT_ID = "default_project"


@dataclass(frozen=True)
class CacheScope:
    tenant_id: str = DEFAULT_TENANT_ID
    user_id: str = DEFAULT_USER_ID
    project_id: str = DEFAULT_PROJECT_ID

    @classmethod
    def from_optional(
        cls,
        tenant_id: str | None = None,
        user_id: str | None = None,
        project_id: str | None = None,
    ) -> "CacheScope":
        return cls(
            tenant_id=tenant_id or DEFAULT_TENANT_ID,
            user_id=user_id or DEFAULT_USER_ID,
            project_id=project_id or DEFAULT_PROJECT_ID,
        )


cache_scope_ctx_var: ContextVar[CacheScope] = ContextVar("cache_scope", default=CacheScope())


def get_cache_scope() -> CacheScope:
    return cache_scope_ctx_var.get()


@contextmanager
def use_cache_scope(scope: CacheScope):
    token = cache_scope_ctx_var.set(scope)
    try:
        yield
    finally:
        cache_scope_ctx_var.reset(token)
