from pydantic import BaseModel


class CacheScopeSchema(BaseModel):
    tenant_id: str | None = None
    user_id: str | None = None
    project_id: str | None = None


class QuerySchema(CacheScopeSchema):
    query: str
    session_id: str | None = None


class ResumeSchema(CacheScopeSchema):
    session_id: str
    confirmed: bool
