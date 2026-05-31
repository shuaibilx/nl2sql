from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import QuerySchema, ResumeSchema
from app.services.query_service import QueryService

query_router = APIRouter()


@query_router.post("/api/query")
async def query(
    query: QuerySchema, query_service: QueryService = Depends(get_query_service)
):
    return StreamingResponse(
        query_service.query(
            query.query,
            session_id=query.session_id,
            tenant_id=query.tenant_id,
            user_id=query.user_id,
            project_id=query.project_id,
        ),
        media_type="text/event-stream",
    )


@query_router.post("/api/query/resume")
async def resume_query(
    resume: ResumeSchema, query_service: QueryService = Depends(get_query_service)
):
    return StreamingResponse(
        query_service.resume(
            resume.session_id,
            resume.confirmed,
            tenant_id=resume.tenant_id,
            user_id=resume.user_id,
            project_id=resume.project_id,
        ),
        media_type="text/event-stream",
    )
