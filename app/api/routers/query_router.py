from fastapi import APIRouter
from fastapi.params import Depends
from starlette.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.api.schemas.query_schema import QuerySchema, ResumeSchema  # [改进] 新增 ResumeSchema
from app.services.query_service import QueryService

query_router = APIRouter()


@query_router.post("/api/query")
async def query(
    query: QuerySchema, query_service: QueryService = Depends(get_query_service)
):
    return StreamingResponse(
        query_service.query(
            query.query,
            session_id=query.session_id  # [改进] 透传客户端 session_id，支持多轮对话
        ),
        media_type="text/event-stream"
    )


# [改进] 人机交互：SQL确认恢复端点
# 客户端收到 interrupt 事件后，POST 到此端点确认/取消 SQL 执行
@query_router.post("/api/query/resume")
async def resume_query(
    resume: ResumeSchema, query_service: QueryService = Depends(get_query_service)
):
    return StreamingResponse(
        query_service.resume(resume.session_id, resume.confirmed),
        media_type="text/event-stream"
    )
