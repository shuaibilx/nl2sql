import uuid

from fastapi import FastAPI, HTTPException, Request
from starlette.responses import Response

from app.api.routers.query_router import query_router
from app.conf.app_config import app_config
from app.core.cache_metrics import CONTENT_TYPE_LATEST, render_metrics
from app.core.context import request_id_ctx_var
from app.core.lifespan import lifespan

# 创建FastAPI应用，并注册生命周期函数
app = FastAPI(lifespan=lifespan) 

# 注册路由
app.include_router(query_router)


@app.get("/metrics")
async def metrics():
    if not app_config.monitoring.prometheus_enabled:
        raise HTTPException(status_code=404, detail="Metrics endpoint disabled")
    return Response(render_metrics(), media_type=CONTENT_TYPE_LATEST)


# 添加中间件，在每个请求中生成唯一的request_id
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # 调用路径函数之前
    request_id_ctx_var.set(uuid.uuid4())
    # 调用路径函数
    response = await call_next(request)
    # 调用路径函数之后
    return response


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
