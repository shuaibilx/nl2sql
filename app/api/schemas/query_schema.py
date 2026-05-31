from pydantic import BaseModel


class QuerySchema(BaseModel):
    query: str
    session_id: str | None = None  # [改进] 可选，客户端传入可复用同一会话的 checkpoint 状态（断点续传/多轮对话）


# [改进] 人机交互：SQL确认的恢复请求体
class ResumeSchema(BaseModel):
    session_id: str       # 必须，指定要恢复的会话
    confirmed: bool       # True=确认执行SQL，False=取消执行SQL
