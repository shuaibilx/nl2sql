"""
[改进] 通用异步重试工具 — 基于 failsafe（Skyscanner），指数退避重试

适用场景：embedding 调用、向量库(Qdrant)查询、ES搜索、数据库操作等非LLM的关键外部调用。

注意：PyPI 包名为 pyfailsafe，但 import 名是 failsafe（不带 py 前缀）。
     on_retry 等回调不接受参数，无法获取具体异常和重试次数。

使用示例：
    from app.core.retry import retry_async
    result = await retry_async(some_async_func, arg1, arg2, operation_name="Qdrant搜索")
"""
from datetime import timedelta

from failsafe import Failsafe, RetryPolicy, Backoff
from app.core.log import logger


def create_retry_policy(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0,
                        operation_name: str = ""):
    """生成一个可复用的重试策略，采用指数退避：第1次等1s，第2次等2s，第3次等4s

    Args:
        max_retries: 最大重试次数（不含首次调用）
        delay: 首次重试等待秒数
        backoff: 退避因子，每次重试等待时间 = delay * backoff^(attempt-1)
        operation_name: 操作名称，用于日志标识
    """
    # [改进] 通过闭包捕获 operation_name，解决 on_retry 回调不接受参数的局限
    prefix = f"[{operation_name}] " if operation_name else ""
    return RetryPolicy(
        allowed_retries=max_retries,
        retriable_exceptions=(Exception,),
        backoff=Backoff(
            delay=timedelta(seconds=delay),
            max_delay=timedelta(seconds=delay * (backoff ** max_retries)),
            factor=backoff,
        ),
        on_retry=lambda: logger.warning(f"{prefix}重试中..."),
        on_retries_exhausted=lambda: logger.error(f"{prefix}已达最大重试次数({max_retries})"),
    )


# [改进] 便捷包装函数，一行代码即可为任意异步调用添加重试
async def retry_async(func, *args, max_retries: int = 3, delay: float = 1.0,
                      backoff: float = 2.0, operation_name: str = "",
                      circuit_breaker=None, **kwargs):
    """对异步函数 func(*args, **kwargs) 添加指数退避重试 + 可选熔断保护

    Args:
        func: 异步可调用对象
        *args: func 的位置参数
        max_retries: 最大重试次数
        delay: 首次重试等待秒数
        backoff: 退避因子
        operation_name: 操作名称（用于日志）
        circuit_breaker: 可选的 CircuitBreaker 实例，提供熔断保护
        **kwargs: func 的关键字参数

    Returns:
        func 的返回值

    Raises:
        RetriesExhausted: 所有重试耗尽后抛出
        CircuitOpenError: 熔断器打开时抛出（仅当提供了 circuit_breaker）
    """
    from app.core.circuit_breaker import CircuitOpenError

    policy = create_retry_policy(max_retries, delay, backoff, operation_name)

    if circuit_breaker:
        # 用熔断器包装：重试耗尽 → 熔断器记录失败 → 可能触发 open
        return await circuit_breaker.call(
            Failsafe(retry_policy=policy).run, func, *args, **kwargs
        )
    else:
        return await Failsafe(retry_policy=policy).run(func, *args, **kwargs)
