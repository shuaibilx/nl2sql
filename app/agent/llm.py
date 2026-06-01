from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_deepseek import ChatDeepSeek

from app.conf.app_config import app_config
from app.core.circuit_breaker import CircuitOpenError
from app.core.circuit_manager import circuit_manager
from app.core.log import logger

llm = init_chat_model(model=app_config.llm.model_name,
                      model_provider="openai",
                      api_key=app_config.llm.api_key,
                      base_url=app_config.llm.base_url,
                      temperature=0,
                      max_retries=3,  # [改进] 网络错误（TimeoutError/ConnectionError）自动重试3次
                      timeout=60)     # [改进] 单次LLM调用60秒超时，防止长时间挂起

llm_flash = ChatTongyi(model="qwen3.7-max-preview",
                      max_retries=3  # [改进] 网络错误（TimeoutError/ConnectionError）自动重试3次
)      # [改进] 单次LLM调用60秒超时，防止长时间挂起

# [降级] 备用模型：主模型不可用时自动切换
llm_fallback = ChatDeepSeek(model=app_config.llm_fallback.model_name,
                            temperature=0,
                            max_retries=2,
                            timeout=60)


async def call_with_fallback(prompt, parser, invoke_args: dict, primary_llm=None, label: str = ""):
    """
    带降级 + 熔断的 LLM 调用：先用主模型，失败后自动切换备用模型。
    两个模型各有独立熔断器，连续失败后直接跳过，不再等待重试超时。
    """
    if primary_llm is None:
        primary_llm = llm

    cb_primary = circuit_manager.get("LLM-primary")
    cb_fallback = circuit_manager.get("LLM-fallback")

    # 主模型：先检查熔断
    if not cb_primary.is_open:
        try:
            result = await cb_primary.call(
                (prompt | primary_llm | parser).ainvoke, invoke_args
            )
            return result
        except (Exception, CircuitOpenError) as primary_err:
            logger.warning(f"[{label}] 主模型调用失败，降级到备用模型: {primary_err}")
    else:
        logger.warning(f"[{label}] 主模型熔断中，直接用备用模型")

    # 备用模型
    if not cb_fallback.is_open:
        try:
            result = await cb_fallback.call(
                (prompt | llm_fallback | parser).ainvoke, invoke_args
            )
            logger.info(f"[{label}] 备用模型调用成功")
            return result
        except (Exception, CircuitOpenError) as fallback_err:
            logger.error(f"[{label}] 备用模型也失败: {fallback_err}")
            raise fallback_err
    else:
        raise CircuitOpenError(f"[{label}] 主备模型均熔断")


if __name__ == '__main__':
    # for chunk in llm_flash.stream("What is the meaning of life?"):
    # for chunk in llm_fallback.stream("What is the meaning of life?"):
    for chunk in llm.stream("What is the meaning of life?"):
        print(chunk.text)
