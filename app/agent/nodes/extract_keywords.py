import jieba.analyse
from langgraph.runtime import Runtime

from app.agent.context import DataAgentContext
from app.agent.state import DataAgentState
from app.core.log import logger


async def extract_keywords(state: DataAgentState, runtime: Runtime[DataAgentContext]):
    writer = runtime.stream_writer
    writer({"type": "progress", "step": "抽取关键字", "status": "running"})

    # [改进] 使用清洗后的查询进行分词，原始 query 保留用于日志和审计
    query = state.get("cleaned_query") or state["query"]

    try:
        # 对查询进行分词，只提取指定词性的词
        allow_pos = (
            "n",  # 名词: 数据、服务器、表格
            "nr",  # 人名: 张三、李四
            "ns",  # 地名: 北京、上海
            "nt",  # 机构团体名: 政府、学校、某公司
            "nz",  # 其他专有名词: Unicode、哈希算法、诺贝尔奖
            "v",  # 动词: 运行、开发
            "vn",  # 名动词: 工作、研究
            "a",  # 形容词: 美丽、快速
            "an",  # 名形词: 难度、合法性、复杂度
            "eng",  # 英文
            "i",  # 成语
            "l",  # 常用固定短语
        )

        keywords = jieba.analyse.extract_tags(query, allowPOS=allow_pos)
        keywords = list(set(keywords + [query]))

        writer({"type": "progress", "step": "抽取关键字", "status": "success"})
        logger.info(f"抽取关键字: {keywords}")
        return {"keywords": keywords}

    except Exception as e:
        # [降级] jieba 分词失败时，用原始查询按空格拆分作为兜底关键词
        logger.warning(f"抽取关键字异常，降级为空格分词: {e}")
        fallback_keywords = list(set(query.split()))
        if not fallback_keywords:
            fallback_keywords = [query]
        writer({"type": "progress", "step": "抽取关键字", "status": "fallback"})
        logger.info(f"降级关键字: {fallback_keywords}")
        return {"keywords": fallback_keywords}
