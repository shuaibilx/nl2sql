from typing import TypedDict

from app.entities.column_info import ColumnInfo
from app.entities.metric_info import MetricInfo
from app.entities.value_info import ValueInfo

# 为什么不用原来的entity，如ColumnInfo：原本的entity有id和table_id字段，给大模型不需要这些。state类型只保留大模型生成SQL需要的字段

class ColumnInfoState(TypedDict):
    name: str
    type: str
    role: str
    examples: list
    description: str
    alias: list[str]


class TableInfoState(TypedDict):
    name: str
    role: str
    description: str
    columns: list[ColumnInfoState]


class MetricInfoState(TypedDict):
    name: str
    description: str
    relevant_columns: list[str]
    alias: list[str]


class DateInfoState(TypedDict):
    date: str
    weekday: str
    quarter: str


class DBInfoState(TypedDict):
    dialect: str
    version: str


class DataAgentState(TypedDict):
    query: str  # 用户原始查询（保留不变，用于日志和审计）
    cleaned_query: str  # [改进] 清洗后的查询（纠错+去噪+规范化），后续节点使用此字段
    keywords: list[str]  # 用户查询的关键字（由 extract_keywords 节点提取）

    # [改进] 由 expand_keywords 节点统一扩展，替代原先 recall_column/recall_value/recall_metric 各自调用 LLM
    column_keywords: list[str]  # 字段维度扩展关键词（字段名、别名等）
    value_keywords: list[str]  # 取值维度扩展关键词（枚举值、分类名等）
    metric_keywords: list[str]  # 指标维度扩展关键词（业务指标名、别名等）

    # [改进] Send API 派发标识：由 send_to_recalls() 写入，recall_node() 读取
    recall_type: str  # "column" / "value" / "metric"

    retrieved_columns: list[ColumnInfo]  # 召回的字段信息
    retrieved_values: list[ValueInfo]  # 召回的值信息
    retrieved_metrics: list[MetricInfo]  # 召回的指标信息

    table_infos: list[TableInfoState]  # 表信息
    metric_infos: list[MetricInfoState]  # 指标信息

    date_info: DateInfoState  # 日期信息
    db_info: DBInfoState  # 数据库信息

    sql: str  # 生成的SQL

    error: str  # 验证SQL时的错误信息
    retry_count: int  # [改进] SQL校正重试计数器，validate_sql+correct_sql回环最多3次，防止死循环

    result_data: list  # SQL执行结果（execute_sql 写入，供评测和前端使用）
