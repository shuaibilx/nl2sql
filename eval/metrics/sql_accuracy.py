"""
[评测] SQL 准确率指标 — 执行准确率、模式匹配准确率、结果准确率

由于 SQL 写法不唯一，不使用精确匹配，而是：
1. 执行准确率：SQL 能否成功执行（EXPLAIN 通过）
2. 模式匹配：SQL 是否包含预期的关键模式
3. 结果准确率：查询结果是否与期望结果一致
"""
import re


def sql_execution_accuracy(sql: str, execution_error: str | None) -> bool:
    """SQL 执行准确率：SQL 能否成功执行"""
    return execution_error is None


def sql_pattern_match(sql: str, sql_pattern: str) -> bool:
    """SQL 模式匹配：SQL 是否包含预期的关键模式"""
    if not sql_pattern:
        return True
    try:
        return bool(re.search(sql_pattern, sql, re.IGNORECASE | re.DOTALL))
    except re.error:
        return sql_pattern.lower() in sql.lower()


def _extract_value(result):
    """从 SQL 结果集中提取实际值

    execute_sql 返回的是 list[dict]，如 [{'销售总额': 279159.5}]
    需要提取出 279159.5 这个数值用于比较
    """
    if result is None:
        return None
    if isinstance(result, (int, float, str)):
        return result
    if isinstance(result, list):
        if len(result) == 0:
            return None
        if len(result) == 1:
            # 单行结果：可能是 dict 或 list
            row = result[0]
            if isinstance(row, dict):
                values = list(row.values())
                if len(values) == 1:
                    return values[0]  # 单值聚合：提取数值
                return row  # 多列单行：保持 dict
            if isinstance(row, (list, tuple)):
                if len(row) == 1:
                    return row[0]  # 单值聚合
                return row
            return row
        # 多行结果：保持原样
        return result
    return result


def result_accuracy(actual_result, expected_result) -> bool:
    """结果准确率：查询结果是否与期望结果一致

    支持比较方式：
    - 单值比较：直接比较数值（容差 0.01）
    - 列表比较：转为集合比较（不考虑顺序）
    - 多列列表比较：转为 frozenset 比较
    - 自动从 SQL 结果集 (list[dict]) 中提取实际值
    """
    if expected_result is None:
        return True  # 无期望结果时，跳过比较

    if actual_result is None:
        return False

    # 从 SQL 结果集中提取实际值
    actual = _extract_value(actual_result)

    # 单值比较（最常见的聚合结果）
    if isinstance(expected_result, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(actual) - float(expected_result)) < 0.01

    # 字符串比较
    if isinstance(expected_result, str) and isinstance(actual, str):
        return expected_result.strip() == actual.strip()

    # 列表比较（集合方式，不考虑顺序）
    if isinstance(expected_result, list) and isinstance(actual, list):
        try:
            return set(map(str, expected_result)) == set(map(str, actual))
        except TypeError:
            return sorted(map(str, expected_result)) == sorted(map(str, actual))

    # 其他类型，转字符串比较
    return str(expected_result).strip() == str(actual).strip()


def sql_keyword_coverage(sql: str, expected_tables: list[str], expected_columns: list[str]) -> dict:
    """SQL 关键词覆盖：SQL 中是否包含了期望的表名和字段名"""
    sql_lower = sql.lower()

    tables_found = sum(1 for t in expected_tables if t.lower() in sql_lower)
    table_coverage = tables_found / len(expected_tables) if expected_tables else 1.0

    column_names = [c.split(".")[-1].lower() for c in expected_columns]
    columns_found = sum(1 for c in column_names if c in sql_lower)
    column_coverage = columns_found / len(column_names) if column_names else 1.0

    return {
        "table_coverage": table_coverage,
        "column_coverage": column_coverage,
        "tables_found": tables_found,
        "tables_total": len(expected_tables),
        "columns_found": columns_found,
        "columns_total": len(column_names),
    }


def evaluate_sql(
    sql: str,
    execution_error: str | None,
    sql_pattern: str = "",
    expected_tables: list[str] = None,
    expected_columns: list[str] = None,
    actual_result=None,
    expected_result=None,
) -> dict:
    """综合评估 SQL 质量"""
    expected_tables = expected_tables or []
    expected_columns = expected_columns or []

    result = {
        "execution_accuracy": sql_execution_accuracy(sql, execution_error),
        "pattern_match": sql_pattern_match(sql, sql_pattern),
        "result_accuracy": result_accuracy(actual_result, expected_result),
        **sql_keyword_coverage(sql, expected_tables, expected_columns),
    }

    # 记录实际结果和期望结果，便于报告展示
    result["actual_result"] = actual_result
    result["expected_result"] = expected_result

    return result
