"""
[评测] 期望结果计算脚本 — 连接 MySQL 数据库，执行 expected_sql，自动填充 expected_result

使用方式：
    python eval/compute_expected.py

前提：MySQL 数据仓库（dw 库）已启动并包含数据
"""
import asyncio
import sys
from pathlib import Path

import yaml
import aiomysql

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.conf.app_config import app_config


async def compute_all():
    """读取 test_cases.yaml，执行每条的 expected_sql，写回 expected_result"""
    # test_cases_path = Path(__file__).parent / "test_cases.yaml"
    test_cases_path = Path(__file__).parent / "test_cases_v3.yaml"
    with open(test_cases_path, "r", encoding="utf-8") as f:
        cases = yaml.safe_load(f)

    # 连接 MySQL 数据仓库
    conn = await aiomysql.connect(
        host=app_config.db_dw.host,
        port=app_config.db_dw.port,
        user=app_config.db_dw.user,
        password=app_config.db_dw.password,
        db=app_config.db_dw.database,
        charset="utf8mb4",
    )

    updated = 0
    skipped = 0
    errors = 0

    try:
        async with conn.cursor() as cursor:
            for case in cases:
                case_id = case.get("id", "unknown")
                expected = case.get("expected", {})
                sql = expected.get("expected_sql", "")

                if not sql:
                    print(f"[{case_id}] 跳过：无 expected_sql")
                    skipped += 1
                    continue

                try:
                    await cursor.execute(sql)
                    rows = await cursor.fetchall()

                    # 获取列名
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    # 格式化结果
                    if len(columns) == 1 and len(rows) == 1:
                        # 单值结果（如 SUM、COUNT、AVG）
                        result = rows[0][0]
                        # 转换为 Python 原生类型
                        if result is not None:
                            result = float(result) if isinstance(result, (int, float, type(None))) else str(result)
                    elif len(columns) == 1:
                        # 单列多行
                        result = [row[0] for row in rows]
                    else:
                        # 多列结果
                        result = [list(row) for row in rows]

                    expected["expected_result"] = result
                    print(f"[{case_id}] OK: {result}")
                    updated += 1

                except Exception as e:
                    print(f"[{case_id}] SQL 执行失败: {e}")
                    print(f"  SQL: {sql}")
                    errors += 1

    finally:
        conn.close()

    # 写回文件
    with open(test_cases_path, "w", encoding="utf-8") as f:
        yaml.dump(cases, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"\n{'='*50}")
    print(f"完成！更新: {updated}, 跳过: {skipped}, 错误: {errors}")
    print(f"已写入: {test_cases_path}")


if __name__ == "__main__":
    asyncio.run(compute_all())
