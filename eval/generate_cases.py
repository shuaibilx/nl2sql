"""
[评测] 测试用例生成器 — 读取 meta_config.yaml，用 LLM 自动生成评测数据集

完整流程：
    1. 读取元数据配置，构建表结构上下文
    2. 用 LLM 生成测试用例（含 expected_sql）
    3. 连接 MySQL 执行 expected_sql，自动填充 expected_result
    4. 写入 eval/test_cases.yaml

使用方式：
    python eval/generate_cases.py                    # 生成 25 条
    python eval/generate_cases.py --num 10           # 生成 10 条
    python eval/generate_cases.py --skip-db          # 跳过数据库验证
    python eval/generate_cases.py --output eval/test_cases_v2.yaml
"""
import asyncio
import argparse
import sys
from pathlib import Path

import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.llm import llm
from app.conf.meta_config import MetaConfig
from app.conf.app_config import app_config


def load_meta_config(config_path: str = r"E:\AAAlxlxx\Code\agent-nl2sql\conf\meta_config.yaml") -> MetaConfig:
    """加载元数据配置"""
    context = OmegaConf.load(config_path)
    schema = OmegaConf.structured(MetaConfig)
    return OmegaConf.to_object(OmegaConf.merge(schema, context))


def build_context_text(meta_config: MetaConfig) -> str:
    """将元数据配置转为文本描述，供 LLM 参考生成测试用例"""
    lines = []
    lines.append("## 数据库表结构")
    for table in meta_config.tables or []:
        lines.append(f"\n### 表: {table.name} ({table.role})")
        lines.append(f"描述: {table.description}")
        lines.append("字段:")
        for col in table.columns:
            lines.append(f"  - {col.name} ({col.role}): {col.description}")
            if col.alias:
                lines.append(f"    别名: {', '.join(col.alias)}")

    lines.append("\n## 业务指标")
    for metric in meta_config.metrics or []:
        lines.append(f"\n### 指标: {metric.name}")
        lines.append(f"描述: {metric.description}")
        lines.append(f"关联字段: {', '.join(metric.relevant_columns)}")
        if metric.alias:
            lines.append(f"别名: {', '.join(metric.alias)}")

    return "\n".join(lines)


GENERATION_PROMPT = """你是一位电商数据分析师，负责为 NL2SQL 系统生成评测测试用例。

## 数据模型
{context}

## 要求
请生成 {num_cases} 条测试用例，覆盖以下 4 种查询类型（每种约 {per_category} 条）：

1. **simple_aggregation**：单表聚合查询，如"统计销售总额"、"查询订单量"
2. **multi_table_join**：多表关联查询，如"统计浙江的销售总额"（需要 JOIN）
3. **conditional_filter**：带过滤条件的查询，如"查询华为手机的订单量"
4. **quality_query**：有质量问题的查询（错别字、口语化、模糊表述），如"帮我看看上个月消售额"

## 输出格式
严格按以下 YAML 格式输出，不要有任何其他内容：

```yaml
- id: TC001
  category: simple_aggregation
  query: "统计销售总额"
  expected:
    cleaned_query: "统计销售总额"
    keywords: ["销售总额", "统计"]
    columns: ["fact_order.order_amount"]
    values: []
    metrics: ["GMV"]
    tables: ["fact_order"]
    sql_pattern: "SELECT SUM.*order_amount"
    expected_sql: "SELECT SUM(order_amount) AS result FROM fact_order"

- id: TC002
  category: multi_table_join
  query: "统计浙江的销售总额"
  expected:
    cleaned_query: "统计浙江的销售总额"
    keywords: ["浙江", "销售总额", "统计"]
    columns: ["fact_order.order_amount", "dim_region.province"]
    values: ["浙江省"]
    metrics: ["GMV"]
    tables: ["fact_order", "dim_region"]
    sql_pattern: "SELECT SUM.*fact_order.*dim_region.*浙江"
    expected_sql: "SELECT SUM(f.order_amount) AS result FROM fact_order f JOIN dim_region r ON f.region_id = r.region_id WHERE r.province = '浙江省'"
```

## 注意
- columns 使用 "表名.字段名" 格式
- values 是期望召回的枚举值（来自 ES 索引）
- metrics 使用指标名称（如 GMV、AOV）
- sql_pattern 是 SQL 的正则表达式模式（宽松匹配）
- expected_sql 是可以直接在 MySQL 数据仓库执行的标准 SQL，必须语法正确
- expected_sql 中使用表别名（如 f、r、p、c、d）提高可读性
- quality_query 类型的 query 应包含错别字/口语化/冗余前缀等质量问题，但 expected_sql 必须是正确的
- 每条用例的 id 按 TC001, TC002, ... 递增
- expected_sql 的结果列名统一用 AS result（单值聚合时）或多列名（分组查询时）

请直接输出 YAML，不要有任何解释文字。
"""


async def _call_llm_with_retry(prompt: str, max_retries: int = 3) -> str | None:
    """调用 LLM，带重试和超时处理"""
    import asyncio
    for attempt in range(1, max_retries + 1):
        try:
            result = await asyncio.wait_for(llm.ainvoke(prompt), timeout=120)
            return result.content.strip()
        except asyncio.TimeoutError:
            print(f"    LLM 调用超时（第 {attempt} 次）")
            if attempt < max_retries:
                print(f"    等待 5 秒后重试...")
                await asyncio.sleep(5)
        except Exception as e:
            print(f"    LLM 调用失败（第 {attempt} 次）: {e}")
            if attempt < max_retries:
                await asyncio.sleep(5)
    return None


def _parse_yaml_cases(content: str) -> list[dict] | None:
    """从 LLM 输出中提取 YAML 测试用例"""
    if "```yaml" in content:
        content = content.split("```yaml")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()

    try:
        cases = yaml.safe_load(content)
        assert isinstance(cases, list), "生成结果不是列表"
        assert len(cases) > 0, "生成结果为空"
        return cases
    except Exception as e:
        print(f"    YAML 解析失败: {e}")
        print(f"    原始输出: {content[:300]}")
        return None


async def generate_with_llm(num_cases: int = 25, batch_size: int = 5) -> list[dict] | None:
    """用 LLM 生成测试用例（分批生成，避免超时）"""
    meta_config = load_meta_config()
    context_text = build_context_text(meta_config)

    per_category = batch_size // 4 if batch_size >= 4 else 1

    # 分批生成
    all_cases = []
    batch_num = (num_cases + batch_size - 1) // batch_size
    case_counter = 1

    print(f"[1/3] 正在用 LLM 生成 {num_cases} 条测试用例（每批 {batch_size} 条，共 {batch_num} 批）...")

    for batch_idx in range(batch_num):
        remaining = num_cases - len(all_cases)
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break

        print(f"  批次 {batch_idx + 1}/{batch_num}: 生成 {this_batch} 条...")

        prompt = GENERATION_PROMPT.format(
            context=context_text,
            num_cases=this_batch,
            per_category=max(1, this_batch // 4),
        )

        content = await _call_llm_with_retry(prompt)
        if content is None:
            print(f"  批次 {batch_idx + 1} 失败，跳过")
            continue

        cases = _parse_yaml_cases(content)
        if cases is None:
            print(f"  批次 {batch_idx + 1} 解析失败，跳过")
            continue

        # 重新编号，确保连续
        for case in cases:
            case["id"] = f"TC{case_counter:03d}"
            case_counter += 1

        all_cases.extend(cases)
        print(f"  批次 {batch_idx + 1} 成功：{len(cases)} 条（累计 {len(all_cases)} 条）")

    if not all_cases:
        print("  所有批次均失败")
        return None

    print(f"  成功生成 {len(all_cases)} 条测试用例")
    return all_cases


async def compute_expected_results(cases: list[dict]) -> tuple[int, int]:
    """连接 MySQL，执行 expected_sql，填充 expected_result

    Returns:
        (成功数, 失败数)
    """
    import aiomysql

    print(f"[2/3] 连接 MySQL 执行 expected_sql...")

    conn = await aiomysql.connect(
        host=app_config.db_dw.host,
        port=app_config.db_dw.port,
        user=app_config.db_dw.user,
        password=app_config.db_dw.password,
        db=app_config.db_dw.database,
        charset="utf8mb4",
    )

    success = 0
    errors = 0

    try:
        async with conn.cursor() as cursor:
            for case in cases:
                case_id = case.get("id", "unknown")
                expected = case.get("expected", {})
                sql = expected.get("expected_sql", "")

                if not sql:
                    print(f"  [{case_id}] 跳过：无 expected_sql")
                    continue

                try:
                    await cursor.execute(sql)
                    rows = await cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []

                    if len(columns) == 1 and len(rows) == 1:
                        result = rows[0][0]
                        if result is not None:
                            result = float(result) if isinstance(result, (int, float, type(None))) else str(result)
                    elif len(columns) == 1:
                        result = [row[0] for row in rows]
                    else:
                        result = [list(row) for row in rows]

                    expected["expected_result"] = result
                    print(f"  [{case_id}] OK: {result}")
                    success += 1

                except Exception as e:
                    print(f"  [{case_id}] SQL 执行失败: {e}")
                    print(f"    SQL: {sql}")
                    expected["expected_result"] = None
                    errors += 1
    finally:
        conn.close()

    return success, errors


async def generate_cases(num_cases: int = 25, batch_size: int = 5, skip_db: bool = False, output: str = None):
    """完整流程：生成 → 验证 → 写入"""
    # Step 1: LLM 生成（分批，避免超时）
    cases = await generate_with_llm(num_cases, batch_size=batch_size)
    if cases is None:
        return

    # Step 2: 数据库验证
    if not skip_db:
        success, errors = await compute_expected_results(cases)
        print(f"  数据库验证: 成功 {success}, 失败 {errors}")
    else:
        print("[2/3] 跳过数据库验证")

    # Step 3: 写入文件
    output_path = Path(output) if output else Path(__file__).parent / "test_cases_v3.yaml"
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cases, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    print(f"[3/3] 已写入: {output_path}")
    print(f"\n{'='*50}")
    print(f"完成！共 {len(cases)} 条测试用例")
    if not skip_db:
        print(f"数据库验证: 成功 {success}, 失败 {errors}")
    print("请人工审核后，运行: python eval/run_eval.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NL2SQL 测试用例生成器")
    parser.add_argument("--num", type=int, default=50, help="生成用例数量（默认 25）")
    parser.add_argument("--batch", type=int, default=5, help="每批生成数量（默认 5，避免超时）")
    parser.add_argument("--skip-db", action="store_true", help="跳过数据库验证")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    asyncio.run(generate_cases(num_cases=args.num, batch_size=args.batch, skip_db=args.skip_db, output=args.output))
