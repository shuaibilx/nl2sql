"""
[评测] 主评测脚本 — 加载测试用例，运行 pipeline，计算指标，生成报告

使用方式：
    python eval/run_eval.py
    python eval/run_eval.py --cases eval/test_cases.yaml --report eval/reports/

输出文件（每次运行按时间戳生成，不覆盖）：
    eval/reports/eval_run_YYYYMMDD_HHMMSS.log     — 运行时详细日志
    eval/reports/eval_report_YYYYMMDD_HHMMSS.md   — Markdown 评测报告
    eval/reports/metrics_log_YYYYMMDD_HHMMSS.yaml — 问题级指标日志
"""
import asyncio
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 项目根目录和 eval 目录
PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent

from langgraph.types import Command
from app.agent.checkpoint import init_checkpointer
from app.agent.graph import setup_graph, get_graph
from app.agent.state import DataAgentState
from app.agent.context import DataAgentContext
from app.core.context import request_id_ctx_var
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import meta_mysql_client_manager, dw_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository

from eval.metrics.retrieval import evaluate_retrieval
from eval.metrics.sql_accuracy import evaluate_sql
from eval.metrics.latency import LatencyTracker


def setup_logging(output_dir: Path) -> Path:
    """配置运行时日志，输出到控制台 + 文件（按时间戳命名，不覆盖）"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f"eval_run_{timestamp}.log"

    logger = logging.getLogger("eval")
    logger.setLevel(logging.DEBUG)

    # 文件 handler — 详细日志
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    # 控制台 handler — 精简输出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return log_path


def load_test_cases(path: str) -> list[dict]:
    """加载测试用例"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_actual_from_chunk(chunk: dict, actual: dict):
    """从 updates 事件中提取各节点的实际输出

    注意：retrieved_* 字段从 recall_node 捕获（召回阶段）
    filtered_* 字段从 filter_table/filter_metric 捕获（过滤后）
    评测对比应使用 filtered_*（过滤后的最终结果）
    """
    for node_name in ["query_cleanup", "extract_keywords", "expand_keywords",
                      "recall_node", "filter_table", "filter_metric",
                      "generate_sql", "execute_sql"]:
        if node_name not in chunk:
            continue
        node_output = chunk[node_name]
        if not isinstance(node_output, dict):
            continue

        # query_cleanup
        if "cleaned_query" in node_output:
            actual["cleaned_query"] = node_output["cleaned_query"]

        # extract_keywords
        if "keywords" in node_output:
            actual["keywords"] = node_output["keywords"]

        # expand_keywords
        if "column_keywords" in node_output:
            actual["column_keywords"] = node_output["column_keywords"]
        if "value_keywords" in node_output:
            actual["value_keywords"] = node_output["value_keywords"]
        if "metric_keywords" in node_output:
            actual["metric_keywords"] = node_output["metric_keywords"]

        # recall_node — 召回阶段原始结果
        if "retrieved_columns" in node_output:
            actual["retrieved_columns"] = [
                c.id if hasattr(c, "id") else c.get("id", "")
                for c in node_output["retrieved_columns"]
            ]
        if "retrieved_values" in node_output:
            actual["retrieved_values"] = [
                v.id if hasattr(v, "id") else v.get("id", "")
                for v in node_output["retrieved_values"]
            ]
        if "retrieved_metrics" in node_output:
            actual["retrieved_metrics"] = [
                m.id if hasattr(m, "id") else m.get("id", "")
                for m in node_output["retrieved_metrics"]
            ]

        # filter_table — 过滤后的表和字段（用于评测对比）
        if "table_infos" in node_output:
            filtered_columns = []
            for table_info in node_output["table_infos"]:
                table_name = table_info.get("name", "")
                for col in table_info.get("columns", []):
                    col_name = col.get("name", "")
                    filtered_columns.append(f"{table_name}.{col_name}")
            actual["filtered_columns"] = filtered_columns

        # filter_metric — 过滤后的指标（用于评测对比）
        if "metric_infos" in node_output:
            actual["filtered_metrics"] = [
                m.get("name", "") for m in node_output["metric_infos"]
            ]

        # generate_sql
        if "sql" in node_output:
            actual["sql"] = node_output["sql"]

        # execute_sql
        if "result_data" in node_output:
            actual["result_data"] = node_output["result_data"]


async def collect_stream(graph, input_, context, config, tracker, actual):
    """收集一次 astream 的所有事件，返回是否遇到 interrupt"""
    logger = logging.getLogger("eval")
    hit_interrupt = False
    async for mode, chunk in graph.astream(
        input=input_, context=context, config=config,
        stream_mode=["custom", "updates"]
    ):
        if mode == "custom" and isinstance(chunk, dict):
            tracker.on_progress(chunk)
            if chunk.get("type") == "interrupt":
                hit_interrupt = True
        elif mode == "updates" and isinstance(chunk, dict):
            if "__interrupt__" in chunk:
                hit_interrupt = True
            # 记录每个节点的更新，便于定位错误
            for node_name, node_output in chunk.items():
                if node_name.startswith("__"):
                    continue
                if node_output is None:
                    logger.debug(f"    [{node_name}] → None")
                elif isinstance(node_output, dict):
                    logger.debug(f"    [{node_name}] → keys: {list(node_output.keys())}")
            extract_actual_from_chunk(chunk, actual)
    return hit_interrupt


async def run_single_case(
    case: dict,
    context: DataAgentContext,
    case_index: int,
) -> dict:
    """运行单条测试用例，收集各节点输出

    Returns:
        包含实际输出和评测指标的字典
    """
    logger = logging.getLogger("eval")
    query = case["query"]
    case_id = case.get("id", f"TC{case_index:03d}")
    request_id_ctx_var.set(f"eval-{case_id}")
    logger.info(f"\n[{case_id}] {query}")

    tracker = LatencyTracker()
    state = DataAgentState(query=query, cleaned_query="", retry_count=0)
    config = {"configurable": {"thread_id": f"eval-{case_id}"}}

    actual = {
        "cleaned_query": "",
        "keywords": [],
        "column_keywords": [],
        "value_keywords": [],
        "metric_keywords": [],
        "retrieved_columns": [],   # 召回阶段原始结果
        "retrieved_values": [],
        "retrieved_metrics": [],
        "filtered_columns": [],    # 过滤后的字段（filter_table 输出）
        "filtered_metrics": [],    # 过滤后的指标（filter_metric 输出）
        "sql": "",
        "execution_error": None,
        "result_data": None,
    }

    tracker.start()
    graph = get_graph()

    try:
        # 第一次运行：从初始 state 开始
        hit_interrupt = await collect_stream(graph, state, context, config, tracker, actual)

        # 如果遇到 interrupt（SQL 确认），自动 resume
        if hit_interrupt:
            logger.debug(f"  [interrupt] 自动确认执行 SQL")
            await collect_stream(graph, Command(resume=True), context, config, tracker, actual)

    except Exception as e:
        import traceback
        actual["execution_error"] = str(e)
        logger.error(f"  [error] {type(e).__name__}: {str(e)[:200]}")
        logger.debug(f"  [traceback]\n{traceback.format_exc()}")

    tracker.stop()

    # 计算各层指标
    expected = case.get("expected", {})

    retrieval_metrics = evaluate_retrieval(
        retrieved_columns=actual["retrieved_columns"],
        retrieved_values=actual["retrieved_values"],
        retrieved_metrics=actual["retrieved_metrics"],
        expected_columns=expected.get("columns", []),
        expected_values=expected.get("values", []),
        expected_metrics=expected.get("metrics", []),
    )

    sql_metrics = evaluate_sql(
        sql=actual["sql"],
        execution_error=actual.get("execution_error"),
        sql_pattern=expected.get("sql_pattern", ""),
        expected_tables=expected.get("tables", []),
        expected_columns=expected.get("columns", []),
        actual_result=actual.get("result_data"),
        expected_result=expected.get("expected_result"),
    )

    latency_metrics = tracker.summary()

    result = {
        "case_id": case_id,
        "category": case.get("category", "unknown"),
        "query": query,
        "actual": actual,
        "expected": expected,
        "metrics": {
            "retrieval": retrieval_metrics,
            "sql": sql_metrics,
            "latency": latency_metrics,
        },
    }

    # 结构化日志：所有指标的期望 vs 实际对比
    expected = case.get("expected", {})
    expected_sql = expected.get("expected_sql", "")
    actual_sql = actual.get("sql", "")
    expected_result = expected.get("expected_result")
    actual_result = actual.get("result_data")

    sep = "  " + "-" * 50
    logger.info(sep)
    logger.info(f"  查询清洗: {actual['cleaned_query'][:50]}")
    logger.info(f"  召回统计: 字段={len(actual['retrieved_columns'])}, "
                f"值={len(actual['retrieved_values'])}, "
                f"指标={len(actual['retrieved_metrics'])}")

    # SQL 对比
    logger.info(f"  [SQL]")
    logger.info(f"    期望: {expected_sql}")
    logger.info(f"    实际: {actual_sql or '(无)'}")
    logger.info(f"    执行: {'OK' if sql_metrics['execution_accuracy'] else 'FAIL'}  "
                f"模式匹配: {'OK' if sql_metrics['pattern_match'] else 'MISS'}")

    # 结果对比
    logger.info(f"  [结果]")
    logger.info(f"    期望: {_fmt_result(expected_result)}")
    logger.info(f"    实际: {_fmt_result(actual_result)}")
    logger.info(f"    准确: {'OK' if sql_metrics['result_accuracy'] else 'FAIL'}")

    # 检索对比（使用过滤后的最终结果，而非召回阶段原始结果）
    logger.info(f"  [检索]（过滤后）")
    for dim, exp_key, act_key in [("字段", "columns", "filtered_columns"),
                                   ("值", "values", "retrieved_values"),
                                   ("指标", "metrics", "filtered_metrics")]:
        exp_ids = sorted(str(x) for x in expected.get(exp_key, []))
        act_ids = sorted(str(x) for x in actual.get(act_key, []))
        missing = sorted(set(exp_ids) - set(act_ids))
        logger.info(f"    {dim}期望: {', '.join(exp_ids) or '(无)'}")
        logger.info(f"    {dim}实际: {', '.join(act_ids) or '(无)'}")
        if missing:
            logger.info(f"    {dim}缺失: {', '.join(missing)}")
        # 注：召回结果中的多余字段由下游节点补充（如主外键、过滤后保留），属正常行为

    # 延迟
    logger.info(f"  [延迟] {latency_metrics['total_latency_s']:.2f}s")

    # 错误信息
    if actual.get("execution_error"):
        logger.info(f"  [错误] {actual['execution_error'][:200]}")

    # 总结行
    exec_status = 'OK' if sql_metrics['execution_accuracy'] else 'FAIL'
    result_status = 'OK' if sql_metrics['result_accuracy'] else 'FAIL'
    logger.info(f"  >>> Recall={retrieval_metrics['overall_recall']:.2f} "
                f"SQL={exec_status} 结果={result_status} "
                f"延迟={latency_metrics['total_latency_s']:.2f}s")
    logger.info(sep)

    return result


def _extract_value_for_log(result):
    """从 SQL 结果集中提取实际值，用于 YAML 日志"""
    if result is None:
        return None
    if isinstance(result, (int, float, str)):
        return result
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        values = list(result[0].values())
        if len(values) == 1:
            v = values[0]
            return float(v) if isinstance(v, (int, float)) else str(v)
    return result


def _fmt_result(result) -> str:
    """格式化结果用于报告展示

    自动处理 SQL 结果集格式：[{'列名': 值}] → 提取值展示
    """
    if result is None:
        return "(无)"

    # 处理 SQL 结果集 [dict]
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], dict):
        values = list(result[0].values())
        if len(values) == 1:
            # 单值聚合：提取数值
            v = values[0]
            return f"{v:,.2f}" if isinstance(v, float) else str(v)
        # 多列单行
        return str(result[0])

    if isinstance(result, float):
        return f"{result:,.2f}"
    if isinstance(result, (int, str)):
        return str(result)
    if isinstance(result, list):
        if len(result) == 0:
            return "(空)"
        if len(result) <= 10:
            return str(result)
        return f"{str(result[:10])}... (共 {len(result)} 行)"
    return str(result)[:200]


def generate_report(results: list[dict], output_dir: str) -> str:
    """生成评测报告

    Returns:
        报告文件路径
    """
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    overall_recall = avg([r["metrics"]["retrieval"]["overall_recall"] for r in results])
    overall_exec_accuracy = avg([1.0 if r["metrics"]["sql"]["execution_accuracy"] else 0.0 for r in results])
    overall_pattern_match = avg([1.0 if r["metrics"]["sql"]["pattern_match"] else 0.0 for r in results])
    overall_result_accuracy = avg([1.0 if r["metrics"]["sql"]["result_accuracy"] else 0.0 for r in results])
    overall_latency = avg([r["metrics"]["latency"]["total_latency_s"] for r in results])

    lines = []
    lines.append(f"# NL2SQL 评测报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试用例数**: {len(results)}")
    lines.append(f"")

    lines.append(f"## 总体指标")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|---|---|")
    lines.append(f"| 检索 Recall（平均） | {overall_recall:.2%} |")
    lines.append(f"| SQL 执行准确率 | {overall_exec_accuracy:.2%} |")
    lines.append(f"| SQL 模式匹配率 | {overall_pattern_match:.2%} |")
    lines.append(f"| 结果准确率 | {overall_result_accuracy:.2%} |")
    lines.append(f"| 平均延迟 | {overall_latency:.2f}s |")
    lines.append(f"")

    lines.append(f"## 分类别统计")
    lines.append(f"")
    for cat, cases in categories.items():
        cat_recall = avg([c["metrics"]["retrieval"]["overall_recall"] for c in cases])
        cat_exec = avg([1.0 if c["metrics"]["sql"]["execution_accuracy"] else 0.0 for c in cases])
        cat_latency = avg([c["metrics"]["latency"]["total_latency_s"] for c in cases])
        lines.append(f"### {cat}")
        lines.append(f"- 用例数: {len(cases)}")
        lines.append(f"- 检索 Recall: {cat_recall:.2%}")
        lines.append(f"- SQL 执行准确率: {cat_exec:.2%}")
        lines.append(f"- 平均延迟: {cat_latency:.2f}s")
        lines.append(f"")

    lines.append(f"## 详细结果")
    lines.append(f"")
    lines.append(f"### 总览")
    lines.append(f"")
    lines.append(f"| 编号 | 查询 | SQL执行 | 结果准确 | 检索Recall | 延迟 |")
    lines.append(f"|---|---|---|---|---|---|")
    for r in results:
        m = r["metrics"]
        exec_mark = "OK" if m['sql']['execution_accuracy'] else "FAIL"
        result_mark = "OK" if m['sql']['result_accuracy'] else "FAIL"
        recall = f"{m['retrieval']['overall_recall']:.2f}"
        latency = f"{m['latency']['total_latency_s']:.1f}s"
        lines.append(f"| {r['case_id']} | {r['query'][:20]} | {exec_mark} | {result_mark} | {recall} | {latency} |")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    for r in results:
        case_id = r["case_id"]
        query = r["query"]
        m = r["metrics"]
        a = r["actual"]
        e = r["expected"]

        # 判定总体状态
        exec_ok = m['sql']['execution_accuracy']
        result_ok = m['sql']['result_accuracy']
        if exec_ok and result_ok:
            status = "PASS"
        elif exec_ok:
            status = "PARTIAL"
        else:
            status = "FAIL"

        lines.append(f"### {case_id}: {query}  [{status}]")
        lines.append(f"")

        # 指标表格
        lines.append(f"| 维度 | 指标 | 值 |")
        lines.append(f"|---|---|---|")
        lines.append(f"| 检索 | 字段 Recall | {m['retrieval']['column_recall@k']:.2f} |")
        lines.append(f"| 检索 | 值 Recall | {m['retrieval']['value_recall@k']:.2f} |")
        lines.append(f"| 检索 | 指标 Recall | {m['retrieval']['metric_recall@k']:.2f} |")
        lines.append(f"| SQL | 执行准确率 | {'OK' if exec_ok else 'FAIL'} |")
        lines.append(f"| SQL | 模式匹配 | {'OK' if m['sql']['pattern_match'] else 'MISS'} |")
        lines.append(f"| SQL | 结果准确率 | {'OK' if result_ok else 'FAIL'} |")
        lines.append(f"| SQL | 表覆盖率 | {m['sql']['table_coverage']:.2f} |")
        lines.append(f"| SQL | 字段覆盖率 | {m['sql']['column_coverage']:.2f} |")
        lines.append(f"| 性能 | 总延迟 | {m['latency']['total_latency_s']:.2f}s |")
        lines.append(f"")

        # SQL 对比
        lines.append(f"**SQL 对比**:")
        lines.append(f"```")
        lines.append(f"期望: {e.get('expected_sql', 'N/A')}")
        lines.append(f"实际: {a.get('sql', 'N/A') or '(无)'}")
        lines.append(f"```")
        lines.append(f"")

        # 结果对比
        expected_result = e.get("expected_result")
        actual_result = a.get("result_data")
        lines.append(f"**结果对比**:")
        lines.append(f"```")
        lines.append(f"期望: {_fmt_result(expected_result)}")
        lines.append(f"实际: {_fmt_result(actual_result)}")
        lines.append(f"```")
        lines.append(f"")

        # 检索对比（使用过滤后的最终结果）
        lines.append(f"**检索对比**（过滤后）:")
        lines.append(f"| 维度 | 期望 | 实际 | 缺失 |")
        lines.append(f"|---|---|---|---|")
        for dim, exp_key, act_key in [
            ("字段", "columns", "filtered_columns"),
            ("值", "values", "retrieved_values"),
            ("指标", "metrics", "filtered_metrics"),
        ]:
            exp_ids = sorted(str(x) for x in e.get(exp_key, []))
            act_ids = sorted(str(x) for x in a.get(act_key, []))
            missing = sorted(set(exp_ids) - set(act_ids))
            lines.append(f"| {dim} | {', '.join(exp_ids) or '-'} | {', '.join(act_ids) or '-'} | {', '.join(missing) or '-'} |")
        lines.append(f"")

        # 错误信息
        if a.get("execution_error"):
            lines.append(f"**错误**: `{a['execution_error'][:200]}`")
            lines.append(f"")

        lines.append(f"---")
        lines.append(f"")

    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(report_path)


def generate_metrics_log(results: list[dict], output_dir: str) -> str:
    """生成详细的问题级指标日志文件（YAML 格式）

    每个问题记录：query、actual SQL、actual/expected result、所有指标值
    末尾附总体汇总指标
    """
    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # 构建每条记录
    entries = []
    for r in results:
        m = r["metrics"]
        a = r["actual"]
        e = r["expected"]

        # 检索对比：期望 vs 过滤后实际 vs 缺失
        retrieval_comparison = {}
        for dim, exp_key, act_key in [("columns", "columns", "filtered_columns"),
                                       ("values", "values", "retrieved_values"),
                                       ("metrics", "metrics", "filtered_metrics")]:
            exp_ids = set(str(x) for x in e.get(exp_key, []))
            act_ids = set(str(x) for x in a.get(act_key, []))
            retrieval_comparison[dim] = {
                "expected": sorted(exp_ids),
                "actual": sorted(act_ids),
                "missing": sorted(exp_ids - act_ids),
                "extra": sorted(act_ids - exp_ids),
            }

        entry = {
            "case_id": r["case_id"],
            "category": r["category"],
            "query": r["query"],
            "cleaned_query": a.get("cleaned_query", ""),
            "expected_sql": e.get("expected_sql", ""),
            "actual_sql": a.get("sql", ""),
            "expected_result": e.get("expected_result"),
            "actual_result": a.get("result_data"),
            "actual_value": _extract_value_for_log(a.get("result_data")),
            "result_match": m["sql"]["result_accuracy"],
            "execution_error": a.get("execution_error"),
            "retrieval_comparison": retrieval_comparison,
            "retrieval_metrics": {
                "column_recall@k": round(m["retrieval"]["column_recall@k"], 4),
                "column_precision@k": round(m["retrieval"]["column_precision@k"], 4),
                "column_mrr": round(m["retrieval"]["column_mrr"], 4),
                "value_recall@k": round(m["retrieval"]["value_recall@k"], 4),
                "value_precision@k": round(m["retrieval"]["value_precision@k"], 4),
                "value_mrr": round(m["retrieval"]["value_mrr"], 4),
                "metric_recall@k": round(m["retrieval"]["metric_recall@k"], 4),
                "metric_precision@k": round(m["retrieval"]["metric_precision@k"], 4),
                "metric_mrr": round(m["retrieval"]["metric_mrr"], 4),
                "overall_recall": round(m["retrieval"]["overall_recall"], 4),
            },
            "sql_metrics": {
                "execution_accuracy": m["sql"]["execution_accuracy"],
                "pattern_match": m["sql"]["pattern_match"],
                "result_accuracy": m["sql"]["result_accuracy"],
                "table_coverage": round(m["sql"]["table_coverage"], 4),
                "column_coverage": round(m["sql"]["column_coverage"], 4),
            },
            "latency_metrics": {k: round(v, 4) for k, v in m["latency"].items()},
        }
        entries.append(entry)

    # 计算总体汇总
    overall = {
        "total_cases": len(results),
        "passed": sum(1 for r in results if r["metrics"]["sql"]["execution_accuracy"] and r["metrics"]["sql"]["result_accuracy"]),
        "failed": sum(1 for r in results if not r["metrics"]["sql"]["execution_accuracy"] or not r["metrics"]["sql"]["result_accuracy"]),

        "retrieval": {
            "avg_column_recall": round(avg([r["metrics"]["retrieval"]["column_recall@k"] for r in results]), 4),
            "avg_column_precision": round(avg([r["metrics"]["retrieval"]["column_precision@k"] for r in results]), 4),
            "avg_column_mrr": round(avg([r["metrics"]["retrieval"]["column_mrr"] for r in results]), 4),
            "avg_value_recall": round(avg([r["metrics"]["retrieval"]["value_recall@k"] for r in results]), 4),
            "avg_value_precision": round(avg([r["metrics"]["retrieval"]["value_precision@k"] for r in results]), 4),
            "avg_value_mrr": round(avg([r["metrics"]["retrieval"]["value_mrr"] for r in results]), 4),
            "avg_metric_recall": round(avg([r["metrics"]["retrieval"]["metric_recall@k"] for r in results]), 4),
            "avg_metric_precision": round(avg([r["metrics"]["retrieval"]["metric_precision@k"] for r in results]), 4),
            "avg_metric_mrr": round(avg([r["metrics"]["retrieval"]["metric_mrr"] for r in results]), 4),
            "avg_overall_recall": round(avg([r["metrics"]["retrieval"]["overall_recall"] for r in results]), 4),
        },

        "sql": {
            "execution_accuracy_rate": round(avg([1.0 if r["metrics"]["sql"]["execution_accuracy"] else 0.0 for r in results]), 4),
            "pattern_match_rate": round(avg([1.0 if r["metrics"]["sql"]["pattern_match"] else 0.0 for r in results]), 4),
            "result_accuracy_rate": round(avg([1.0 if r["metrics"]["sql"]["result_accuracy"] else 0.0 for r in results]), 4),
            "avg_table_coverage": round(avg([r["metrics"]["sql"]["table_coverage"] for r in results]), 4),
            "avg_column_coverage": round(avg([r["metrics"]["sql"]["column_coverage"] for r in results]), 4),
        },

        "latency": {
            "avg_total_latency_s": round(avg([r["metrics"]["latency"]["total_latency_s"] for r in results]), 4),
            "max_latency_s": round(max([r["metrics"]["latency"]["total_latency_s"] for r in results]), 4),
            "min_latency_s": round(min([r["metrics"]["latency"]["total_latency_s"] for r in results]), 4),
        },

        "by_category": {},
    }

    # 分类别汇总
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)

    for cat, cases in categories.items():
        overall["by_category"][cat] = {
            "count": len(cases),
            "avg_recall": round(avg([c["metrics"]["retrieval"]["overall_recall"] for c in cases]), 4),
            "execution_accuracy_rate": round(avg([1.0 if c["metrics"]["sql"]["execution_accuracy"] else 0.0 for c in cases]), 4),
            "result_accuracy_rate": round(avg([1.0 if c["metrics"]["sql"]["result_accuracy"] else 0.0 for c in cases]), 4),
            "avg_latency_s": round(avg([c["metrics"]["latency"]["total_latency_s"] for c in cases]), 4),
        }

    # 组装最终结构
    log_data = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_cases": len(results),
        },
        "results": entries,
        "overall": overall,
    }

    # 写入文件
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"metrics_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(log_path, "w", encoding="utf-8") as f:
        yaml.dump(log_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return str(log_path)


async def main():
    parser = argparse.ArgumentParser(description="NL2SQL 评测工具")
    parser.add_argument("--cases", default=str(EVAL_DIR / "test_cases_v3.yaml"), help="测试用例文件路径")
    parser.add_argument("--report", default=str(EVAL_DIR / "reports"), help="报告输出目录")
    args = parser.parse_args()

    # 统一解析为绝对路径
    cases_path = Path(args.cases).resolve()
    report_dir = Path(args.report).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # 初始化运行时日志（按时间戳生成，不覆盖）
    log_path = setup_logging(report_dir)
    logger = logging.getLogger("eval")
    logger.info(f"运行日志: {log_path}")
    logger.info(f"测试用例: {cases_path}")
    logger.info(f"报告目录: {report_dir}")

    logger.info("初始化客户端...")
    embedding_client_manager.init()
    qdrant_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()
    await init_checkpointer()
    setup_graph()

    async with meta_mysql_client_manager.session_factory() as meta_session, \
               dw_mysql_client_manager.session_factory() as dw_session:
        context = DataAgentContext(
            embedding_client=embedding_client_manager.client,
            column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
            value_es_repository=ValueESRepository(es_client_manager.client),
            metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
            meta_mysql_repository=MetaMySQLRepository(meta_session),
            dw_mysql_repository=DWMySQLRepository(dw_session),
        )

        test_cases = load_test_cases(str(cases_path))
        logger.info(f"共 {len(test_cases)} 条用例")

        results = []
        for i, case in enumerate(test_cases):
            result = await run_single_case(case, context, i + 1)
            results.append(result)
            logger.info(f"[{result['case_id']}] 完成 - SQL执行: {'OK' if result['metrics']['sql']['execution_accuracy'] else 'FAIL'}, "
                        f"结果: {'OK' if result['metrics']['sql']['result_accuracy'] else 'FAIL'}")

        report_path = generate_report(results, str(report_dir))
        metrics_log_path = generate_metrics_log(results, str(report_dir))

        logger.info(f"\n{'='*60}")
        logger.info(f"评测完成！")
        logger.info(f"  运行日志: {log_path}")
        logger.info(f"  Markdown 报告: {report_path}")
        logger.info(f"  指标日志: {metrics_log_path}")

        await qdrant_client_manager.close()
        await es_client_manager.close()
        await meta_mysql_client_manager.close()
        await dw_mysql_client_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
