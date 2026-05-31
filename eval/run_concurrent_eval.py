"""
[评测] 并发评测脚本 — 并发运行测试用例，测量并发性能和瓶颈

使用方式：
    python eval/run_concurrent_eval.py --concurrency 5
    python eval/run_concurrent_eval.py --concurrency 1,2,5,10 --run-sequential-first
    python eval/run_concurrent_eval.py --concurrency 5 --cases eval/test_cases.yaml

输出文件（每次运行按时间戳生成，不覆盖）：
    eval/reports/concurrent_eval_YYYYMMDD_HHMMSS.md  — 并发评测报告
    eval/reports/concurrent_eval_YYYYMMDD_HHMMSS.log — 运行日志
"""
import asyncio
import argparse
import logging
import time
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from eval.metrics.sql_accuracy import evaluate_sql, _extract_value
from eval.metrics.latency import LatencyTracker
from eval.metrics.concurrency import ConcurrencyMetrics, percentile

# 从 run_eval.py 复用的函数
from eval.run_eval import (
    extract_actual_from_chunk,
    load_test_cases,
    setup_logging,
    _fmt_result,
    _extract_value_for_log,
)


async def collect_stream_concurrent(graph, input_, context, config, tracker, actual):
    """收集一次 astream 的所有事件（并发安全版，不写共享日志）"""
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
            extract_actual_from_chunk(chunk, actual)
    return hit_interrupt


def _make_actual_dict() -> dict:
    """创建 actual 字典的初始值"""
    return {
        "cleaned_query": "",
        "keywords": [],
        "column_keywords": [],
        "value_keywords": [],
        "metric_keywords": [],
        "retrieved_columns": [],
        "retrieved_values": [],
        "retrieved_metrics": [],
        "filtered_columns": [],
        "filtered_metrics": [],
        "sql": "",
        "execution_error": None,
        "result_data": None,
    }


async def run_single_case_concurrent(
    case: dict,
    semaphore: asyncio.Semaphore,
    case_index: int,
) -> dict:
    """并发运行单条测试用例

    每个任务创建独立的 MySQL session（从 pool 获取），不共享 context。
    semaphore 控制最大并发数，防止 pool 耗尽。
    """
    async with semaphore:
        wall_start = time.time()
        case_id = case.get("id", f"TC{case_index:03d}")
        query = case["query"]

        # 设置唯一的 request_id，区分并发任务的日志
        request_id_ctx_var.set(f"concurrent-{case_id}")

        actual = _make_actual_dict()
        tracker = LatencyTracker()
        error = None

        try:
            # 每个并发任务创建独立的 session
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

                state = DataAgentState(query=query, cleaned_query="", retry_count=0)
                config = {"configurable": {"thread_id": f"concurrent-{case_id}"}}
                graph = get_graph()

                tracker.start()
                hit_interrupt = await collect_stream_concurrent(
                    graph, state, context, config, tracker, actual
                )

                if hit_interrupt:
                    await collect_stream_concurrent(
                        graph, Command(resume=True), context, config, tracker, actual
                    )
                tracker.stop()

        except Exception as e:
            import traceback
            error = f"{type(e).__name__}: {str(e)[:200]}"
            tracker.stop()

        wall_end = time.time()

        # 计算指标
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
            execution_error=actual.get("execution_error") or error,
            sql_pattern=expected.get("sql_pattern", ""),
            expected_tables=expected.get("tables", []),
            expected_columns=expected.get("columns", []),
            actual_result=actual.get("result_data"),
            expected_result=expected.get("expected_result"),
        )
        latency_metrics = tracker.summary()

        return {
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
            "wall_start": wall_start,
            "wall_end": wall_end,
            "wall_latency": wall_end - wall_start,
            "error": error,
        }


async def run_sequential(test_cases: list[dict]) -> list[dict]:
    """串行运行所有测试用例（作为基线）"""
    logger = logging.getLogger("eval")
    logger.info("=" * 50)
    logger.info("串行基线运行")
    logger.info("=" * 50)

    results = []
    for i, case in enumerate(test_cases):
        semaphore = asyncio.Semaphore(1)  # 串行
        result = await run_single_case_concurrent(case, semaphore, i + 1)
        results.append(result)
        exec_ok = result["metrics"]["sql"]["execution_accuracy"]
        result_ok = result["metrics"]["sql"]["result_accuracy"]
        logger.info(f"  [{result['case_id']}] {result['query'][:30]} | "
                    f"SQL={'OK' if exec_ok else 'FAIL'} 结果={'OK' if result_ok else 'FAIL'} "
                    f"延迟={result['wall_latency']:.1f}s")

    return results


async def run_concurrent(test_cases: list[dict], concurrency: int) -> list[dict]:
    """并发运行所有测试用例"""
    logger = logging.getLogger("eval")
    logger.info("=" * 50)
    logger.info(f"并发运行（并发度={concurrency}）")
    logger.info("=" * 50)

    semaphore = asyncio.Semaphore(concurrency)

    tasks = [
        run_single_case_concurrent(case, semaphore, i + 1)
        for i, case in enumerate(test_cases)
    ]

    overall_start = time.time()
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    overall_end = time.time()

    # 分离正常结果和异常
    results = []
    for i, r in enumerate(raw_results):
        if isinstance(r, Exception):
            case = test_cases[i]
            case_id = case.get("id", f"TC{i+1:03d}")
            results.append({
                "case_id": case_id,
                "category": case.get("category", "unknown"),
                "query": case["query"],
                "actual": _make_actual_dict(),
                "expected": case.get("expected", {}),
                "metrics": {
                    "retrieval": {"overall_recall": 0},
                    "sql": {"execution_accuracy": False, "result_accuracy": False,
                            "pattern_match": False, "table_coverage": 0, "column_coverage": 0},
                    "latency": {"total_latency_s": 0},
                },
                "wall_start": overall_start,
                "wall_end": overall_end,
                "wall_latency": 0,
                "error": f"{type(r).__name__}: {str(r)[:200]}",
            })
        else:
            results.append(r)
            exec_ok = r["metrics"]["sql"]["execution_accuracy"]
            result_ok = r["metrics"]["sql"]["result_accuracy"]
            logger.info(f"  [{r['case_id']}] {r['query'][:30]} | "
                        f"SQL={'OK' if exec_ok else 'FAIL'} 结果={'OK' if result_ok else 'FAIL'} "
                        f"延迟={r['wall_latency']:.1f}s")

    # 记录总墙钟时间
    for r in results:
        r["overall_wall_clock"] = overall_end - overall_start

    logger.info(f"  总墙钟时间: {overall_end - overall_start:.1f}s")

    return results


def compute_concurrency_report(
    results: list[dict],
    concurrency: int,
    label: str = "concurrent",
) -> dict:
    """计算并发运行的汇总指标"""
    metrics = ConcurrencyMetrics()
    metrics.start()

    errors = []
    for r in results:
        if r.get("error"):
            metrics.record_error(r["error"])
            errors.append(r)
        else:
            metrics.record_success(r["wall_latency"])

    metrics.stop()

    summary = metrics.summary_dict()
    summary["concurrency"] = concurrency
    summary["label"] = label
    summary["overall_wall_clock"] = results[0].get("overall_wall_clock", metrics.total_wall_clock) if results else 0

    # 计算加速比
    sequential_estimate = sum(r["wall_latency"] for r in results if not r.get("error"))
    wall_clock = summary["overall_wall_clock"]
    summary["sequential_estimate_s"] = round(sequential_estimate, 3)
    summary["speedup_ratio"] = round(sequential_estimate / wall_clock, 2) if wall_clock > 0 else 0

    return summary


def generate_comparison_report(
    reports: list[dict],
    results_by_concurrency: dict[int, list[dict]],
    output_dir: Path,
) -> str:
    """生成并发评测对比报告"""
    lines = []
    lines.append("# NL2SQL 并发评测报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试用例数**: {len(list(results_by_concurrency.values())[0]) if results_by_concurrency else 0}")
    lines.append("")

    # 总览表
    lines.append("## 总览")
    lines.append("")
    lines.append("| 并发度 | 墙钟时间(s) | 吞吐量(req/s) | p50(s) | p95(s) | p99(s) | 错误率 | 加速比 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r['concurrency']} "
            f"| {r.get('overall_wall_clock', r['wall_clock_s']):.1f} "
            f"| {r['throughput_rps']:.3f} "
            f"| {r['latency_p50_s']:.1f} "
            f"| {r['latency_p95_s']:.1f} "
            f"| {r['latency_p99_s']:.1f} "
            f"| {r['error_rate']:.1%} "
            f"| {r['speedup_ratio']:.2f}x |"
        )
    lines.append("")

    # 瓶颈分析
    lines.append("## 瓶颈分析")
    lines.append("")
    if len(reports) >= 2:
        base = reports[0]
        for r in reports[1:]:
            speedup = r["speedup_ratio"]
            if speedup < 1.2:
                lines.append(f"- **并发度 {r['concurrency']}**: 加速比 {speedup:.2f}x，几乎无并行收益。"
                             f"瓶颈可能是 SQLite checkpointer（单写入锁）或 LLM 串行处理。")
            elif speedup < 2.5:
                lines.append(f"- **并发度 {r['concurrency']}**: 加速比 {speedup:.2f}x，有部分并行收益。"
                             f"瓶颈开始显现，可能是 MySQL 连接池或 Embedding 服务。")
            else:
                lines.append(f"- **并发度 {r['concurrency']}**: 加速比 {speedup:.2f}x，并行效果良好。")

            if r["error_rate"] > 0.1:
                lines.append(f"  错误率 {r['error_rate']:.1%}，主要错误类型: {r.get('error_categories', {})}")
    lines.append("")

    # 逐请求详情（每个并发度一个表）
    for concurrency, results in results_by_concurrency.items():
        lines.append(f"## 并发度 {concurrency} 详情")
        lines.append("")
        lines.append("| 编号 | 查询 | SQL执行 | 结果准确 | 延迟(s) | 错误 |")
        lines.append("|---|---|---|---|---|---|")
        for r in results:
            exec_ok = r["metrics"]["sql"]["execution_accuracy"]
            result_ok = r["metrics"]["sql"]["result_accuracy"]
            error = r.get("error", "")
            lines.append(
                f"| {r['case_id']} "
                f"| {r['query'][:25]} "
                f"| {'OK' if exec_ok else 'FAIL'} "
                f"| {'OK' if result_ok else 'FAIL'} "
                f"| {r['wall_latency']:.1f} "
                f"| {error[:30] if error else '-'} |"
            )
        lines.append("")

    # 写入文件
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"concurrent_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(report_path)


async def main():
    parser = argparse.ArgumentParser(description="NL2SQL 并发评测工具")
    parser.add_argument("--cases", default=str(EVAL_DIR / "test_cases_v3.yaml"), help="测试用例文件路径")
    parser.add_argument("--report", default=str(EVAL_DIR / "reports"), help="报告输出目录")
    parser.add_argument("--concurrency", type=str, default="10",
                        help="并发度，多个用逗号分隔（如 1,2,5,10）")
    parser.add_argument("--run-sequential-first", action="store_true",
                        help="先运行串行基线，再运行并发")
    args = parser.parse_args()

    cases_path = Path(args.cases).resolve()
    report_dir = Path(args.report).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # 初始化日志
    log_path = setup_logging(report_dir)
    logger = logging.getLogger("eval")
    logger.info(f"运行日志: {log_path}")
    logger.info(f"测试用例: {cases_path}")

    # 初始化客户端
    logger.info("初始化客户端...")
    embedding_client_manager.init()
    qdrant_client_manager.init()
    es_client_manager.init()
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()
    await init_checkpointer()
    setup_graph()

    # 加载测试用例
    test_cases = load_test_cases(str(cases_path))
    logger.info(f"共 {len(test_cases)} 条用例")

    # 解析并发度列表
    concurrency_levels = [int(x.strip()) for x in args.concurrency.split(",")]

    reports = []
    results_by_concurrency = {}

    # 可选：先运行串行基线
    if args.run_sequential_first and 1 not in concurrency_levels:
        seq_results = await run_sequential(test_cases)
        seq_report = compute_concurrency_report(seq_results, concurrency=1, label="sequential")
        reports.append(seq_report)
        results_by_concurrency[1] = seq_results

    # 运行各并发度
    for concurrency in concurrency_levels:
        if concurrency == 1 and args.run_sequential_first and 1 in results_by_concurrency:
            continue  # 已经运行过串行基线

        results = await run_concurrent(test_cases, concurrency)
        report = compute_concurrency_report(results, concurrency)
        reports.append(report)
        results_by_concurrency[concurrency] = results

    # 生成对比报告
    report_path = generate_comparison_report(reports, results_by_concurrency, report_dir)

    logger.info(f"\n{'='*50}")
    logger.info(f"并发评测完成！")
    logger.info(f"  报告: {report_path}")
    logger.info(f"  日志: {log_path}")

    # 打印摘要
    logger.info(f"\n{'='*50}")
    logger.info(f"| 并发度 | 墙钟(s) | 吞吐量 | p50(s) | 错误率 | 加速比 |")
    logger.info(f"|---|---|---|---|---|---|")
    for r in reports:
        logger.info(
            f"| {r['concurrency']} "
            f"| {r.get('overall_wall_clock', r['wall_clock_s']):.1f} "
            f"| {r['throughput_rps']:.3f} "
            f"| {r['latency_p50_s']:.1f} "
            f"| {r['error_rate']:.1%} "
            f"| {r['speedup_ratio']:.2f}x |"
        )

    await qdrant_client_manager.close()
    await es_client_manager.close()
    await meta_mysql_client_manager.close()
    await dw_mysql_client_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
