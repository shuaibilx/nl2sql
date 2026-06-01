from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.context import DataAgentContext
from app.agent.graph import get_graph, setup_graph
from app.agent.state import DataAgentState
from app.clients.embedding_client_manager import embedding_client_manager
from app.clients.es_client_manager import es_client_manager
from app.clients.mysql_client_manager import dw_mysql_client_manager, meta_mysql_client_manager
from app.clients.qdrant_client_manager import qdrant_client_manager
from app.conf.app_config import app_config
from app.core.cache_context import CacheScope, use_cache_scope
from app.core.cache_registry import caches
from app.core.context import request_id_ctx_var
from app.repositories.es.value_es_repository import ValueESRepository
from app.repositories.mysql.dw.dw_mysql_repository import DWMySQLRepository
from app.repositories.mysql.meta.meta_mysql_repository import MetaMySQLRepository
from app.repositories.qdrant.column_qdrant_repository import ColumnQdrantRepository
from app.repositories.qdrant.metric_qdrant_repository import MetricQdrantRepository
from checkpoints.manager import close_checkpointer, init_checkpointer
from eval.metrics.benchmark import (
    canonical_result,
    evaluate_case,
    extract_scalar_result,
    summarize,
    summarize_cache_stats,
)


PROJECT_ROOT = Path(__file__).parent.parent
EVAL_DIR = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent NL2SQL benchmark runner")
    parser.add_argument("--cases", default=str(EVAL_DIR / "semantic_cache_cases.yaml"))
    parser.add_argument("--reports-dir", default=str(EVAL_DIR / "reports"))
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tenant-id", default="benchmark_tenant")
    parser.add_argument("--user-id", default="benchmark_user")
    parser.add_argument("--project-id", default="benchmark_project")
    parser.add_argument("--resume-interrupt", action="store_true", help="Auto-confirm interrupted SQL after max validation retries.")
    parser.add_argument("--langsmith-dataset", default=None, help="Optional LangSmith dataset name for uploading case inputs/outputs.")
    parser.add_argument("--case-timeout", type=float, default=180.0, help="Per-case timeout in seconds.")
    parser.add_argument("--validate-only", action="store_true", help="Only validate benchmark case schema and exit.")
    parser.add_argument(
        "--allow-duplicate-expected-sql",
        action="store_true",
        help="Allow repeated expected_sql values for semantic-cache paraphrase suites.",
    )
    return parser.parse_args()


def setup_logging(reports_dir: Path, timestamp: str) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / f"benchmark_{timestamp}.log"
    logger = logging.getLogger("benchmark")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)
    return log_path


def load_cases(path: Path, limit: int | None = None, allow_duplicate_expected_sql: bool = False) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        cases = yaml.safe_load(file) or []
    if limit is not None:
        cases = cases[:limit]
    validate_cases(cases, allow_duplicate_expected_sql=allow_duplicate_expected_sql)
    return cases


def is_semantic_cache_suite(cases: list[dict]) -> bool:
    return any(
        case.get("semantic_group") or str(case.get("category", "")).startswith("semantic_cache")
        for case in cases
    )


def validate_cases(cases: list[dict], allow_duplicate_expected_sql: bool = False) -> None:
    allow_duplicate_expected_sql = allow_duplicate_expected_sql or is_semantic_cache_suite(cases)
    required = {
        "id",
        "category",
        "query",
        "expected_sql",
        "expected_result",
        "expected_tables",
        "expected_columns",
    }
    ids = set()
    queries = set()
    expected_sqls = set()
    for index, case in enumerate(cases, start=1):
        missing = required - set(case)
        if missing:
            raise ValueError(f"case #{index} missing required fields: {sorted(missing)}")
        if case["id"] in ids:
            raise ValueError(f"duplicate case id: {case['id']}")
        ids.add(case["id"])
        if case["query"] in queries:
            raise ValueError(f"duplicate case query: {case['query']}")
        queries.add(case["query"])
        if not allow_duplicate_expected_sql and case["expected_sql"] in expected_sqls:
            raise ValueError(f"duplicate expected_sql in case: {case['id']}")
        expected_sqls.add(case["expected_sql"])


async def init_runtime() -> None:
    logger = logging.getLogger("benchmark")
    logger.info("Initializing embedding client")
    embedding_client_manager.init()
    logger.info("Initializing Qdrant client")
    qdrant_client_manager.init()
    logger.info("Initializing Elasticsearch client")
    es_client_manager.init()
    logger.info("Initializing MySQL clients")
    meta_mysql_client_manager.init()
    dw_mysql_client_manager.init()
    logger.info("Initializing cache backend")
    await caches.init(app_config.cache, app_config.redis)
    logger.info("Initializing LangGraph checkpoint")
    await init_checkpointer()
    logger.info("Compiling LangGraph")
    setup_graph()


async def close_runtime() -> None:
    await embedding_client_manager.close()
    await qdrant_client_manager.close()
    await es_client_manager.close()
    await meta_mysql_client_manager.close()
    await dw_mysql_client_manager.close()
    await close_checkpointer()
    await caches.close()


def make_actual() -> dict:
    return {
        "cleaned_query": "",
        "sql": "",
        "result_data": None,
        "progress_events": [],
        "node_updates": [],
        "hit_interrupt": False,
    }


def extract_actual_from_update(chunk: dict, actual: dict) -> None:
    for node_name, node_output in chunk.items():
        if node_name.startswith("__") or not isinstance(node_output, dict):
            continue
        actual["node_updates"].append({"node": node_name, "keys": list(node_output.keys())})
        if "cleaned_query" in node_output:
            actual["cleaned_query"] = node_output["cleaned_query"]
        if "sql" in node_output:
            actual["sql"] = node_output["sql"]
        if "result_data" in node_output:
            actual["result_data"] = node_output["result_data"]
        if "error" in node_output:
            actual["validation_error"] = node_output["error"]


async def collect_stream(input_: Any, context: DataAgentContext, config: dict, actual: dict) -> None:
    graph = get_graph()
    async for mode, chunk in graph.astream(
        input=input_,
        context=context,
        config=config,
        stream_mode=["custom", "updates"],
    ):
        if mode == "custom" and isinstance(chunk, dict):
            event = {
                "type": chunk.get("type"),
                "step": chunk.get("step"),
                "status": chunk.get("status"),
                "detail": chunk.get("detail"),
            }
            actual["progress_events"].append({key: value for key, value in event.items() if value is not None})
        elif mode == "updates" and isinstance(chunk, dict):
            if "__interrupt__" in chunk:
                actual["hit_interrupt"] = True
            extract_actual_from_update(chunk, actual)


async def run_case(case: dict, semaphore: asyncio.Semaphore, args: argparse.Namespace, index: int) -> dict:
    async with semaphore:
        logger = logging.getLogger("benchmark")
        case_id = case.get("id", f"B{index:03d}")
        query = case["query"]
        scope = CacheScope.from_optional(args.tenant_id, args.user_id, args.project_id)
        request_id_ctx_var.set(f"benchmark-{case_id}-{uuid.uuid4().hex[:8]}")
        actual = make_actual()
        error = None
        start = time.perf_counter()

        try:
            async with asyncio.timeout(args.case_timeout):
                async with (
                    meta_mysql_client_manager.session_factory() as meta_session,
                    dw_mysql_client_manager.session_factory() as dw_session,
                ):
                    context = DataAgentContext(
                        embedding_client=embedding_client_manager.client,
                        column_qdrant_repository=ColumnQdrantRepository(qdrant_client_manager.client),
                        value_es_repository=ValueESRepository(es_client_manager.client),
                        metric_qdrant_repository=MetricQdrantRepository(qdrant_client_manager.client),
                        meta_mysql_repository=MetaMySQLRepository(meta_session),
                        dw_mysql_repository=DWMySQLRepository(dw_session),
                        cache_scope=scope,
                    )
                    state = DataAgentState(query=query, cleaned_query="", retry_count=0)
                    config = {"configurable": {"thread_id": f"benchmark-{case_id}-{uuid.uuid4().hex[:8]}"}}
                    with use_cache_scope(scope):
                        await collect_stream(state, context, config, actual)
                        if actual["hit_interrupt"] and args.resume_interrupt:
                            await collect_stream(Command(resume=True), context, config, actual)
        except TimeoutError:
            error = f"TimeoutError: case exceeded {args.case_timeout:.1f}s"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.debug("case %s failed", case_id, exc_info=True)

        latency_s = time.perf_counter() - start
        metrics = evaluate_case(case, actual, error, latency_s)
        result = {
            "id": case_id,
            "category": case.get("category", "unknown"),
            "query": query,
            "expected": {
                "sql": case.get("expected_sql"),
                "result": case.get("expected_result"),
                "tables": case.get("expected_tables", []),
                "columns": case.get("expected_columns", []),
            },
            "actual": {
                "cleaned_query": actual.get("cleaned_query"),
                "sql": actual.get("sql"),
                "result": actual.get("result_data"),
                "result_normalized": canonical_result(actual.get("result_data")),
                "result_scalar": extract_scalar_result(actual.get("result_data")),
                "progress_events": actual.get("progress_events", []),
                "node_updates": actual.get("node_updates", []),
            },
            "metrics": metrics,
            "error": error,
        }
        logger.info(
            "[%s] result=%s sql=%s latency=%.2fs %s",
            case_id,
            "OK" if metrics["result_accuracy"] else "FAIL",
            "OK" if metrics["sql_execution_ok"] else "FAIL",
            latency_s,
            query,
        )
        return result


async def run_benchmark(cases: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    semaphore = asyncio.Semaphore(args.concurrency)
    start = time.perf_counter()
    tasks = [run_case(case, semaphore, args, index + 1) for index, case in enumerate(cases)]
    results = await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - start
    return results, summarize(results, wall_time, args.concurrency)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def format_ratio(value: float) -> str:
    return f"{value:.2%}"


def write_markdown_report(path: Path, summary: dict, rows: list[dict], log_path: Path, jsonl_path: Path) -> None:
    failed = [row for row in rows if not row["metrics"]["result_accuracy"] or row.get("error")]
    cache_metrics = summary.get("cache_metrics", {})
    cache_total = cache_metrics.get("total", {})
    cache_by_name = cache_metrics.get("by_cache", {})
    lines = [
        "# NL2SQL Concurrent Benchmark Report",
        "",
        f"- Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Log file: `{log_path}`",
        f"- Sample JSONL: `{jsonl_path}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Case count | {summary['case_count']} |",
        f"| Concurrency | {summary['concurrency']} |",
        f"| Result accuracy | {summary['result_accuracy']:.2%} |",
        f"| SQL execution rate | {summary['sql_execution_rate']:.2%} |",
        f"| P50 latency | {summary['p50_latency_s']:.3f}s |",
        f"| P95 latency | {summary['p95_latency_s']:.3f}s |",
        f"| Throughput | {summary['throughput_qps']:.3f} qps |",
        f"| Error rate | {summary['error_rate']:.2%} |",
        f"| Total wall time | {summary['total_wall_time_s']:.3f}s |",
        f"| Cache requests | {cache_total.get('requests', 0)} |",
        f"| Cache hit ratio | {format_ratio(cache_total.get('hit_ratio', 0.0))} |",
        f"| Cache expired ratio | {format_ratio(cache_total.get('expired_ratio', 0.0))} |",
        f"| Cache eviction ratio | {format_ratio(cache_total.get('eviction_ratio', 0.0))} |",
        "",
        "## Cache Metrics",
        "",
        "| Cache | Backend | Requests | Hits | Misses | Expired | Stale hits | Stale misses | Sets | Evictions | Hit ratio | Expired ratio | Eviction ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if cache_by_name:
        for cache_name, stats in cache_by_name.items():
            lines.append(
                f"| {cache_name} | {stats.get('backend', 'unknown')} | "
                f"{stats.get('requests', 0)} | {stats.get('hits', 0)} | {stats.get('misses', 0)} | "
                f"{stats.get('expired', 0)} | {stats.get('stale_hits', 0)} | {stats.get('stale_misses', 0)} | "
                f"{stats.get('sets', 0)} | {stats.get('evictions', 0)} | "
                f"{format_ratio(stats.get('hit_ratio', 0.0))} | "
                f"{format_ratio(stats.get('expired_ratio', 0.0))} | "
                f"{format_ratio(stats.get('eviction_ratio', 0.0))} |"
            )
    else:
        lines.append("| - | - | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.00% | 0.00% | 0.00% |")

    lines.extend([
        "",
        "## Case Results",
        "",
        "| ID | Category | Result | SQL | Latency | Query |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    for row in rows:
        result_mark = "OK" if row["metrics"]["result_accuracy"] else "FAIL"
        sql_mark = "OK" if row["metrics"]["sql_execution_ok"] else "FAIL"
        lines.append(
            f"| {row['id']} | {row['category']} | {result_mark} | {sql_mark} | "
            f"{row['metrics']['latency_s']:.3f}s | {row['query']} |"
        )

    lines.extend(["", "## Failed / Suspicious Samples", ""])
    if not failed:
        lines.append("No failed samples.")
    for row in failed:
        lines.extend([
            f"### {row['id']} {row['query']}",
            "",
            f"- Error: `{row.get('error')}`",
            f"- SQL execution ok: `{row['metrics']['sql_execution_ok']}`",
            f"- Result accuracy: `{row['metrics']['result_accuracy']}`",
            f"- SQL shape similarity: `{row['metrics']['sql_shape_similarity']}`",
            f"- Missing tables: `{row['metrics']['missing_tables']}`",
            f"- Missing columns: `{row['metrics']['missing_columns']}`",
            "",
            "**Expected SQL**",
            "```sql",
            str(row["expected"]["sql"] or ""),
            "```",
            "",
            "**Actual SQL**",
            "```sql",
            str(row["actual"]["sql"] or ""),
            "```",
            "",
            "**Expected Result**",
            "```json",
            json.dumps(row["expected"]["result"], ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "**Actual Result**",
            "```json",
            json.dumps(row["actual"].get("result_normalized", row["actual"]["result"]), ensure_ascii=False, indent=2, default=str),
            "```",
            "",
        ])
        raw_result = row["actual"]["result"]
        normalized_result = row["actual"].get("result_normalized", raw_result)
        if raw_result != normalized_result:
            lines.extend([
                "**Actual Result (raw)**",
                "```json",
                json.dumps(raw_result, ensure_ascii=False, indent=2, default=str),
                "```",
                "",
            ])
    path.write_text("\n".join(lines), encoding="utf-8")


def maybe_upload_langsmith(dataset_name: str | None, rows: list[dict], summary: dict) -> None:
    if not dataset_name:
        return
    logger = logging.getLogger("benchmark")
    try:
        from langsmith import Client
    except Exception as exc:
        logger.warning("LangSmith upload skipped: %s", exc)
        return

    client = Client()
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="NL2SQL benchmark cases with expected SQL/result and actual run outputs.",
        )
    for row in rows:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"query": row["query"], "category": row["category"]},
            outputs={
                "expected_sql": row["expected"]["sql"],
                "expected_result": row["expected"]["result"],
                "actual_sql": row["actual"]["sql"],
                "actual_result": row["actual"]["result"],
                "actual_result_normalized": row["actual"].get("result_normalized"),
                "metrics": row["metrics"],
            },
        )
    logger.info("Uploaded %s examples to LangSmith dataset %s", len(rows), dataset_name)


async def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(args.reports_dir)
    log_path = setup_logging(reports_dir, timestamp)
    logger = logging.getLogger("benchmark")

    cases = load_cases(
        Path(args.cases),
        args.limit,
        allow_duplicate_expected_sql=args.allow_duplicate_expected_sql,
    )
    logger.info("Loaded %s cases from %s", len(cases), args.cases)
    logger.info("Concurrency=%s, resume_interrupt=%s", args.concurrency, args.resume_interrupt)
    if args.validate_only:
        logger.info("Case schema validation passed.")
        return

    await init_runtime()
    try:
        cache_before = caches.all_stats()
        rows, summary = await run_benchmark(cases, args)
        cache_after = caches.all_stats()
        summary["cache_metrics"] = summarize_cache_stats(cache_before, cache_after)
    finally:
        await close_runtime()

    jsonl_path = reports_dir / f"benchmark_samples_{timestamp}.jsonl"
    report_path = reports_dir / f"benchmark_report_{timestamp}.md"
    write_jsonl(jsonl_path, rows)
    write_markdown_report(report_path, summary, rows, log_path, jsonl_path)
    maybe_upload_langsmith(args.langsmith_dataset, rows, summary)

    logger.info("Report: %s", report_path)
    logger.info("Samples: %s", jsonl_path)
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
