from eval.metrics.benchmark import (
    canonical_result,
    compare_results,
    summarize,
    summarize_cache_stats,
    table_column_coverage,
)
from eval.refresh_expected_results import assert_readonly_sql, normalize_rows
from eval.run_benchmark import validate_cases, write_markdown_report


def test_compare_results_handles_scalar_and_row_sets():
    assert compare_results([{"result": 8.0}], 8)
    assert compare_results([{"total_sales_quantity": "622"}], 622)
    assert compare_results([{"SUM(f.order_quantity)": "322"}], 322)
    assert compare_results('[{"SUM(f.order_quantity)": "322"}]', 322)
    assert compare_results(
        [{"province": "浙江省", "result": 1.0}, {"province": "上海市", "result": 2.0}],
        [["上海市", 2.0], ["浙江省", 1.0]],
    )
    assert not compare_results([{"result": 8.0}], 9)


def test_canonical_result_ignores_single_column_aliases():
    assert canonical_result([{"total_sales_quantity": "622"}]) == "622"
    assert canonical_result([{"SUM(f.order_quantity)": "322"}]) == "322"
    assert canonical_result([{"quarter": "Q1", "total_sales": 8999.0}]) == [["Q1", 8999.0]]


def test_table_column_coverage_reports_missing_items():
    coverage = table_column_coverage(
        "select sum(f.order_amount) from fact_order f",
        ["fact_order", "dim_region"],
        ["order_amount", "province"],
    )
    assert coverage["table_coverage"] == 0.5
    assert coverage["column_coverage"] == 0.5
    assert coverage["missing_tables"] == ["dim_region"]
    assert coverage["missing_columns"] == ["province"]


def test_summarize_concurrent_metrics():
    rows = [
        {"metrics": {"latency_s": 1.0, "result_accuracy": True, "sql_execution_ok": True}},
        {"metrics": {"latency_s": 3.0, "result_accuracy": False, "sql_execution_ok": True}},
    ]
    summary = summarize(rows, total_wall_time_s=2.0, concurrency=2)
    assert summary["case_count"] == 2
    assert summary["result_accuracy"] == 0.5
    assert summary["sql_execution_rate"] == 1.0
    assert summary["p50_latency_s"] == 2.0
    assert summary["throughput_qps"] == 1.0


def test_summarize_cache_stats_reports_run_delta():
    before = {
        "generate_sql": {
            "backend": "redis",
            "hits": 2,
            "misses": 3,
            "expired": 1,
            "stale_hits": 0,
            "stale_misses": 1,
            "sets": 3,
            "evictions": 0,
        }
    }
    after = {
        "generate_sql": {
            "backend": "redis",
            "hits": 7,
            "misses": 5,
            "expired": 2,
            "stale_hits": 1,
            "stale_misses": 2,
            "sets": 8,
            "evictions": 1,
        },
        "embedding": {
            "backend": "redis",
            "hits": 1,
            "misses": 1,
            "expired": 0,
            "stale_hits": 0,
            "stale_misses": 0,
            "sets": 1,
            "evictions": 0,
        },
    }

    summary = summarize_cache_stats(before, after)

    assert summary["by_cache"]["generate_sql"]["requests"] == 8
    assert summary["by_cache"]["generate_sql"]["hits"] == 5
    assert summary["by_cache"]["generate_sql"]["hit_ratio"] == 0.625
    assert summary["total"]["requests"] == 10
    assert summary["total"]["hits"] == 6
    assert summary["total"]["sets"] == 6
    assert summary["total"]["evictions"] == 1


def test_benchmark_report_contains_cache_metrics(tmp_path):
    report_path = tmp_path / "report.md"
    log_path = tmp_path / "benchmark.log"
    jsonl_path = tmp_path / "samples.jsonl"
    summary = {
        "case_count": 1,
        "concurrency": 1,
        "result_accuracy": 1.0,
        "sql_execution_rate": 1.0,
        "p50_latency_s": 1.0,
        "p95_latency_s": 1.0,
        "throughput_qps": 1.0,
        "error_rate": 0.0,
        "total_wall_time_s": 1.0,
        "cache_metrics": {
            "total": {
                "requests": 2,
                "hit_ratio": 0.5,
                "expired_ratio": 0.0,
                "eviction_ratio": 0.0,
            },
            "by_cache": {
                "generate_sql": {
                    "backend": "redis",
                    "requests": 2,
                    "hits": 1,
                    "misses": 1,
                    "expired": 0,
                    "stale_hits": 0,
                    "stale_misses": 0,
                    "sets": 1,
                    "evictions": 0,
                    "hit_ratio": 0.5,
                    "expired_ratio": 0.0,
                    "eviction_ratio": 0.0,
                }
            },
        },
    }
    rows = [
        {
            "id": "B001",
            "category": "simple",
            "query": "统计销售额",
            "expected": {"sql": "SELECT 1", "result": 1},
            "actual": {"sql": "SELECT 1", "result": 1, "result_normalized": 1},
            "metrics": {
                "result_accuracy": True,
                "sql_execution_ok": True,
                "latency_s": 1.0,
                "sql_shape_similarity": 1.0,
                "missing_tables": [],
                "missing_columns": [],
            },
            "error": None,
        }
    ]

    write_markdown_report(report_path, summary, rows, log_path, jsonl_path)

    content = report_path.read_text(encoding="utf-8")
    assert "## Cache Metrics" in content
    assert "| generate_sql | redis | 2 | 1 | 1 |" in content
    assert "| Cache hit ratio | 50.00% |" in content


def test_refresh_expected_results_normalizes_rows():
    assert normalize_rows([{"result": 8}]) == 8
    assert normalize_rows([{"province": "浙江省", "result": 1.5}]) == [["浙江省", 1.5]]


def test_refresh_expected_results_rejects_write_sql():
    try:
        assert_readonly_sql("delete from fact_order")
    except ValueError:
        pass
    else:
        raise AssertionError("write SQL should be rejected")


def test_validate_cases_rejects_duplicate_queries():
    base = {
        "category": "simple",
        "query": "统计销售额",
        "expected_result": 1,
        "expected_tables": ["fact_order"],
        "expected_columns": ["order_amount"],
    }
    cases = [
        {**base, "id": "B001", "expected_sql": "SELECT 1"},
        {**base, "id": "B002", "expected_sql": "SELECT 2"},
    ]
    try:
        validate_cases(cases)
    except ValueError as exc:
        assert "duplicate case query" in str(exc)
    else:
        raise AssertionError("duplicate query should be rejected")


def test_validate_cases_can_allow_duplicate_expected_sql():
    base = {
        "category": "simple",
        "expected_sql": "SELECT SUM(order_amount) AS result FROM fact_order",
        "expected_result": 1,
        "expected_tables": ["fact_order"],
        "expected_columns": ["order_amount"],
    }
    cases = [
        {**base, "id": "S001", "query": "统计销售总额"},
        {**base, "id": "S002", "query": "销售额一共是多少"},
    ]

    try:
        validate_cases(cases)
    except ValueError as exc:
        assert "duplicate expected_sql" in str(exc)
    else:
        raise AssertionError("duplicate expected_sql should be rejected by default")

    validate_cases(cases, allow_duplicate_expected_sql=True)


def test_validate_cases_auto_allows_semantic_cache_suite_duplicates():
    base = {
        "category": "semantic_cache_cluster",
        "semantic_group": "total_sales_amount",
        "expected_sql": "SELECT SUM(order_amount) AS result FROM fact_order",
        "expected_result": 1,
        "expected_tables": ["fact_order"],
        "expected_columns": ["order_amount"],
    }
    cases = [
        {**base, "id": "S001", "query": "统计销售总额"},
        {**base, "id": "S002", "query": "销售额一共是多少"},
    ]

    validate_cases(cases)
