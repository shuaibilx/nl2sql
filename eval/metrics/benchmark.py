from __future__ import annotations

import math
import re
from decimal import Decimal
import json
from collections.abc import Iterable
from typing import Any


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def canonical_result(result: Any) -> Any:
    if result is None:
        return None

    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith(("[", "{")):
            try:
                return canonical_result(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        return result

    if isinstance(result, (int, float, Decimal)):
        return result

    if isinstance(result, dict):
        values = list(result.values())
        if len(values) == 1:
            return canonical_result(values[0])
        return [canonical_result(value) for value in values]

    if isinstance(result, list):
        if not result:
            return None
        if len(result) == 1:
            row = result[0]
            if isinstance(row, dict):
                values = list(row.values())
                if len(values) == 1:
                    return canonical_result(values[0])
                return [[canonical_result(value) for value in values]]
            if isinstance(row, (list, tuple)):
                if len(row) == 1:
                    return canonical_result(row[0])
                return [[canonical_result(value) for value in row]]
            return canonical_result(row)
        return [normalize_row(row) for row in result]

    if isinstance(result, tuple):
        if len(result) == 1:
            return canonical_result(result[0])
        return [canonical_result(value) for value in result]

    return result


def extract_scalar_result(result: Any) -> Any:
    return canonical_result(result)


def normalize_row(row: Any) -> Any:
    if isinstance(row, dict):
        return [canonical_result(value) for value in row.values()]
    if isinstance(row, tuple):
        return [canonical_result(value) for value in row]
    if isinstance(row, list):
        return [canonical_result(value) for value in row]
    return row


def normalize_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return float(value)
        return round(float(value), 6)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return round(float(stripped), 6)
        except ValueError:
            return stripped
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_value(val) for key, val in value.items()}
    return value


def compare_results(actual_result: Any, expected_result: Any, tolerance: float = 0.01) -> bool:
    if expected_result is None:
        return True
    actual = normalize_value(canonical_result(actual_result))
    expected = normalize_value(canonical_result(expected_result))
    return values_equal(actual, expected, tolerance)


def values_equal(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return abs(actual - expected) <= tolerance
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return False
        if all(not isinstance(item, list) for item in actual + expected):
            return all(values_equal(a, b, tolerance) for a, b in zip(actual, expected))
        actual_rows = sorted(stable_repr(item) for item in actual)
        expected_rows = sorted(stable_repr(item) for item in expected)
        return actual_rows == expected_rows
    return stable_repr(actual) == stable_repr(expected)


def stable_repr(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, list):
        return "[" + ",".join(stable_repr(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{key}:{stable_repr(value[key])}" for key in sorted(value)) + "}"
    return str(value)


def table_column_coverage(sql: str, expected_tables: Iterable[str], expected_columns: Iterable[str]) -> dict:
    sql_lower = sql.lower()
    tables = list(expected_tables or [])
    columns = list(expected_columns or [])
    table_hits = [item for item in tables if item.lower() in sql_lower]
    column_hits = [item for item in columns if item.split(".")[-1].lower() in sql_lower]
    return {
        "table_coverage": len(table_hits) / len(tables) if tables else 1.0,
        "column_coverage": len(column_hits) / len(columns) if columns else 1.0,
        "missing_tables": [item for item in tables if item not in table_hits],
        "missing_columns": [item for item in columns if item not in column_hits],
    }


def sql_shape_similarity(actual_sql: str, expected_sql: str) -> float:
    actual_tokens = set(sql_tokens(actual_sql))
    expected_tokens = set(sql_tokens(expected_sql))
    if not expected_tokens:
        return 1.0
    return len(actual_tokens & expected_tokens) / len(expected_tokens)


def sql_tokens(sql: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[\u4e00-\u9fa5]+|\d+(?:\.\d+)?", sql.lower())


def evaluate_case(case: dict, actual: dict, execution_error: str | None, latency_s: float) -> dict:
    expected_result = case.get("expected_result")
    result_ok = compare_results(actual.get("result_data"), expected_result)
    execution_ok = bool(actual.get("sql")) and execution_error is None
    coverage = table_column_coverage(
        actual.get("sql", ""),
        case.get("expected_tables", []),
        case.get("expected_columns", []),
    )
    return {
        "result_accuracy": result_ok,
        "sql_execution_ok": execution_ok,
        "latency_s": round(latency_s, 3),
        "sql_shape_similarity": round(sql_shape_similarity(actual.get("sql", ""), case.get("expected_sql", "")), 4),
        **coverage,
    }


def summarize(results: list[dict], total_wall_time_s: float, concurrency: int) -> dict:
    latencies = [item["metrics"]["latency_s"] for item in results]
    completed = len(results)
    errors = [item for item in results if item.get("error")]
    result_ok = [item for item in results if item["metrics"]["result_accuracy"]]
    sql_ok = [item for item in results if item["metrics"]["sql_execution_ok"]]
    return {
        "case_count": completed,
        "concurrency": concurrency,
        "result_accuracy": len(result_ok) / completed if completed else 0.0,
        "sql_execution_rate": len(sql_ok) / completed if completed else 0.0,
        "error_rate": len(errors) / completed if completed else 0.0,
        "avg_latency_s": round(sum(latencies) / completed, 3) if completed else 0.0,
        "p50_latency_s": round(percentile(latencies, 0.50), 3),
        "p95_latency_s": round(percentile(latencies, 0.95), 3),
        "throughput_qps": round(completed / total_wall_time_s, 3) if total_wall_time_s > 0 else 0.0,
        "total_wall_time_s": round(total_wall_time_s, 3),
    }


CACHE_COUNT_FIELDS = (
    "hits",
    "misses",
    "expired",
    "stale_hits",
    "stale_misses",
    "sets",
    "evictions",
)


def summarize_cache_stats(before: dict[str, dict], after: dict[str, dict]) -> dict:
    caches = {}
    totals = {field: 0 for field in CACHE_COUNT_FIELDS}

    for cache_name in sorted(set(before) | set(after)):
        before_stats = before.get(cache_name, {})
        after_stats = after.get(cache_name, {})
        delta = {
            field: max(int(after_stats.get(field, 0)) - int(before_stats.get(field, 0)), 0)
            for field in CACHE_COUNT_FIELDS
        }
        requests = delta["hits"] + delta["misses"] + delta["expired"]
        stale_requests = delta["stale_hits"] + delta["stale_misses"]
        for field in CACHE_COUNT_FIELDS:
            totals[field] += delta[field]
        caches[cache_name] = {
            "backend": after_stats.get("backend", before_stats.get("backend", "unknown")),
            **delta,
            "requests": requests,
            "stale_requests": stale_requests,
            "hit_ratio": round(delta["hits"] / requests, 6) if requests else 0.0,
            "expired_ratio": round(delta["expired"] / requests, 6) if requests else 0.0,
            "eviction_ratio": round(delta["evictions"] / delta["sets"], 6) if delta["sets"] else 0.0,
            "stale_hit_ratio": round(delta["stale_hits"] / stale_requests, 6) if stale_requests else 0.0,
        }

    total_requests = totals["hits"] + totals["misses"] + totals["expired"]
    total_stale_requests = totals["stale_hits"] + totals["stale_misses"]
    return {
        "total": {
            **totals,
            "requests": total_requests,
            "stale_requests": total_stale_requests,
            "hit_ratio": round(totals["hits"] / total_requests, 6) if total_requests else 0.0,
            "expired_ratio": round(totals["expired"] / total_requests, 6) if total_requests else 0.0,
            "eviction_ratio": round(totals["evictions"] / totals["sets"], 6) if totals["sets"] else 0.0,
            "stale_hit_ratio": round(totals["stale_hits"] / total_stale_requests, 6) if total_stale_requests else 0.0,
        },
        "by_cache": caches,
    }
