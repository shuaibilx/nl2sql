from __future__ import annotations

import argparse
import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.clients.mysql_client_manager import dw_mysql_client_manager


DEFAULT_CASES = PROJECT_ROOT / "eval" / "benchmark_cases.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute benchmark expected_sql and refresh expected_result."
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--output", default=None)
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def normalize_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return float(value)
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_rows(rows: list[dict[str, Any]]) -> Any:
    if len(rows) == 1 and len(rows[0]) == 1:
        return normalize_value(next(iter(rows[0].values())))
    return [[normalize_value(value) for value in row.values()] for row in rows]


def assert_readonly_sql(sql: str) -> None:
    stripped = sql.strip().lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        raise ValueError("Only SELECT/WITH expected_sql is allowed")
    forbidden = [" insert ", " update ", " delete ", " drop ", " alter ", " truncate "]
    padded = f" {stripped} "
    if any(token in padded for token in forbidden):
        raise ValueError("expected_sql must be read-only")


async def refresh_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dw_mysql_client_manager.init()
    try:
        async with dw_mysql_client_manager.session_factory() as session:
            for case in cases:
                sql = case["expected_sql"]
                assert_readonly_sql(sql)
                result = await session.execute(text(sql))
                rows = [dict(row) for row in result.mappings().fetchall()]
                case["expected_result"] = normalize_rows(rows)
    finally:
        await dw_mysql_client_manager.close()
    return cases


async def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases)
    cases = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or []
    refreshed = await refresh_cases(cases)

    if args.in_place:
        output_path = cases_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = cases_path.with_name(f"{cases_path.stem}_with_results.yaml")

    output_path.write_text(
        yaml.safe_dump(refreshed, allow_unicode=True, sort_keys=False, width=120),
        encoding="utf-8",
    )
    print(f"refreshed {len(refreshed)} cases -> {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
