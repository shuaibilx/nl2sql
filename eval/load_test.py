"""
[评测] HTTP 压测脚本 — 对运行中的 FastAPI 服务发并发请求，测量端到端 API 性能

使用方式：
    # 先启动服务
    python main.py

    # 压测（单个并发度）
    python eval/load_test.py --concurrency 5

    # 压测（多轮递增并发度）
    python eval/load_test.py --concurrency 1 2 5 10

    # 指定查询
    python eval/load_test.py --concurrency 1 2 5 --query-file eval/test_cases.yaml

输出文件：
    eval/reports/load_test_YYYYMMDD_HHMMSS.md  — 压测报告
"""
import asyncio
import json
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

EVAL_DIR = Path(__file__).parent


# ─── RequestResult ────────────────────────────────────────────────

class RequestResult:
    """单次 HTTP 请求的结果"""
    __slots__ = (
        "request_id", "query", "session_id", "total_latency_s",
        "first_byte_latency_s", "success", "error", "had_interrupt",
        "interrupt_latency_s", "resume_latency_s", "status_code",
    )

    def __init__(self, **kwargs):
        for k in self.__slots__:
            setattr(self, k, kwargs.get(k))
        if self.success is None:
            self.success = True
        if self.total_latency_s is None:
            self.total_latency_s = 0.0
        if self.first_byte_latency_s is None:
            self.first_byte_latency_s = 0.0


# ─── SSE 解析 ─────────────────────────────────────────────────────

async def consume_sse_stream(resp) -> list[dict]:
    """消费 SSE 流，返回解析后的事件列表"""
    events = []
    buffer = ""
    async for chunk in resp.content:
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    events.append(payload)
                except json.JSONDecodeError:
                    pass
    return events


# ─── 单请求 ───────────────────────────────────────────────────────

async def send_query_request(
    session,
    base_url: str,
    query: str,
    request_id: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    """发送单次查询到 /api/query，消费完整 SSE 流

    如果遇到 interrupt 事件，自动发送 resume 请求。
    """
    import aiohttp
    async with semaphore:
        start = time.time()
        first_byte_time = None
        interrupt_time = None
        session_id = None
        had_interrupt = False

        try:
            # Step 1: 发送查询
            async with session.post(
                f"{base_url}/api/query",
                json={"query": query},
                timeout=aiohttp.ClientTimeout(total=180),
            ) as resp:
                if resp.status != 200:
                    return RequestResult(
                        request_id=request_id, query=query,
                        total_latency_s=time.time() - start,
                        success=False, error=f"HTTP {resp.status}",
                        status_code=resp.status,
                    )

                events = await consume_sse_stream(resp)
                first_byte_time = time.time()

                for event in events:
                    if event.get("type") == "interrupt":
                        had_interrupt = True
                        interrupt_time = time.time()
                        session_id = event.get("session_id")

            # Step 2: 如果有 interrupt，自动 resume
            if had_interrupt and session_id:
                resume_start = time.time()
                async with session.post(
                    f"{base_url}/api/query/resume",
                    json={"session_id": session_id, "confirmed": True},
                    timeout=aiohttp.ClientTimeout(total=180),
                ) as resp:
                    await consume_sse_stream(resp)
                resume_end = time.time()

                return RequestResult(
                    request_id=request_id, query=query, session_id=session_id,
                    total_latency_s=resume_end - start,
                    first_byte_latency_s=(first_byte_time or start) - start,
                    success=True, had_interrupt=True,
                    interrupt_latency_s=interrupt_time - start if interrupt_time else 0,
                    resume_latency_s=resume_end - resume_start,
                    status_code=200,
                )

            end = time.time()
            return RequestResult(
                request_id=request_id, query=query,
                total_latency_s=end - start,
                first_byte_latency_s=(first_byte_time or end) - start,
                success=True, status_code=200,
            )

        except Exception as e:
            return RequestResult(
                request_id=request_id, query=query,
                total_latency_s=time.time() - start,
                success=False, error=f"{type(e).__name__}: {str(e)[:200]}",
            )


# ─── 批量压测 ─────────────────────────────────────────────────────

async def run_load_test(
    base_url: str,
    queries: list[str],
    concurrency: int,
) -> dict:
    """运行单轮压测"""
    import aiohttp

    semaphore = asyncio.Semaphore(concurrency)
    print(f"\n  并发度={concurrency}, 请求数={len(queries)}")

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_query_request(session, base_url, q, i + 1, semaphore)
            for i, q in enumerate(queries)
        ]

        start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.time() - start

    # 处理异常
    final_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            final_results.append(RequestResult(
                request_id=i + 1, query=queries[i],
                total_latency_s=0, success=False,
                error=f"{type(r).__name__}: {str(r)[:200]}",
            ))
        else:
            final_results.append(r)

    return _build_report(final_results, concurrency, duration)


def _build_report(results: list, concurrency: int, duration: float) -> dict:
    """汇总压测结果"""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    latencies = sorted([r.total_latency_s for r in successful])
    first_byte_latencies = sorted([r.first_byte_latency_s for r in successful])

    error_categories = {}
    for r in failed:
        err_type = r.error.split(":")[0].strip() if r.error and ":" in r.error else (r.error or "Unknown")[:50]
        error_categories[err_type] = error_categories.get(err_type, 0) + 1

    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "duration_s": round(duration, 3),
        "successful": len(successful),
        "failed": len(failed),
        "error_rate": round(len(failed) / len(results), 4) if results else 0,
        "throughput_rps": round(len(successful) / duration, 3) if duration > 0 else 0,
        "latency_p50_s": round(_percentile(latencies, 50), 3),
        "latency_p95_s": round(_percentile(latencies, 95), 3),
        "latency_p99_s": round(_percentile(latencies, 99), 3),
        "latency_min_s": round(min(latencies), 3) if latencies else 0,
        "latency_max_s": round(max(latencies), 3) if latencies else 0,
        "latency_mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "first_byte_p50_s": round(_percentile(first_byte_latencies, 50), 3),
        "first_byte_p95_s": round(_percentile(first_byte_latencies, 95), 3),
        "error_categories": error_categories,
        "interrupt_count": sum(1 for r in results if r.had_interrupt),
        "results": results,
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


# ─── 报告生成 ─────────────────────────────────────────────────────

def generate_load_test_report(reports: list[dict], output_dir: Path) -> str:
    """生成压测报告"""
    lines = []
    lines.append("# NL2SQL HTTP 压测报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**测试模式**: HTTP 并发压测（含 interrupt/resume）")
    lines.append("")

    # 总览表
    lines.append("## 并发度 vs 性能")
    lines.append("")
    lines.append("| 并发度 | 请求总数 | 成功 | 失败 | 错误率 | 吞吐量(req/s) | p50(s) | p95(s) | p99(s) | 首字节p50(s) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        lines.append(
            f"| {r['concurrency']} "
            f"| {r['total_requests']} "
            f"| {r['successful']} "
            f"| {r['failed']} "
            f"| {r['error_rate']:.1%} "
            f"| {r['throughput_rps']:.3f} "
            f"| {r['latency_p50_s']:.1f} "
            f"| {r['latency_p95_s']:.1f} "
            f"| {r['latency_p99_s']:.1f} "
            f"| {r['first_byte_p50_s']:.1f} |"
        )
    lines.append("")

    # 错误分析
    has_errors = any(r["failed"] > 0 for r in reports)
    if has_errors:
        lines.append("## 错误分析")
        lines.append("")
        for r in reports:
            if r["error_categories"]:
                lines.append(f"### 并发度 {r['concurrency']}")
                lines.append("")
                for err_type, count in r["error_categories"].items():
                    lines.append(f"- `{err_type}`: {count} 次")
                lines.append("")

    # 逐请求详情（每个并发度）
    for r in reports:
        lines.append(f"## 并发度 {r['concurrency']} 详情")
        lines.append("")
        lines.append("| 编号 | 查询 | 成功 | 延迟(s) | 首字节(s) | interrupt | 错误 |")
        lines.append("|---|---|---|---|---|---|---|")
        for res in r["results"]:
            interrupt_mark = "Y" if res.had_interrupt else "-"
            error = res.error[:30] if res.error else "-"
            lines.append(
                f"| {res.request_id} "
                f"| {res.query[:25]} "
                f"| {'OK' if res.success else 'FAIL'} "
                f"| {res.total_latency_s:.1f} "
                f"| {res.first_byte_latency_s:.1f} "
                f"| {interrupt_mark} "
                f"| {error} |"
            )
        lines.append("")

    # 写入文件
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return str(report_path)


# ─── main ─────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="NL2SQL HTTP 压测工具")
    parser.add_argument("--url", default="http://localhost:8080", help="FastAPI 服务地址")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 5],
                        help="并发度列表（如 1 2 5 10）")
    parser.add_argument("--query-file", default=str(EVAL_DIR / "test_cases.yaml"),
                        help="测试用例文件（YAML）")
    parser.add_argument("--queries", nargs="+", help="直接指定查询列表")
    parser.add_argument("--report", default=str(EVAL_DIR / "reports"), help="报告输出目录")
    args = parser.parse_args()

    report_dir = Path(args.report).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    # 加载查询
    if args.queries:
        queries = args.queries
    else:
        with open(args.query_file, "r", encoding="utf-8") as f:
            test_cases = yaml.safe_load(f)
        queries = [case["query"] for case in test_cases]

    print(f"查询数: {len(queries)}")
    print(f"并发度: {args.concurrency}")
    print(f"目标: {args.url}")

    # 运行各并发度
    reports = []
    for concurrency in sorted(args.concurrency):
        report = await run_load_test(args.url, queries, concurrency)
        reports.append(report)

        # 打印单轮摘要
        print(f"  并发度={concurrency}: "
              f"成功={report['successful']}/{report['total_requests']}, "
              f"吞吐量={report['throughput_rps']:.3f} req/s, "
              f"p50={report['latency_p50_s']:.1f}s, "
              f"p95={report['latency_p95_s']:.1f}s, "
              f"错误率={report['error_rate']:.1%}")

    # 生成报告
    report_path = generate_load_test_report(reports, report_dir)
    print(f"\n{'='*50}")
    print(f"压测完成！报告: {report_path}")


if __name__ == "__main__":
    try:
        import aiohttp
    except ImportError:
        print("需要安装 aiohttp: pip install aiohttp")
        sys.exit(1)

    import yaml
    asyncio.run(main())
