"""
[评测] 并发指标收集器 — 聚合多次请求的延迟/错误，计算分位数、吞吐量、错误率

用于 run_concurrent_eval.py 和 load_test.py 的指标收集。
"""
import time
from collections import Counter
from dataclasses import dataclass, field


def percentile(sorted_values: list[float], p: float) -> float:
    """计算第 p 百分位数（已排序的列表）"""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])


@dataclass
class ConcurrencyMetrics:
    """并发指标收集器

    与 LatencyTracker（单请求节点级计时）不同，
    本类聚合跨多个并发请求的系统级指标。
    """
    latencies: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    wall_clock_start: float = 0.0
    wall_clock_end: float = 0.0

    def record_success(self, latency_s: float):
        self.latencies.append(latency_s)

    def record_error(self, error_msg: str):
        self.errors.append(error_msg)

    def start(self):
        self.wall_clock_start = time.time()

    def stop(self):
        self.wall_clock_end = time.time()

    @property
    def total_wall_clock(self) -> float:
        return self.wall_clock_end - self.wall_clock_start

    @property
    def total_requests(self) -> int:
        return len(self.latencies) + len(self.errors)

    @property
    def error_rate(self) -> float:
        total = self.total_requests
        return len(self.errors) / total if total > 0 else 0.0

    @property
    def throughput_rps(self) -> float:
        duration = self.total_wall_clock
        return len(self.latencies) / duration if duration > 0 else 0.0

    @property
    def p50(self) -> float:
        return percentile(sorted(self.latencies), 50)

    @property
    def p95(self) -> float:
        return percentile(sorted(self.latencies), 95)

    @property
    def p99(self) -> float:
        return percentile(sorted(self.latencies), 99)

    def error_categories(self) -> dict[str, int]:
        """按错误类型前缀分类"""
        categories = Counter()
        for err in self.errors:
            prefix = err.split(":")[0].strip() if ":" in err else err[:50]
            categories[prefix] += 1
        return dict(categories)

    def summary_dict(self) -> dict:
        """生成汇总字典"""
        return {
            "total_requests": self.total_requests,
            "successful": len(self.latencies),
            "failed": len(self.errors),
            "error_rate": round(self.error_rate, 4),
            "wall_clock_s": round(self.total_wall_clock, 3),
            "throughput_rps": round(self.throughput_rps, 3),
            "latency_p50_s": round(self.p50, 3),
            "latency_p95_s": round(self.p95, 3),
            "latency_p99_s": round(self.p99, 3),
            "latency_min_s": round(min(self.latencies), 3) if self.latencies else 0,
            "latency_max_s": round(max(self.latencies), 3) if self.latencies else 0,
            "latency_mean_s": round(sum(self.latencies) / len(self.latencies), 3) if self.latencies else 0,
            "error_categories": self.error_categories(),
        }
