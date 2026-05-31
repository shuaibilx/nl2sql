"""
[评测] 性能指标 — 各节点延迟、端到端延迟、缓存命中率

通过收集 stream 事件中的时间戳计算。
"""
import time


class LatencyTracker:
    """延迟追踪器：记录各节点的执行耗时"""

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.node_times: dict[str, list[float]] = {}  # node_name → [start, end]
        self._current_node: str | None = None
        self._current_start: float = 0

    def start(self):
        """开始计时"""
        self.start_time = time.time()

    def stop(self):
        """停止计时"""
        self.end_time = time.time()

    def on_progress(self, event: dict):
        """处理 progress 事件，记录节点延迟

        Args:
            event: {"type": "progress", "step": "xxx", "status": "running/success/error"}
        """
        step = event.get("step", "")
        status = event.get("status", "")

        if status == "running":
            self._current_node = step
            self._current_start = time.time()
        elif status in ("success", "error", "cancelled", "warning") and self._current_node:
            elapsed = time.time() - self._current_start
            if self._current_node not in self.node_times:
                self.node_times[self._current_node] = []
            self.node_times[self._current_node].append(elapsed)
            self._current_node = None

    @property
    def total_latency(self) -> float:
        """端到端总延迟（秒）"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

    def summary(self) -> dict:
        """生成延迟摘要

        Returns:
            包含各节点延迟和总延迟的字典
        """
        result = {"total_latency_s": round(self.total_latency, 3)}
        for node, times in self.node_times.items():
            result[f"{node}_latency_s"] = round(sum(times), 3)
        return result
