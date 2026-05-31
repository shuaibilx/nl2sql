"""
[评测] 检索质量指标 — Recall@K, Precision@K, MRR

适用于字段召回、值召回、指标召回三个维度。
"""


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """Recall@K：期望的结果中有多少被召回了

    Args:
        retrieved_ids: 实际召回的 ID 列表（已按相关性排序）
        expected_ids: 期望的正确 ID 列表
        k: 取前 K 个结果计算

    Returns:
        Recall@K 值，0.0 ~ 1.0
    """
    if not expected_ids:
        return 1.0  # 无期望结果时，视为完全召回
    retrieved_k = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    return len(retrieved_k & expected_set) / len(expected_set)


def precision_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int = 5) -> float:
    """Precision@K：前 K 个结果中有多少是正确的

    Args:
        retrieved_ids: 实际召回的 ID 列表
        expected_ids: 期望的正确 ID 列表
        k: 取前 K 个结果计算

    Returns:
        Precision@K 值，0.0 ~ 1.0
    """
    if k == 0:
        return 0.0
    retrieved_k = set(retrieved_ids[:k])
    expected_set = set(expected_ids)
    return len(retrieved_k & expected_set) / k


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """MRR（Mean Reciprocal Rank）：第一个正确结果的排名倒数

    Args:
        retrieved_ids: 实际召回的 ID 列表
        expected_ids: 期望的正确 ID 列表

    Returns:
        MRR 值，0.0 ~ 1.0
    """
    expected_set = set(expected_ids)
    for i, item in enumerate(retrieved_ids):
        if item in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retrieval(
    retrieved_columns: list[str],
    retrieved_values: list[str],
    retrieved_metrics: list[str],
    expected_columns: list[str],
    expected_values: list[str],
    expected_metrics: list[str],
    k: int = 5,
) -> dict:
    """综合评估三路召回的检索质量

    Returns:
        包含各维度指标的字典
    """
    return {
        "column_recall@k": recall_at_k(retrieved_columns, expected_columns, k),
        "column_precision@k": precision_at_k(retrieved_columns, expected_columns, k),
        "column_mrr": mrr(retrieved_columns, expected_columns),
        "value_recall@k": recall_at_k(retrieved_values, expected_values, k),
        "value_precision@k": precision_at_k(retrieved_values, expected_values, k),
        "value_mrr": mrr(retrieved_values, expected_metrics),
        "metric_recall@k": recall_at_k(retrieved_metrics, expected_metrics, k),
        "metric_precision@k": precision_at_k(retrieved_metrics, expected_metrics, k),
        "metric_mrr": mrr(retrieved_metrics, expected_metrics),
        "overall_recall": (
            recall_at_k(retrieved_columns, expected_columns, k)
            + recall_at_k(retrieved_values, expected_values, k)
            + recall_at_k(retrieved_metrics, expected_metrics, k)
        ) / 3,
    }
