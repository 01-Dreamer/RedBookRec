from __future__ import annotations

import numpy as np


def hit_rate_at_k(labels: list[int], k: int) -> float:
    return float(any(labels[:k]))


def mrr_at_k(labels: list[int], k: int) -> float:
    for idx, label in enumerate(labels[:k], start=1):
        if label:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(labels: list[int], k: int) -> float:
    gains = np.asarray(labels[:k], dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(gains)[::-1]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(labels: list[int], k: int) -> float:
    total_relevant = sum(1 for label in labels if label)
    if total_relevant == 0:
        return 0.0
    retrieved_relevant = sum(1 for label in labels[:k] if label)
    return retrieved_relevant / total_relevant


def evaluate_ranked_groups(groups: list[list[int]], ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not groups:
        return {f"{name}@{k}": 0.0 for k in ks for name in ["Recall", "HitRate", "MRR", "NDCG"]}
    for k in ks:
        metrics[f"Recall@{k}"] = float(np.mean([recall_at_k(g, k) for g in groups]))
        metrics[f"HitRate@{k}"] = float(np.mean([hit_rate_at_k(g, k) for g in groups]))
        metrics[f"MRR@{k}"] = float(np.mean([mrr_at_k(g, k) for g in groups]))
        metrics[f"NDCG@{k}"] = float(np.mean([ndcg_at_k(g, k) for g in groups]))
    return metrics
