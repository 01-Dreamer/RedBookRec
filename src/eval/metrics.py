from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(labels: Iterable[float], scores: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=float)
    s = np.asarray(list(scores), dtype=float)
    out: dict[str, float] = {}
    if len(np.unique(y)) > 1:
        out["auc"] = float(roc_auc_score(y, s))
        out["logloss"] = float(log_loss(y, np.clip(s, 1e-6, 1 - 1e-6)))
    return out


def ranking_metrics(
    ranked: pd.DataFrame,
    positives: dict[int, set[int]],
    ks: list[int] | None = None,
    score_col: str = "final_score",
) -> dict[str, float]:
    ks = ks or [10, 20, 50]
    metrics: dict[str, float] = {}
    users = [u for u in positives if positives[u]]
    if not users:
        return {f"recall@{k}": 0.0 for k in ks}
    grouped = ranked.sort_values(["user_id", score_col], ascending=[True, False]).groupby("user_id")
    coverage_items = set()
    avg_scores = []
    for k in ks:
        recalls = []
        hits = []
        ndcgs = []
        mrrs = []
        for user_id in users:
            pos = positives[user_id]
            if user_id not in grouped.groups:
                recalls.append(0.0)
                hits.append(0.0)
                ndcgs.append(0.0)
                mrrs.append(0.0)
                continue
            top = grouped.get_group(user_id).head(k)
            items = top["note_id"].astype(int).tolist()
            scores = top[score_col].astype(float).tolist() if score_col in top else []
            coverage_items.update(items)
            avg_scores.extend(scores)
            hit_positions = [idx for idx, item in enumerate(items) if item in pos]
            recalls.append(len(set(items) & pos) / len(pos))
            hits.append(float(bool(hit_positions)))
            dcg = sum(1.0 / math.log2(idx + 2) for idx in hit_positions)
            ideal = sum(1.0 / math.log2(idx + 2) for idx in range(min(len(pos), k)))
            ndcgs.append(dcg / ideal if ideal > 0 else 0.0)
            mrrs.append(1.0 / (hit_positions[0] + 1) if hit_positions else 0.0)
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"hitrate@{k}"] = float(np.mean(hits))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
        metrics[f"mrr@{k}"] = float(np.mean(mrrs))
    metrics["coverage"] = float(len(coverage_items))
    metrics["avg_recommendation_score"] = float(np.mean(avg_scores)) if avg_scores else 0.0
    return metrics
