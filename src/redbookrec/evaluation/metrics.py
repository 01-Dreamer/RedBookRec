from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from redbookrec.utils.io import write_json


def _rank_metrics(df: pd.DataFrame, k: int) -> dict[str, float]:
    recalls = []
    hits = []
    mrrs = []
    ndcgs = []
    maps = []
    for _, group in df.groupby("request_idx"):
        g = group.head(k)
        labels = g["label_click"].astype(int).to_numpy()
        total_pos = max(1, int(group["label_click"].sum()))
        hit_positions = np.flatnonzero(labels > 0)
        hits.append(float(len(hit_positions) > 0))
        recalls.append(float(labels.sum() / total_pos))
        mrrs.append(float(1.0 / (hit_positions[0] + 1)) if len(hit_positions) else 0.0)
        dcg = sum(float(rel) / math.log2(i + 2) for i, rel in enumerate(labels))
        ideal = sorted(group["label_click"].astype(int).tolist(), reverse=True)[:k]
        idcg = sum(float(rel) / math.log2(i + 2) for i, rel in enumerate(ideal))
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)
        precisions = [float(labels[: i + 1].sum() / (i + 1)) for i in hit_positions]
        maps.append(float(sum(precisions) / min(total_pos, k)) if precisions else 0.0)
    return {
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"HitRate@{k}": float(np.mean(hits)) if hits else 0.0,
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
    }


def evaluate_dataframe(df: pd.DataFrame, score_col: str, ks: list[int]) -> dict[str, float]:
    if df.empty:
        return {"rows": 0}
    df = df.sort_values(["request_idx", score_col], ascending=[True, False]).copy()
    metrics: dict[str, float] = {"rows": int(len(df)), "requests": int(df["request_idx"].nunique()), "positives": int(df["label_click"].sum())}
    for k in ks:
        metrics.update(_rank_metrics(df, k))
    y = df["label_click"].astype(int)
    if y.nunique() > 1:
        pred = df[score_col].astype(float)
        lo, hi = pred.min(), pred.max()
        prob = (pred - lo) / (hi - lo) if hi > lo else pred * 0 + 0.5
        metrics["AUC"] = float(roc_auc_score(y, prob))
        metrics["LogLoss"] = float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6)))
    return metrics


def evaluate_stage(cfg: dict, stage: str) -> dict[str, float]:
    stage_map = {
        "recall": (cfg.get("infer", {}).get("dual_output_path", "outputs/recall/test_top1000.parquet"), "recall_score", [50, 100, 500, 1000], "outputs/recall/recall_metrics.json"),
        "search_recall": (cfg.get("infer", {}).get("search_output_path", "outputs/search_recall/test_top1000.parquet"), "search_score", [50, 100, 500, 1000], "outputs/search_recall/search_metrics.json"),
        "hybrid_recall": (cfg.get("infer", {}).get("hybrid_output_path", "outputs/hybrid_recall/test_top1000.parquet"), "hybrid_score", [50, 100, 500, 1000], "outputs/hybrid_recall/hybrid_metrics.json"),
        "prerank": (cfg.get("infer", {}).get("output_path", "outputs/prerank/test_top200.parquet"), "dcn_score", [10, 50, 100, 200], cfg.get("paths", {}).get("metrics", "outputs/prerank/prerank_metrics.json")),
        "rank": (cfg.get("infer", {}).get("output_path", "outputs/rank/test_top50.parquet"), "sim_score", [10, 20, 50], cfg.get("paths", {}).get("metrics", "outputs/rank/rank_metrics.json")),
        "rerank": (cfg.get("infer", {}).get("output_path", "outputs/rerank/test_top10.parquet"), "dpp_score", [10], cfg.get("paths", {}).get("metrics", "outputs/rerank/rerank_metrics.json")),
    }
    if stage not in stage_map:
        raise ValueError(f"unknown stage: {stage}")
    path, score_col, ks, metrics_path = stage_map[stage]
    df = pd.read_parquet(path)
    metrics = evaluate_dataframe(df, score_col, ks)
    write_json(metrics_path, metrics)
    return metrics
