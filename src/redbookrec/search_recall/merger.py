from __future__ import annotations

from pathlib import Path

import pandas as pd


def _norm_group(df: pd.DataFrame, score_col: str, out_col: str) -> pd.Series:
    grouped = df.groupby("request_idx")[score_col]
    lo = grouped.transform("min")
    hi = grouped.transform("max")
    return ((df[score_col] - lo) / (hi - lo).replace(0, 1)).fillna(0.0).rename(out_col)


def merge_recall(cfg: dict) -> pd.DataFrame:
    dual_path = Path(cfg["infer"]["dual_output_path"])
    search_path = Path(cfg["infer"]["search_output_path"])
    frames: list[pd.DataFrame] = []
    if dual_path.exists():
        dual = pd.read_parquet(dual_path)
        dual["dual_norm"] = _norm_group(dual, "recall_score", "dual_norm")
        dual = dual.rename(columns={"recall_score": "dual_score", "recall_rank": "dual_rank"})
        frames.append(dual[["request_idx", "user_idx", "note_idx", "dual_score", "dual_norm", "label_click"]])
    if search_path.exists():
        search = pd.read_parquet(search_path)
        search["search_norm"] = _norm_group(search, "search_score", "search_norm")
        frames.append(search[["request_idx", "user_idx", "note_idx", "search_score", "search_norm", "label_click"]])
    if not frames:
        raise FileNotFoundError("no recall outputs found to merge")
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["request_idx", "user_idx", "note_idx"], how="outer", suffixes=("", "_search"))
    for col in ["dual_score", "dual_norm", "search_score", "search_norm", "label_click", "label_click_search"]:
        if col not in merged:
            merged[col] = 0.0
    merged["label_click"] = merged[["label_click", "label_click_search"]].max(axis=1).astype(int)
    w_dual = float(cfg.get("hybrid_recall", {}).get("w_dual", 0.7))
    w_search = float(cfg.get("hybrid_recall", {}).get("w_search", 0.3))
    merged["hybrid_score"] = w_dual * merged["dual_norm"].fillna(0.0) + w_search * merged["search_norm"].fillna(0.0)
    merged = merged.sort_values(["request_idx", "hybrid_score"], ascending=[True, False])
    top_k = int(cfg["infer"].get("top_k", 1000))
    merged["hybrid_rank"] = merged.groupby("request_idx").cumcount() + 1
    merged = merged[merged["hybrid_rank"] <= top_k]
    out_cols = ["request_idx", "user_idx", "note_idx", "dual_score", "search_score", "hybrid_score", "hybrid_rank", "label_click"]
    out = merged[out_cols].copy()
    path = Path(cfg["infer"]["hybrid_output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
