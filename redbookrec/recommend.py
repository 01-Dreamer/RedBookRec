from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .rerank import greedy_dpp
from .search import load_index, search


def popular_candidates(notes: pd.DataFrame, top_k: int) -> pd.DataFrame:
    out = notes.sort_values("popularity_score", ascending=False).head(top_k).copy()
    out["score"] = out["popularity_score"].fillna(0.0)
    out["recall_source"] = "popular_fallback"
    return out[["note_id", "raw_note_idx", "score", "recall_source"]]


def recommend(cfg: dict[str, Any], user_id: int = 0, query: str = "", top_k: int = 20) -> pd.DataFrame:
    processed = Path(cfg["paths"]["processed_dir"])
    notes = pd.read_parquet(processed / "notes.parquet")
    if query:
        index_path = Path(cfg["paths"]["indexes_dir"]) / "search_index.joblib"
        if not index_path.exists():
            index_path = Path(cfg["paths"]["indexes_dir"]) / "search_tfidf.joblib"
        if index_path.exists():
            candidates = search(load_index(index_path), query, max(top_k * 10, 100))
        else:
            candidates = popular_candidates(notes, max(top_k * 10, 100))
    else:
        tw_path = Path(cfg["paths"]["indexes_dir"]) / "twotower_candidates.joblib"
        if tw_path.exists():
            data = joblib.load(tw_path)
            candidates = pd.DataFrame(data.get(int(user_id), []))
            if candidates.empty:
                candidates = popular_candidates(notes, max(top_k * 10, 100))
        else:
            candidates = popular_candidates(notes, max(top_k * 10, 100))
    merged = candidates.merge(notes, on=["note_id", "raw_note_idx"], how="left")
    if "score" not in merged:
        merged["score"] = merged.get("popularity_score", 0.0)
    history_notes = set()
    history_path = processed / "user_history.parquet"
    if history_path.exists():
        history = pd.read_parquet(history_path)
        row = history[history["user_id"] == int(user_id)]
        if not row.empty:
            history_notes = set(int(x) for x in row.iloc[0]["history_note_ids"])
    return greedy_dpp(
        merged,
        score_col="score",
        top_k=top_k,
        diversity_weight=cfg.get("rerank", {}).get("diversity_weight", 0.25),
        history_note_ids=history_notes,
    )
