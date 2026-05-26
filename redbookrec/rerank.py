from __future__ import annotations

import numpy as np
import pandas as pd


def taxonomy_similarity(a: pd.Series, b: pd.Series) -> float:
    cols = ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
    matches = 0
    total = 0
    for col in cols:
        av = a.get(col, "")
        bv = b.get(col, "")
        if av or bv:
            total += 1
            matches += int(av == bv and av != "")
    return matches / total if total else 0.0


def text_signature(row: pd.Series) -> str:
    title = str(row.get("note_title", "") or "").strip()
    content = str(row.get("note_content", "") or "").strip()
    return (title[:40] + "|" + content[:60]).lower()


def prefilter_items(items: pd.DataFrame, history_note_ids: set[int] | None = None) -> pd.DataFrame:
    pool = items.copy()
    history_note_ids = history_note_ids or set()
    if "note_id" in pool and history_note_ids:
        pool = pool[~pool["note_id"].astype(int).isin(history_note_ids)]
    if "raw_note_idx" in pool:
        pool = pool.drop_duplicates(subset=["raw_note_idx"], keep="first")
    if "note_id" in pool:
        pool = pool.drop_duplicates(subset=["note_id"], keep="first")
    if {"note_title", "note_content"}.issubset(pool.columns):
        pool["_text_signature"] = pool.apply(text_signature, axis=1)
        pool = pool.drop_duplicates(subset=["_text_signature"], keep="first")
        pool = pool.drop(columns=["_text_signature"])
    return pool


def greedy_dpp(
    items: pd.DataFrame,
    score_col: str = "score",
    top_k: int = 20,
    diversity_weight: float = 0.25,
    history_note_ids: set[int] | None = None,
) -> pd.DataFrame:
    if items.empty:
        return items
    pool = prefilter_items(items, history_note_ids=history_note_ids).reset_index(drop=True)
    if pool.empty:
        pool = items.copy().reset_index(drop=True)
    selected: list[int] = []
    remaining = set(range(len(pool)))
    scores = pool[score_col].fillna(0.0).astype(float)
    if scores.max() > scores.min():
        rel = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        rel = scores * 0 + 0.5
    while remaining and len(selected) < top_k:
        best_idx = None
        best_score = -1e18
        for idx in remaining:
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(taxonomy_similarity(pool.iloc[idx], pool.iloc[j]) for j in selected)
            value = (1.0 - diversity_weight) * rel.iloc[idx] - diversity_weight * diversity_penalty
            if value > best_score:
                best_score = float(value)
                best_idx = idx
        selected.append(int(best_idx))
        remaining.remove(int(best_idx))
    out = pool.iloc[selected].copy()
    out["rerank_position"] = range(1, len(out) + 1)
    return out
