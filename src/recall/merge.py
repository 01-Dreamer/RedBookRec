from __future__ import annotations

import pandas as pd


def merge_recall_frames(frames: list[pd.DataFrame], top_k_per_user: int = 300) -> pd.DataFrame:
    non_empty = [df for df in frames if df is not None and len(df)]
    if not non_empty:
        return pd.DataFrame(columns=["user_id", "note_id", "recall_score", "recall_source", "merged_recall_score"])
    recalls = pd.concat(non_empty, ignore_index=True)
    recalls["recall_score"] = pd.to_numeric(recalls["recall_score"], errors="coerce").fillna(0.0)
    recalls["source_rank"] = recalls.groupby(["user_id", "recall_source"])["recall_score"].rank(ascending=False, method="first")
    recalls["rank_score"] = 1.0 / (recalls["source_rank"] + 10.0)
    merged = (
        recalls.groupby(["user_id", "note_id"])
        .agg(
            merged_recall_score=("rank_score", "sum"),
            recall_score=("recall_score", "max"),
            recall_source=("recall_source", lambda x: "+".join(sorted(set(x)))),
        )
        .reset_index()
    )
    merged = merged.sort_values(["user_id", "merged_recall_score"], ascending=[True, False])
    return merged.groupby("user_id").head(top_k_per_user).reset_index(drop=True)
