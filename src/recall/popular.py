from __future__ import annotations

import pandas as pd


class PopularRecall:
    def __init__(self, top_k: int = 200) -> None:
        self.top_k = top_k
        self.popular_notes: list[tuple[int, float]] = []

    def fit(self, notes: pd.DataFrame, train: pd.DataFrame | None = None) -> "PopularRecall":
        if train is not None and len(train):
            clicks = train.groupby("note_id")["click"].sum()
            imps = train.groupby("note_id")["click"].count()
            train_score = (clicks + 1.0) / (imps + 5.0) + clicks.map(lambda x: 0.01 * x)
            score_df = train_score.rename("train_score").reset_index()
            merged = notes[["note_id", "popularity_score"]].merge(score_df, on="note_id", how="left")
            merged["score"] = merged["popularity_score"].fillna(0) + merged["train_score"].fillna(0)
        else:
            merged = notes[["note_id", "popularity_score"]].copy()
            merged["score"] = merged["popularity_score"].fillna(0)
        self.popular_notes = [
            (int(row.note_id), float(row.score))
            for row in merged.sort_values("score", ascending=False).head(self.top_k * 5).itertuples()
        ]
        return self

    def recommend(self, history: list[int], top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.top_k
        seen = set(int(x) for x in history)
        rows = []
        for note_id, score in self.popular_notes:
            if note_id in seen:
                continue
            rows.append({"note_id": note_id, "recall_score": score, "recall_source": "popular"})
            if len(rows) >= top_k:
                break
        return rows
