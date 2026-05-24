from __future__ import annotations

import pandas as pd


def diversity_rerank(
    ranked: pd.DataFrame,
    notes: pd.DataFrame,
    history: list[int],
    top_k: int = 20,
    max_per_category: int = 4,
) -> pd.DataFrame:
    if len(ranked) == 0:
        return ranked
    meta = notes[["note_id", "category", "topic", "title", "content"]].drop_duplicates("note_id")
    df = ranked.merge(meta, on="note_id", how="left")
    seen = set(int(x) for x in history)
    category_count: dict[str, int] = {}
    rows = []
    for row in df.sort_values("final_score", ascending=False).itertuples(index=False):
        reason = []
        if int(row.note_id) in seen:
            continue
        category = str(getattr(row, "category", "") or "")
        if category:
            if category_count.get(category, 0) >= max_per_category:
                continue
            category_count[category] = category_count.get(category, 0) + 1
            reason.append(f"category_diversity:{category_count[category]}/{max_per_category}")
        reason.append("history_filtered")
        item = row._asdict()
        item["rerank_reason"] = "; ".join(reason)
        rows.append(item)
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)
