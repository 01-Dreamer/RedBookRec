from __future__ import annotations

import math
from typing import Any

import pandas as pd

from redbookrec.rerank.loss import dpp_objective_score, item_similarity


META_COLS = ["note_idx", "note_type", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id", "content_length"]


def _clean(value: Any) -> str:
    text = str(value or "UNK")
    return "UNK" if text in {"", "nan", "None"} else text


def _meta_lookup(note_meta: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if note_meta.empty or "note_idx" not in note_meta:
        return {}
    df = note_meta.copy()
    for col in META_COLS:
        if col not in df:
            df[col] = "UNK" if col.startswith("taxonomy") else 0
    out: dict[int, dict[str, Any]] = {}
    for row in df[META_COLS].itertuples(index=False):
        out[int(row.note_idx)] = {
            "note_type": int(float(row.note_type or 0)),
            "tax1": _clean(row.taxonomy1_id),
            "tax2": _clean(row.taxonomy2_id),
            "tax3": _clean(row.taxonomy3_id),
            "length_bucket": int(math.log1p(max(0.0, float(row.content_length or 0.0))) * 2),
        }
    return out


def greedy_dpp(
    group: pd.DataFrame,
    note_meta: pd.DataFrame,
    top_k: int,
    lambda_diversity: float,
    score_col: str = "sim_score",
) -> pd.DataFrame:
    if group.empty:
        return pd.DataFrame()
    meta = _meta_lookup(note_meta)
    candidates = group.sort_values(score_col, ascending=False).drop_duplicates("note_idx").to_dict("records")
    raw_scores = [float(row.get(score_col, 0.0) or 0.0) for row in candidates]
    lo, hi = min(raw_scores), max(raw_scores)
    for row, raw in zip(candidates, raw_scores):
        row["_rel"] = (raw - lo) / (hi - lo) if hi > lo else raw

    selected: list[dict] = []
    while candidates and len(selected) < int(top_k):
        best_i = 0
        best_score = -float("inf")
        for i, row in enumerate(candidates):
            cur_meta = meta.get(int(row["note_idx"]), {})
            diversity_penalty = max(
                (item_similarity(cur_meta, meta.get(int(chosen["note_idx"]), {})) for chosen in selected),
                default=0.0,
            )
            dpp_score = dpp_objective_score(float(row["_rel"]), diversity_penalty, lambda_diversity)
            if dpp_score > best_score:
                best_i = i
                best_score = dpp_score
        chosen = candidates.pop(best_i)
        chosen["dpp_score"] = float(best_score)
        chosen.pop("_rel", None)
        selected.append(chosen)

    out = pd.DataFrame(selected)
    if not out.empty:
        out["final_rank"] = range(1, len(out) + 1)
    return out
