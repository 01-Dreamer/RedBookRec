from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


LABEL_COLUMNS = ["click", "like", "collect", "comment", "share"]

CANONICAL_ALIASES = {
    "user_id": ["user_id", "user_idx"],
    "note_id": ["note_id", "note_idx"],
    "candidate_note_ids": ["candidate_note_ids", "rec_results", "search_results"],
    "history_note_ids": ["history_note_ids", "recent_clicked_note_idxs"],
    "query": ["query"],
    "title": ["title", "note_title"],
    "content": ["content", "note_content"],
    "category": ["category", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id", "note_type"],
}


def find_column(columns: Iterable[str], canonical_name: str) -> str | None:
    column_set = set(columns)
    for candidate in CANONICAL_ALIASES.get(canonical_name, [canonical_name]):
        if candidate in column_set:
            return candidate
    return None


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, float) and np.isnan(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if np.isnan(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def parse_recommendation_frame(df: pd.DataFrame, split: str, history_max_len: int) -> pd.DataFrame:
    user_col = find_column(df.columns, "user_id")
    history_col = find_column(df.columns, "history_note_ids")
    detail_col = "rec_result_details_with_idx"
    if user_col is None or history_col is None or detail_col not in df.columns:
        raise ValueError(
            "Recommendation data requires user_idx/user_id, recent_clicked_note_idxs/history_note_ids, "
            "and rec_result_details_with_idx. Run scripts/00_inspect_dataset.py to inspect the schema."
        )

    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        raw = row._asdict()
        user_id = safe_int(raw[user_col])
        history = [safe_int(x) for x in to_list(raw[history_col]) if safe_int(x) >= 0][-history_max_len:]
        query = str(raw.get("query") or "")
        request_idx = safe_int(raw.get("request_idx"), default=len(rows))
        session_idx = safe_int(raw.get("session_idx"), default=-1)
        for detail in to_list(raw[detail_col]):
            if hasattr(detail, "as_py"):
                detail = detail.as_py()
            if not isinstance(detail, dict):
                continue
            note_id = safe_int(detail.get("note_idx", detail.get("note_id")))
            if note_id < 0:
                continue
            parsed = {
                "split": split,
                "request_id": request_idx,
                "session_id": session_idx,
                "user_id": user_id,
                "note_id": note_id,
                "history_note_ids": history,
                "query": query,
                "position": safe_int(detail.get("position"), default=-1),
                "request_timestamp": safe_float(detail.get("request_timestamp"), default=0.0),
                "page_time": safe_float(detail.get("page_time"), default=0.0),
            }
            for label in LABEL_COLUMNS:
                parsed[label] = float(safe_int(detail.get(label), default=0) > 0)
            rows.append(parsed)
    return pd.DataFrame(rows)


def parse_notes_frame(df: pd.DataFrame) -> pd.DataFrame:
    note_col = find_column(df.columns, "note_id")
    if note_col is None:
        raise ValueError("Notes data requires note_idx/note_id.")
    out = pd.DataFrame({"note_id": df[note_col].map(safe_int)})
    out["title"] = df[find_column(df.columns, "title")].fillna("").astype(str) if find_column(df.columns, "title") else ""
    out["content"] = df[find_column(df.columns, "content")].fillna("").astype(str) if find_column(df.columns, "content") else ""
    for src, dst in [
        ("taxonomy1_id", "category"),
        ("taxonomy2_id", "subcategory"),
        ("taxonomy3_id", "topic"),
        ("note_type", "note_type"),
    ]:
        out[dst] = df[src].fillna("").astype(str) if src in df.columns else ""
    for src in ["imp_num", "click_num", "like_num", "collect_num", "comment_num", "share_num", "view_time"]:
        out[src] = pd.to_numeric(df[src], errors="coerce").fillna(0.0) if src in df.columns else 0.0
    out["tags"] = out[["category", "subcategory", "topic"]].agg(lambda x: [v for v in x if v], axis=1)
    out["text"] = (out["title"] + " " + out["content"]).str.strip()
    return out.drop_duplicates("note_id")


def parse_user_frame(df: pd.DataFrame) -> pd.DataFrame:
    user_col = find_column(df.columns, "user_id")
    if user_col is None:
        raise ValueError("User feature data requires user_idx/user_id.")
    out = df.copy()
    out = out.rename(columns={user_col: "user_id"})
    return out.drop_duplicates("user_id")
