from __future__ import annotations

from typing import Any

import pandas as pd

from redbookrec.data.load_qilin import normalize_detail, parse_nested, read_dataset_split


REC_COLUMNS = ["recent_clicked_note_idxs", "request_idx", "session_idx", "user_idx", "query", "rec_result_details_with_idx"]


def read_recommendation(dataset_dir: str, split: str, max_requests: int | None = None) -> pd.DataFrame:
    return read_dataset_split(dataset_dir, split, columns=REC_COLUMNS, max_rows=max_requests)


def clicked_set(details: list[Any]) -> set[int]:
    out: set[int] = set()
    for item in details:
        detail = normalize_detail(item)
        if int(detail.get("click", 0) or 0) > 0:
            try:
                out.add(int(detail.get("note_idx")))
            except Exception:
                pass
    return out


def expand_recommendation_requests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for base in df.itertuples(index=False):
        details = parse_nested(getattr(base, "rec_result_details_with_idx", []))
        history = parse_nested(getattr(base, "recent_clicked_note_idxs", []))
        for item in details:
            detail = normalize_detail(item)
            if not detail:
                continue
            try:
                note_idx = int(detail.get("note_idx"))
            except Exception:
                continue
            rows.append(
                {
                    "request_idx": int(getattr(base, "request_idx", -1)),
                    "session_idx": int(getattr(base, "session_idx", -1)),
                    "user_idx": int(getattr(base, "user_idx", -1)),
                    "query": getattr(base, "query", "") or "",
                    "recent_clicked_note_idxs": history,
                    "note_idx": note_idx,
                    "position": int(detail.get("position", 0) or 0),
                    "label_click": int(float(detail.get("click", 0) or 0) > 0),
                    "like": int(float(detail.get("like", 0) or 0) > 0),
                    "collect": int(float(detail.get("collect", 0) or 0) > 0),
                    "comment": int(float(detail.get("comment", 0) or 0) > 0),
                    "share": int(float(detail.get("share", 0) or 0) > 0),
                    "page_time": float(detail.get("page_time", 0.0) or 0.0),
                }
            )
    return pd.DataFrame(rows)
