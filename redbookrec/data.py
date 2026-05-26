from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import dataset_path


ENGAGEMENT_COLS = ["click", "like", "collect", "comment", "share"]


def read_parquet(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


def read_notes(cfg: dict[str, Any]) -> pd.DataFrame:
    pattern = str(Path(cfg["paths"]["dataset_dir"]) / cfg["data"]["notes_glob"])
    paths = sorted(glob.glob(pattern))
    frames = []
    max_notes = cfg.get("limits", {}).get("max_notes")
    remaining = max_notes
    for path in paths:
        df = pd.read_parquet(path)
        if remaining is not None:
            df = df.head(max(0, int(remaining)))
            remaining -= len(df)
        frames.append(df)
        if remaining is not None and remaining <= 0:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_users(cfg: dict[str, Any]) -> pd.DataFrame:
    df = read_parquet(dataset_path(cfg, "user_feat"))
    max_users = cfg.get("limits", {}).get("max_users")
    if max_users is not None:
        df = df.head(int(max_users))
    return df


def _to_py_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return value
    return list(value) if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)) else []


def expand_recommendation(cfg: dict[str, Any], split: str = "recommendation_train") -> pd.DataFrame:
    df = read_parquet(dataset_path(cfg, split))
    rows: list[dict[str, Any]] = []
    max_interactions = cfg.get("limits", {}).get("max_interactions")
    for base in df.itertuples(index=False):
        details = _to_py_list(getattr(base, "rec_result_details_with_idx", []))
        history = [int(x) for x in _to_py_list(getattr(base, "recent_clicked_note_idxs", [])) if pd.notna(x)]
        for item in details:
            if hasattr(item, "as_py"):
                item = item.as_py()
            if not isinstance(item, dict):
                try:
                    item = dict(item)
                except Exception:
                    continue
            row = {
                "raw_user_idx": int(getattr(base, "user_idx")),
                "request_idx": int(getattr(base, "request_idx", -1)),
                "session_idx": int(getattr(base, "session_idx", -1)),
                "query": getattr(base, "query", "") or "",
                "history_raw_note_idxs": history,
                "raw_note_idx": int(item.get("note_idx", -1)) if pd.notna(item.get("note_idx", -1)) else -1,
                "position": int(item.get("position", -1)) if pd.notna(item.get("position", -1)) else -1,
                "request_timestamp": float(item.get("request_timestamp", item.get("search_timestamp", 0.0)) or 0.0),
                "page_time": float(item.get("page_time", 0.0) or 0.0),
            }
            for col in ENGAGEMENT_COLS:
                row[col] = int(item.get(col, 0) or 0)
            row["click_label"] = int(row["click"] > 0)
            row["engage_label"] = int(any(row[col] > 0 for col in ENGAGEMENT_COLS))
            row["relevance"] = (
                row["click"]
                + 2 * row["like"]
                + 3 * row["collect"]
                + 2 * row["comment"]
                + row["share"]
            )
            rows.append(row)
            if max_interactions is not None and len(rows) >= int(max_interactions):
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def build_mapping(values: pd.Series | list[Any], default_id: int = 0) -> dict[int, int]:
    clean = pd.Series(values).dropna().astype("int64").unique().tolist()
    clean = sorted(int(v) for v in clean if int(v) >= 0)
    return {raw: idx + 1 for idx, raw in enumerate(clean)}


def map_with_default(values: pd.Series, mapping: dict[int, int], default_id: int = 0) -> pd.Series:
    return values.map(lambda x: mapping.get(int(x), default_id) if pd.notna(x) else default_id).astype("int64")


def map_history(history: list[int], mapping: dict[int, int], default_id: int = 0, max_len: int | None = None) -> list[int]:
    mapped = [mapping.get(int(x), default_id) for x in history if int(x) >= 0]
    if max_len is not None:
        mapped = mapped[-int(max_len) :]
    return mapped


def save_mapping(path: Path, raw_name: str, mapped_name: str, mapping: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({raw_name: list(mapping.keys()), mapped_name: list(mapping.values())})
    df.to_parquet(path, index=False)


def load_mapping(path: Path, raw_name: str, mapped_name: str) -> dict[int, int]:
    df = pd.read_parquet(path)
    return dict(zip(df[raw_name].astype(int), df[mapped_name].astype(int)))


def build_user_history(interactions: pd.DataFrame, max_history: int) -> pd.DataFrame:
    pos = interactions[interactions["engage_label"] > 0].copy()
    if pos.empty:
        return pd.DataFrame(columns=["user_id", "history_note_ids"])
    pos = pos.sort_values(["user_id", "request_timestamp", "request_idx", "position"])
    rows = []
    for user_id, group in pos.groupby("user_id"):
        history = group["note_id"].astype(int).tolist()[-int(max_history) :]
        rows.append({"user_id": int(user_id), "history_note_ids": history})
    return pd.DataFrame(rows)


def load_processed(cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    base = Path(cfg["paths"]["processed_dir"])
    data = {}
    for name in ["interactions", "notes", "users", "user_history"]:
        path = base / f"{name}.parquet"
        if path.exists():
            data[name] = pd.read_parquet(path)
    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
