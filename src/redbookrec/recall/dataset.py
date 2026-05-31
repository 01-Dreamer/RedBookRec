from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from redbookrec.data.id_mapping import map_raw, map_sequence

USER_DENSE_COLS = ["fans_num", "follows_num"] + [f"dense_feat{i}" for i in range(1, 41)]
USER_CAT_COLS = ["gender", "platform", "age", "location"]
NOTE_DENSE_COLS = ["content_length", "commercial_flag", "image_num", "video_duration"]
USER_CAT_BUCKETS = 128
TAX_BUCKETS = 4096


def stable_bucket(value: Any, buckets: int) -> int:
    if value is None or str(value) in {"", "nan", "None", "UNK"}:
        return 0
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, buckets - 1) + 1


def build_user_map(values: list[int]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    next_id = 1
    for value in values:
        try:
            key = str(int(value))
        except Exception:
            continue
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
    return mapping


def build_user_features(user_df: pd.DataFrame | None) -> dict[str, dict[str, list[float] | list[int]]]:
    if user_df is None or user_df.empty:
        return {}
    df = user_df.copy()
    for col in USER_DENSE_COLS:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in USER_CAT_COLS:
        if col not in df:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    out: dict[str, dict[str, list[float] | list[int]]] = {}
    for row in df.itertuples(index=False):
        raw = str(int(getattr(row, "user_idx")))
        dense = [float(np.log1p(max(0.0, float(getattr(row, col))))) for col in USER_DENSE_COLS]
        cats = [stable_bucket(getattr(row, col), USER_CAT_BUCKETS) for col in USER_CAT_COLS]
        out[raw] = {"dense": dense, "cat": cats}
    return out


def build_note_features(note_df: pd.DataFrame | None) -> dict[str, dict[str, list[float] | list[int] | int]]:
    if note_df is None or note_df.empty:
        return {}
    df = note_df.copy()
    for col in NOTE_DENSE_COLS:
        if col not in df:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]:
        if col not in df:
            df[col] = "UNK"
        df[col] = df[col].fillna("UNK").astype(str)
    df["note_type"] = pd.to_numeric(df.get("note_type", 0), errors="coerce").fillna(0).astype("int64")
    out: dict[str, dict[str, list[float] | list[int] | int]] = {}
    for row in df.itertuples(index=False):
        raw = str(int(getattr(row, "note_idx")))
        dense = [
            float(np.log1p(max(0.0, float(getattr(row, "content_length")))) / 10.0),
            float(getattr(row, "commercial_flag")),
            float(np.log1p(max(0.0, float(getattr(row, "image_num")))) / 5.0),
            float(np.log1p(max(0.0, float(getattr(row, "video_duration")))) / 10.0),
        ]
        out[raw] = {
            "note_type": int(max(0, min(15, int(getattr(row, "note_type"))))),
            "tax": [
                stable_bucket(getattr(row, "taxonomy1_id"), TAX_BUCKETS),
                stable_bucket(getattr(row, "taxonomy2_id"), TAX_BUCKETS),
                stable_bucket(getattr(row, "taxonomy3_id"), TAX_BUCKETS),
            ],
            "dense": dense,
        }
    return out


class RecallDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        note_map: dict[str, int],
        user_map: dict[str, int],
        note_features: dict[str, dict[str, Any]],
        user_features: dict[str, dict[str, Any]],
        max_history_len: int,
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.note_map = note_map
        self.user_map = user_map
        self.note_features = note_features
        self.user_features = user_features
        self.max_history_len = int(max_history_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples.iloc[idx]
        raw_note = int(row["note_idx"] if "note_idx" in row else row["pos_note_idx"])
        raw_user = int(row["user_idx"])
        user_id = map_raw(self.user_map, raw_user)
        note_id = map_raw(self.note_map, raw_note)
        history = map_sequence(self.note_map, row.get("recent_clicked_note_idxs", []), self.max_history_len)
        if len(history) < self.max_history_len:
            history = [0] * (self.max_history_len - len(history)) + history

        user_feat = self.user_features.get(str(raw_user), {})
        note_feat = self.note_features.get(str(raw_note), {})
        label = float(row.get("label", row.get("label_click", 1.0)))
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "note_id": torch.tensor(note_id, dtype=torch.long),
            "history_note_ids": torch.tensor(history, dtype=torch.long),
            "user_cat": torch.tensor(user_feat.get("cat", [0] * len(USER_CAT_COLS)), dtype=torch.long),
            "user_dense": torch.tensor(user_feat.get("dense", [0.0] * len(USER_DENSE_COLS)), dtype=torch.float32),
            "note_type": torch.tensor(int(note_feat.get("note_type", 0)), dtype=torch.long),
            "note_tax": torch.tensor(note_feat.get("tax", [0, 0, 0]), dtype=torch.long),
            "note_dense": torch.tensor(note_feat.get("dense", [0.0] * len(NOTE_DENSE_COLS)), dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }


RecallPositiveDataset = RecallDataset
