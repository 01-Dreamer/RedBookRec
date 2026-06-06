from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from redbookrec.data.id_mapping import map_raw, map_sequence
from redbookrec.data.preprocess_rec import expand_recommendation_requests
from redbookrec.recall.dataset import NOTE_DENSE_COLS, USER_CAT_COLS, USER_DENSE_COLS

RANK_DENSE_COLS = ["dcn_score", "dcn_rank_feature", "position_feature"]


def _rank_bonus(values: pd.Series) -> pd.Series:
    rank = pd.to_numeric(values, errors="coerce").fillna(1.0).clip(lower=1.0)
    return 1.0 / np.log2(rank + 2.0)


def build_rank_train_frame(rec_df: pd.DataFrame, max_samples: int | None = None) -> pd.DataFrame:
    df = expand_recommendation_requests(rec_df)
    if df.empty:
        return df
    df["dcn_score"] = 0.0
    df["dcn_rank_feature"] = 0.0
    df["position_feature"] = _rank_bonus(df["position"])
    if max_samples:
        df = df.head(int(max_samples))
    return df


def build_rank_infer_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dcn_score" not in out:
        out["dcn_score"] = 0.0
    out["dcn_score"] = pd.to_numeric(out["dcn_score"], errors="coerce").fillna(0.0)
    if "dcn_rank" in out:
        out["dcn_rank_feature"] = _rank_bonus(out["dcn_rank"])
    else:
        out["dcn_rank_feature"] = 0.0
    out["position_feature"] = out["dcn_rank_feature"]
    if "recent_clicked_note_idxs" not in out:
        out["recent_clicked_note_idxs"] = [[] for _ in range(len(out))]
    if "label_click" not in out:
        out["label_click"] = 0
    return out


class RankDataset(Dataset):
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples.iloc[idx]
        raw_user = int(row["user_idx"])
        raw_note = int(row["note_idx"])
        user_feat = self.user_features.get(str(raw_user), {})
        note_feat = self.note_features.get(str(raw_note), {})
        history = map_sequence(self.note_map, row.get("recent_clicked_note_idxs", []), self.max_history_len)
        if len(history) < self.max_history_len:
            history = [0] * (self.max_history_len - len(history)) + history

        dense = (
            list(user_feat.get("dense", [0.0] * len(USER_DENSE_COLS)))
            + list(note_feat.get("dense", [0.0] * len(NOTE_DENSE_COLS)))
            + [float(row.get(col, 0.0) or 0.0) for col in RANK_DENSE_COLS]
        )
        return {
            "user_id": torch.tensor(map_raw(self.user_map, raw_user), dtype=torch.long),
            "note_id": torch.tensor(map_raw(self.note_map, raw_note), dtype=torch.long),
            "history_note_ids": torch.tensor(history, dtype=torch.long),
            "user_cat": torch.tensor(user_feat.get("cat", [0] * len(USER_CAT_COLS)), dtype=torch.long),
            "note_type": torch.tensor(int(note_feat.get("note_type", 0)), dtype=torch.long),
            "note_tax": torch.tensor(note_feat.get("tax", [0, 0, 0]), dtype=torch.long),
            "dense": torch.tensor(dense, dtype=torch.float32),
            "label": torch.tensor(float(row.get("label_click", 0.0)), dtype=torch.float32),
        }


def dense_dim() -> int:
    return len(USER_DENSE_COLS) + len(NOTE_DENSE_COLS) + len(RANK_DENSE_COLS)


def read_rank_candidates(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
