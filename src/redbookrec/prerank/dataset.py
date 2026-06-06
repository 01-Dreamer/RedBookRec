from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from redbookrec.data.id_mapping import map_raw
from redbookrec.data.preprocess_rec import expand_recommendation_requests
from redbookrec.recall.dataset import (
    NOTE_DENSE_COLS,
    TAX_BUCKETS,
    USER_CAT_COLS,
    USER_DENSE_COLS,
    build_note_features,
    build_user_features,
    build_user_map,
)

RECALL_DENSE_COLS = ["dual_score", "search_score", "hybrid_score", "rank_feature", "position_feature"]


def build_prerank_train_frame(rec_df: pd.DataFrame, max_samples: int | None = None) -> pd.DataFrame:
    df = expand_recommendation_requests(rec_df)
    if df.empty:
        return df
    df["dual_score"] = 0.0
    df["search_score"] = 0.0
    df["hybrid_score"] = 0.0
    df["rank_feature"] = 0.0
    df["position_feature"] = 1.0 / np.log2(pd.to_numeric(df["position"], errors="coerce").fillna(1.0).clip(lower=1.0) + 2.0)
    if max_samples:
        df = df.head(int(max_samples))
    return df


def build_prerank_infer_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["dual_score", "search_score", "hybrid_score"]:
        if col not in out:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    rank = pd.to_numeric(out.get("hybrid_rank", 1.0), errors="coerce").fillna(1.0).clip(lower=1.0)
    out["rank_feature"] = 1.0 / np.log2(rank + 2.0)
    out["position_feature"] = out["rank_feature"]
    if "label_click" not in out:
        out["label_click"] = 0
    return out


class PrerankDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        note_map: dict[str, int],
        user_map: dict[str, int],
        note_features: dict[str, dict[str, Any]],
        user_features: dict[str, dict[str, Any]],
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.note_map = note_map
        self.user_map = user_map
        self.note_features = note_features
        self.user_features = user_features

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.samples.iloc[idx]
        raw_user = int(row["user_idx"])
        raw_note = int(row["note_idx"])
        user_id = map_raw(self.user_map, raw_user)
        note_id = map_raw(self.note_map, raw_note)
        user_feat = self.user_features.get(str(raw_user), {})
        note_feat = self.note_features.get(str(raw_note), {})
        user_dense = user_feat.get("dense", [0.0] * len(USER_DENSE_COLS))
        note_dense = note_feat.get("dense", [0.0] * len(NOTE_DENSE_COLS))
        recall_dense = [float(row.get(col, 0.0) or 0.0) for col in RECALL_DENSE_COLS]
        dense = list(user_dense) + list(note_dense) + recall_dense
        label = float(row.get("label_click", 0.0))
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "note_id": torch.tensor(note_id, dtype=torch.long),
            "user_cat": torch.tensor(user_feat.get("cat", [0] * len(USER_CAT_COLS)), dtype=torch.long),
            "note_type": torch.tensor(int(note_feat.get("note_type", 0)), dtype=torch.long),
            "note_tax": torch.tensor(note_feat.get("tax", [0, 0, 0]), dtype=torch.long),
            "dense": torch.tensor(dense, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
        }


def dense_dim() -> int:
    return len(USER_DENSE_COLS) + len(NOTE_DENSE_COLS) + len(RECALL_DENSE_COLS)
