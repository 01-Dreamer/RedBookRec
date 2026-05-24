from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.parser import LABEL_COLUMNS


class TwoTowerTrainDataset(Dataset):
    def __init__(self, interactions: pd.DataFrame, max_samples: int | None = None) -> None:
        positives = interactions[interactions["click"] > 0].copy()
        if len(positives) == 0:
            positives = interactions.copy()
        if max_samples:
            positives = positives.sample(min(max_samples, len(positives)), random_state=2025)
        self.user_idx = positives["user_idx_internal"].astype(int).to_numpy()
        self.item_idx = positives["note_idx_internal"].astype(int).to_numpy()
        self.history = np.stack(positives["history_idx_internal"].to_numpy()).astype("int64")

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(self.user_idx[idx], dtype=torch.long),
            "item_idx": torch.tensor(self.item_idx[idx], dtype=torch.long),
            "history_idx": torch.tensor(self.history[idx], dtype=torch.long),
        }


class RankerDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, max_samples: int | None = None) -> None:
        df = samples.copy()
        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=2025)
        self.user_idx = df["user_idx_internal"].astype(int).to_numpy()
        self.item_idx = df["note_idx_internal"].astype(int).to_numpy()
        self.history = np.stack(df["history_idx_internal"].to_numpy()).astype("int64")
        self.position = df.get("position", pd.Series(0, index=df.index)).fillna(0).astype(float).to_numpy()
        self.recall_score = df.get("merged_recall_score", pd.Series(0, index=df.index)).fillna(0).astype(float).to_numpy()
        self.labels = df[[c for c in LABEL_COLUMNS if c in df.columns]].astype(float)
        if "page_time_norm" in df.columns:
            self.labels["page_time"] = df["page_time_norm"].astype(float)
        self.label_names = list(self.labels.columns)
        self.labels_np = self.labels.to_numpy(dtype="float32")

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "user_idx": torch.tensor(self.user_idx[idx], dtype=torch.long),
            "item_idx": torch.tensor(self.item_idx[idx], dtype=torch.long),
            "history_idx": torch.tensor(self.history[idx], dtype=torch.long),
            "position": torch.tensor([self.position[idx]], dtype=torch.float32),
            "recall_score": torch.tensor([self.recall_score[idx]], dtype=torch.float32),
            "labels": torch.tensor(self.labels_np[idx], dtype=torch.float32),
        }
