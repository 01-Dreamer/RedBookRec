from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .features import numeric_feature_columns


class InteractionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        dense_cols: list[str],
        label_col: str = "click_label",
        label_cols: list[str] | None = None,
        history: dict[int, list[int]] | None = None,
        note_taxonomy: dict[int, tuple[str, str, str]] | None = None,
        last_n: int = 20,
        top_k: int = 20,
    ):
        self.df = df.reset_index(drop=True)
        self.dense_cols = dense_cols
        dense = self.df[dense_cols].fillna(0.0).astype("float32").values if dense_cols else np.zeros((len(df), 0), dtype="float32")
        self.dense = dense
        self.label_cols = label_cols or [label_col]
        labels = []
        for col in self.label_cols:
            if col in self.df:
                labels.append(self.df[col].fillna(0).astype("float32").values)
            else:
                labels.append(np.zeros(len(self.df), dtype="float32"))
        self.labels = np.stack(labels, axis=1).astype("float32")
        self.user_ids = self.df["user_id"].fillna(0).astype("int64").values
        self.note_ids = self.df["note_id"].fillna(0).astype("int64").values
        self.history = history or {}
        self.note_taxonomy = note_taxonomy or {}
        self.last_n = last_n
        self.top_k = top_k

    def __len__(self) -> int:
        return len(self.df)

    def _seq(self, user_id: int, n: int) -> list[int]:
        seq = self.history.get(int(user_id), [])
        seq = seq[-n:]
        if len(seq) < n:
            seq = [0] * (n - len(seq)) + seq
        return seq

    def _target_topk_seq(self, user_id: int, note_id: int, k: int) -> list[int]:
        history = self.history.get(int(user_id), [])
        target_tax = self.note_taxonomy.get(int(note_id), ("", "", ""))
        scored = []
        for pos, hist_note in enumerate(history):
            hist_tax = self.note_taxonomy.get(int(hist_note), ("", "", ""))
            sim = sum(1 for a, b in zip(target_tax, hist_tax) if a and a == b)
            recency = pos / max(1, len(history))
            scored.append((sim, recency, hist_note))
        scored.sort(reverse=True)
        seq = [int(note) for _, _, note in scored[:k]]
        if len(seq) < k:
            seq = [0] * (k - len(seq)) + seq
        return seq

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        user_id = int(self.user_ids[idx])
        note_id = int(self.note_ids[idx])
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "note_id": torch.tensor(note_id, dtype=torch.long),
            "dense": torch.tensor(self.dense[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "lastn_seq": torch.tensor(self._seq(user_id, self.last_n), dtype=torch.long),
            "topk_seq": torch.tensor(self._target_topk_seq(user_id, note_id, self.top_k), dtype=torch.long),
        }


def load_training_frame(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "training_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.read_parquet(Path(processed_dir) / "interactions.parquet")


def load_history(processed_dir: str | Path) -> dict[int, list[int]]:
    path = Path(processed_dir) / "user_history.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {int(r.user_id): list(r.history_note_ids) for r in df.itertuples(index=False)}


def load_note_taxonomy(processed_dir: str | Path) -> dict[int, tuple[str, str, str]]:
    path = Path(processed_dir) / "notes.parquet"
    if not path.exists():
        return {}
    cols = ["note_id", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
    df = pd.read_parquet(path, columns=[c for c in cols if c])
    result = {}
    for row in df.itertuples(index=False):
        result[int(row.note_id)] = (
            str(getattr(row, "taxonomy1_id", "") or ""),
            str(getattr(row, "taxonomy2_id", "") or ""),
            str(getattr(row, "taxonomy3_id", "") or ""),
        )
    return result


def prepare_dense(df: pd.DataFrame, processed_dir: str | Path, model_name: str) -> tuple[pd.DataFrame, list[str], StandardScaler]:
    cols = numeric_feature_columns(df)
    scaler = StandardScaler()
    if cols:
        df = df.copy()
        df[cols] = scaler.fit_transform(df[cols].fillna(0.0).astype("float32"))
    Path(processed_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump({"cols": cols, "scaler": scaler}, Path(processed_dir) / f"{model_name}_dense_scaler.joblib")
    return df, cols, scaler


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=max(0, int(num_workers)))


def train_binary_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> list[float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(int(epochs)):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad()
            kind = getattr(model, "model_kind", "")
            if kind == "twotower":
                logits = model(batch["user_id"], batch["note_id"], batch["lastn_seq"])
            elif kind == "dcn":
                logits = model(batch["user_id"], batch["note_id"], batch["dense"])
            elif kind == "sim":
                logits = model(batch["user_id"], batch["note_id"], batch["dense"], batch["lastn_seq"], batch["topk_seq"])
            else:
                logits = model(batch["user_id"], batch["note_id"])
            if logits.ndim == 1 and batch["label"].ndim == 2 and batch["label"].shape[1] == 1:
                labels = batch["label"].squeeze(1)
            else:
                labels = batch["label"]
            loss = loss_fn(logits, labels)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)
    return losses


def max_ids(df: pd.DataFrame) -> tuple[int, int]:
    return int(df["user_id"].max()) + 1, int(df["note_id"].max()) + 1
