from __future__ import annotations

import hashlib
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from redbookrec.data.id_mapping import map_raw, map_sequence

USER_DENSE_COLS = ["fans_num", "follows_num"] + [f"dense_feat{i}" for i in range(1, 41)]
USER_CAT_COLS = ["gender", "platform", "age", "location"]
NOTE_DENSE_COLS = ["content_length", "commercial_flag"]
USER_CAT_BUCKETS = 128
TAX_BUCKETS = 4096
TEXT_BUCKETS = 50000
TEXT_MAX_LEN = 64
_TOKEN_RE = re.compile(r"[\w]+|[\u4e00-\u9fff]")


def stable_bucket(value: Any, buckets: int) -> int:
    if value is None or str(value) in {"", "nan", "None", "UNK"}:
        return 0
    digest = hashlib.md5(str(value).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, buckets - 1) + 1


def text_to_ids(text: Any, max_len: int = TEXT_MAX_LEN, buckets: int = TEXT_BUCKETS) -> list[int]:
    raw = "" if text is None else str(text).lower()
    tokens = _TOKEN_RE.findall(raw)
    if not tokens and raw:
        tokens = list(raw)
    ids = [stable_bucket(token, buckets) for token in tokens[: int(max_len)]]
    if len(ids) < int(max_len):
        ids.extend([0] * (int(max_len) - len(ids)))
    return ids


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
        ]
        out[raw] = {
            "note_type": int(max(0, min(15, int(getattr(row, "note_type"))))),
            "tax": [
                stable_bucket(getattr(row, "taxonomy1_id"), TAX_BUCKETS),
                stable_bucket(getattr(row, "taxonomy2_id"), TAX_BUCKETS),
                stable_bucket(getattr(row, "taxonomy3_id"), TAX_BUCKETS),
            ],
            "dense": dense,
            "text_ids": text_to_ids(getattr(row, "note_text", "")),
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
        query_text_embeddings: dict[int, np.ndarray] | None = None,
        note_text_pos_map: dict[int, int] | None = None,
        text_emb_dim: int = 768,
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.note_map = note_map
        self.user_map = user_map
        self.note_features = note_features
        self.user_features = user_features
        self.max_history_len = int(max_history_len)
        self.query_text_embeddings = query_text_embeddings or {}
        self.note_text_pos_map = note_text_pos_map or {}
        self.text_emb_dim = int(text_emb_dim)

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
            "note_text_pos": torch.tensor(self.note_text_pos_map.get(raw_note, -1), dtype=torch.long),
            "history_note_ids": torch.tensor(history, dtype=torch.long),
            "query_text_ids": torch.tensor(text_to_ids(row.get("query", "")), dtype=torch.long),
            "query_text_emb": torch.tensor(
                self.query_text_embeddings.get(int(row.get("request_idx", -1)), np.zeros(self.text_emb_dim, dtype="float32")),
                dtype=torch.float32,
            ),
            "user_cat": torch.tensor(user_feat.get("cat", [0] * len(USER_CAT_COLS)), dtype=torch.long),
            "user_dense": torch.tensor(user_feat.get("dense", [0.0] * len(USER_DENSE_COLS)), dtype=torch.float32),
            "note_type": torch.tensor(int(note_feat.get("note_type", 0)), dtype=torch.long),
            "note_tax": torch.tensor(note_feat.get("tax", [0, 0, 0]), dtype=torch.long),
            "note_dense": torch.tensor(note_feat.get("dense", [0.0] * len(NOTE_DENSE_COLS)), dtype=torch.float32),
            "note_text_ids": torch.tensor(note_feat.get("text_ids", [0] * TEXT_MAX_LEN), dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float32),
        }


RecallPositiveDataset = RecallDataset


class QilinRecallDataset(RecallDataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        note_map: dict[str, int],
        user_map: dict[str, int],
        note_features: dict[str, dict[str, Any]],
        user_features: dict[str, dict[str, Any]],
        max_history_len: int,
        negative_pools: dict[int, list[int]],
        global_negative_pool: list[int],
        negative_samples: int,
        query_text_embeddings: dict[int, np.ndarray] | None = None,
        note_text_pos_map: dict[int, int] | None = None,
        text_emb_dim: int = 768,
    ) -> None:
        super().__init__(
            samples,
            note_map,
            user_map,
            note_features,
            user_features,
            max_history_len,
            query_text_embeddings=query_text_embeddings,
            note_text_pos_map=note_text_pos_map,
            text_emb_dim=text_emb_dim,
        )
        self.negative_pools = negative_pools
        self.global_negative_pool = global_negative_pool
        self.negative_samples = int(negative_samples)

    def _sample_negatives(self, row: pd.Series, positive_note: int) -> list[int]:
        if self.negative_samples <= 0:
            return []
        request_idx = int(row.get("request_idx", -1))
        pool = [x for x in self.negative_pools.get(request_idx, []) if int(x) != int(positive_note)]
        if len(pool) >= self.negative_samples:
            return random.sample(pool, self.negative_samples)
        negatives = list(pool)
        selected = set(int(x) for x in negatives)
        attempts = 0
        while len(negatives) < self.negative_samples and self.global_negative_pool and attempts < self.negative_samples * 10:
            candidate = int(random.choice(self.global_negative_pool))
            attempts += 1
            if candidate == int(positive_note) or candidate in selected:
                continue
            negatives.append(candidate)
            selected.add(candidate)
        while len(negatives) < self.negative_samples:
            negatives.append(positive_note)
        return negatives[: self.negative_samples]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples.iloc[idx]
        item = super().__getitem__(idx)
        positive_note = int(row["note_idx"] if "note_idx" in row else row["pos_note_idx"])
        raw_notes = [positive_note] + self._sample_negatives(row, positive_note)
        note_ids: list[int] = []
        note_text_pos: list[int] = []
        note_types: list[int] = []
        note_tax: list[list[int]] = []
        note_dense: list[list[float]] = []
        note_text_ids: list[list[int]] = []
        for raw_note in raw_notes:
            feat = self.note_features.get(str(int(raw_note)), {})
            note_ids.append(map_raw(self.note_map, raw_note))
            note_text_pos.append(self.note_text_pos_map.get(int(raw_note), -1))
            note_types.append(int(feat.get("note_type", 0)))
            note_tax.append(feat.get("tax", [0, 0, 0]))
            note_dense.append(feat.get("dense", [0.0] * len(NOTE_DENSE_COLS)))
            note_text_ids.append(feat.get("text_ids", [0] * TEXT_MAX_LEN))
        item["note_id"] = torch.tensor(note_ids, dtype=torch.long)
        item["note_text_pos"] = torch.tensor(note_text_pos, dtype=torch.long)
        item["note_type"] = torch.tensor(note_types, dtype=torch.long)
        item["note_tax"] = torch.tensor(note_tax, dtype=torch.long)
        item["note_dense"] = torch.tensor(note_dense, dtype=torch.float32)
        item["note_text_ids"] = torch.tensor(note_text_ids, dtype=torch.long)
        return item
