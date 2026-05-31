from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from redbookrec.data.id_mapping import map_raw, map_sequence


class RecallPositiveDataset(Dataset):
    def __init__(
        self,
        samples: pd.DataFrame,
        note_map: dict[str, int],
        user_map: dict[str, int],
        max_history_len: int,
    ) -> None:
        self.samples = samples.reset_index(drop=True)
        self.note_map = note_map
        self.user_map = user_map
        self.max_history_len = int(max_history_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples.iloc[idx]
        user_id = map_raw(self.user_map, row["user_idx"])
        note_id = map_raw(self.note_map, row["pos_note_idx"])
        history = map_sequence(self.note_map, row.get("recent_clicked_note_idxs", []), self.max_history_len)
        if len(history) < self.max_history_len:
            history = [0] * (self.max_history_len - len(history)) + history
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "note_id": torch.tensor(note_id, dtype=torch.long),
            "history_note_ids": torch.tensor(history, dtype=torch.long),
        }


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
