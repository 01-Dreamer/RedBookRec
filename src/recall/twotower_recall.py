from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.models.two_tower import TwoTowerModel


class TwoTowerRecall:
    def __init__(self, model: TwoTowerModel, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    @torch.no_grad()
    def recommend(
        self,
        users: pd.DataFrame,
        notes: pd.DataFrame,
        mappings: dict,
        top_k: int = 200,
        batch_size: int = 1024,
    ) -> pd.DataFrame:
        self.model.eval()
        note2idx = mappings["note2idx"]
        note_ids = notes["note_id"].astype(int).tolist()
        item_idx = torch.tensor([note2idx.get(str(n), 0) for n in note_ids], dtype=torch.long, device=self.device)
        item_vecs = []
        for start in range(0, len(item_idx), batch_size):
            item_vecs.append(self.model.encode_item(item_idx[start : start + batch_size]).cpu())
        item_vec = torch.cat(item_vecs, dim=0)
        item_vec = torch.nn.functional.normalize(item_vec, dim=1)

        rows = []
        for row in tqdm(users.itertuples(index=False), total=len(users), desc="twotower recall"):
            hist_idx = torch.tensor([row.history_idx_internal], dtype=torch.long, device=self.device)
            user_idx = torch.tensor([row.user_idx_internal], dtype=torch.long, device=self.device)
            user_vec = self.model.encode_user(user_idx, hist_idx).cpu()
            user_vec = torch.nn.functional.normalize(user_vec, dim=1)
            scores = torch.matmul(user_vec, item_vec.T).squeeze(0).numpy()
            seen = set(int(x) for x in row.history_note_ids)
            candidate_idx = np.argpartition(-scores, kth=min(top_k * 3, len(scores) - 1))[: top_k * 3]
            ranked = candidate_idx[np.argsort(-scores[candidate_idx])]
            count = 0
            for idx in ranked:
                note_id = int(note_ids[idx])
                if note_id in seen:
                    continue
                rows.append(
                    {
                        "user_id": int(row.user_id),
                        "note_id": note_id,
                        "recall_score": float(scores[idx]),
                        "recall_source": "twotower",
                    }
                )
                count += 1
                if count >= top_k:
                    break
        return pd.DataFrame(rows)
