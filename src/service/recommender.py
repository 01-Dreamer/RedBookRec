from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.data.io import get_device, read_json
from src.models.din_ranker import DINRanker


OBJECTIVE_WEIGHTS = {
    "click": 0.45,
    "collect": 0.25,
    "like": 0.15,
    "comment": 0.05,
    "share": 0.05,
    "page_time": 0.05,
}


def normalize_objective_weights(label_names: list[str]) -> dict[str, float]:
    weights = {name: OBJECTIVE_WEIGHTS.get(name, 0.0) for name in label_names}
    total = sum(weights.values())
    if total <= 0:
        return {name: 1.0 / len(label_names) for name in label_names}
    return {name: weight / total for name, weight in weights.items()}


def load_ranker(config: dict[str, Any]) -> tuple[DINRanker, dict] | tuple[None, None]:
    checkpoint = Path(config["paths"]["models_dir"]) / "ranker" / "din_ranker.pt"
    if not checkpoint.exists():
        return None, None
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = DINRanker(
        ckpt["num_users"],
        ckpt["num_items"],
        ckpt["num_outputs"],
        ckpt["embedding_dim"],
        ckpt["hidden_dims"],
        ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


@torch.no_grad()
def score_with_ranker(samples: pd.DataFrame, config: dict[str, Any], batch_size: int = 1024) -> pd.DataFrame:
    model, ckpt = load_ranker(config)
    out = samples.copy()
    if model is None:
        out["final_score"] = out.get("merged_recall_score", out.get("recall_score", 0.0))
        return out
    device = get_device(config)
    model = model.to(device)
    model.eval()
    label_names = ckpt["label_names"]
    weights = normalize_objective_weights(label_names)
    preds = []
    for start in range(0, len(out), batch_size):
        batch = out.iloc[start : start + batch_size]
        logits = model(
            torch.tensor(batch["user_idx_internal"].astype(int).to_numpy(), dtype=torch.long, device=device),
            torch.tensor(batch["note_idx_internal"].astype(int).to_numpy(), dtype=torch.long, device=device),
            torch.tensor(np.stack(batch["history_idx_internal"].to_numpy()), dtype=torch.long, device=device),
            torch.tensor(batch.get("position", pd.Series(0, index=batch.index)).fillna(0).to_numpy()[:, None], dtype=torch.float32, device=device),
            torch.tensor(batch.get("merged_recall_score", pd.Series(0, index=batch.index)).fillna(0).to_numpy()[:, None], dtype=torch.float32, device=device),
        )
        preds.append(torch.sigmoid(logits).cpu().numpy())
    pred = np.vstack(preds) if preds else np.zeros((0, len(label_names)))
    for idx, name in enumerate(label_names):
        out[f"p_{name}"] = pred[:, idx]
    out["final_score"] = sum(out[f"p_{name}"] * weights[name] for name in label_names)
    return out


def build_candidate_samples(candidates: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    processed = Path(config["paths"]["processed_dir"])
    mappings = read_json(processed / "mappings" / "id_mappings.json")
    profiles = pd.read_parquet(processed / "features" / "user_profiles.parquet")
    train = pd.read_parquet(processed / "samples" / "train_interactions.parquet")
    user_enc = train[["user_id", "user_idx_internal", "history_idx_internal"]].drop_duplicates("user_id")
    profiles = profiles.merge(user_enc, on="user_id", how="left")
    samples = candidates.merge(profiles[["user_id", "history_note_ids", "user_idx_internal", "history_idx_internal"]], on="user_id", how="left")
    samples["note_idx_internal"] = samples["note_id"].astype(str).map(mappings["note2idx"]).fillna(0).astype(int)
    samples["position"] = samples.get("position", pd.Series(0, index=samples.index)).fillna(0)
    samples["merged_recall_score"] = samples.get("merged_recall_score", samples.get("recall_score", 0.0)).fillna(0.0)
    return samples.dropna(subset=["user_idx_internal", "history_idx_internal"]).reset_index(drop=True)
