from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from redbookrec.data.id_mapping import map_raw
from redbookrec.recall.dataset import RecallPositiveDataset, build_user_map
from redbookrec.recall.loss import info_nce_loss
from redbookrec.recall.model import DualTowerRecall
from redbookrec.utils.config import get_device
from redbookrec.utils.io import read_json


def train_recall_model(cfg: dict, smoke_test: bool = False) -> dict:
    device = torch.device(get_device(cfg["train"].get("device", "auto")))
    samples = pd.read_parquet(cfg["data"]["train_samples_path"])
    max_train = cfg["train"].get("max_train_samples")
    if smoke_test or cfg["train"].get("smoke_test", False):
        max_train = min(int(max_train or 20000), 20000)
    if max_train:
        samples = samples.head(int(max_train))
    note_map = read_json(cfg["data"]["note_id_map_path"])["raw_to_model"]
    user_map = build_user_map(samples["user_idx"].astype(int).tolist())
    samples = samples[samples["pos_note_idx"].map(lambda x: map_raw(note_map, x) > 0)].copy()
    if samples.empty:
        raise RuntimeError("no positive recall samples can be mapped to note ids")

    dataset = RecallPositiveDataset(samples, note_map, user_map, cfg["model"]["max_history_len"])
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=len(dataset) >= int(cfg["train"]["batch_size"]),
    )
    num_users = max(user_map.values(), default=0) + 1
    num_notes = max(note_map.values(), default=0) + 1
    model = DualTowerRecall(
        num_users=num_users,
        num_notes=num_notes,
        embed_dim=int(cfg["model"]["embed_dim"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        temperature=float(cfg["model"].get("temperature", 0.05)),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    losses: list[float] = []
    model.train()
    for _ in range(int(cfg["train"].get("epochs", 1))):
        pbar = tqdm(loader, desc="train_recall", leave=False)
        for batch in pbar:
            user_id = batch["user_id"].to(device)
            note_id = batch["note_id"].to(device)
            hist = batch["history_note_ids"].to(device)
            user_emb, note_emb = model(user_id, hist, note_id)
            loss = info_nce_loss(user_emb, note_emb, model.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")

    ckpt_path = Path(cfg["paths"]["recall_checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_users": num_users,
            "num_notes": num_notes,
            "embed_dim": int(cfg["model"]["embed_dim"]),
            "temperature": float(cfg["model"].get("temperature", 0.05)),
            "dropout": float(cfg["model"].get("dropout", 0.1)),
            "user_map": user_map,
            "note_map": note_map,
            "max_history_len": int(cfg["model"]["max_history_len"]),
        },
        ckpt_path,
    )
    return {"checkpoint": str(ckpt_path), "train_samples": int(len(samples)), "loss": losses[-1] if losses else None}
