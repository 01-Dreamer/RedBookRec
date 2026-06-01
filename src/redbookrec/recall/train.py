from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from redbookrec.data.id_mapping import map_raw
from redbookrec.data.load_qilin import read_dataset_split
from redbookrec.recall.dataset import (
    NOTE_DENSE_COLS,
    USER_DENSE_COLS,
    RecallDataset,
    build_note_features,
    build_user_features,
    build_user_map,
)
from redbookrec.recall.loss import binary_recall_loss
from redbookrec.recall.model import DualTowerRecall
from redbookrec.utils.config import get_device
from redbookrec.utils.io import read_json


def _save_recall_checkpoint(
    path: Path,
    model: DualTowerRecall,
    cfg: dict,
    num_users: int,
    num_notes: int,
    user_map: dict[str, int],
    note_map: dict[str, int],
    user_features: dict,
    epoch: int,
    epoch_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            "user_features": user_features,
            "max_history_len": int(cfg["model"]["max_history_len"]),
            "user_dense_dim": len(USER_DENSE_COLS),
            "note_dense_dim": len(NOTE_DENSE_COLS),
            "loss_type": "binary_exposure",
            "best_epoch": int(epoch),
            "best_epoch_loss": float(epoch_loss),
        },
        path,
    )


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
    note_col = "note_idx" if "note_idx" in samples else "pos_note_idx"
    samples = samples[samples[note_col].map(lambda x: map_raw(note_map, x) > 0)].copy()
    if "label" not in samples:
        samples["label"] = 1.0
    if samples.empty:
        raise RuntimeError("no recall samples can be mapped to note ids")

    needed_notes = set(int(x) for x in samples[note_col].dropna().astype("int64").tolist())
    if "recent_clicked_note_idxs" in samples:
        for hist in samples["recent_clicked_note_idxs"]:
            if isinstance(hist, list):
                needed_notes.update(int(x) for x in hist if str(x).lstrip("-").isdigit())
    note_df = pd.read_parquet(cfg["data"]["note_text_path"])
    note_df = note_df[note_df["note_idx"].astype("int64").isin(needed_notes)]
    note_features = build_note_features(note_df)
    try:
        user_df = read_dataset_split(cfg["data"]["dataset_dir"], "user_feat")
    except Exception:
        user_df = pd.DataFrame()
    user_features = build_user_features(user_df)

    dataset = RecallDataset(samples, note_map, user_map, note_features, user_features, cfg["model"]["max_history_len"])
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
        user_dense_dim=len(USER_DENSE_COLS),
        note_dense_dim=len(NOTE_DENSE_COLS),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    pos = float(samples["label"].sum())
    neg = float(len(samples) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device) if neg > 0 else None
    losses: list[float] = []
    model.train()
    epoch_losses: list[float] = []
    best_epoch = 0
    best_loss = float("inf")
    ckpt_path = Path(cfg["paths"]["recall_checkpoint"])
    for epoch in range(int(cfg["train"].get("epochs", 1))):
        pbar = tqdm(loader, desc=f"train_recall epoch={epoch + 1}", leave=False)
        running = 0.0
        seen = 0
        for batch in pbar:
            user_id = batch["user_id"].to(device)
            note_id = batch["note_id"].to(device)
            hist = batch["history_note_ids"].to(device)
            user_cat = batch["user_cat"].to(device)
            user_dense = batch["user_dense"].to(device)
            note_type = batch["note_type"].to(device)
            note_tax = batch["note_tax"].to(device)
            note_dense = batch["note_dense"].to(device)
            labels = batch["label"].to(device)
            user_emb, note_emb = model(user_id, hist, note_id, user_cat, user_dense, note_type, note_tax, note_dense)
            loss = binary_recall_loss(user_emb, note_emb, labels, model.temperature, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            running += float(loss.detach().cpu()) * int(labels.numel())
            seen += int(labels.numel())
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
        epoch_losses.append(running / max(1, seen))
        epoch_loss = epoch_losses[-1]
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            _save_recall_checkpoint(ckpt_path, model, cfg, num_users, num_notes, user_map, note_map, user_features, best_epoch, best_loss)
            print(f"epoch={epoch + 1} avg_loss={epoch_loss:.6f} saved_best={ckpt_path}")
        else:
            print(f"epoch={epoch + 1} avg_loss={epoch_loss:.6f} best_epoch={best_epoch} best_loss={best_loss:.6f}")
    return {
        "checkpoint": str(ckpt_path),
        "train_samples": int(len(samples)),
        "positive_samples": int(pos),
        "negative_samples": int(neg),
        "loss": losses[-1] if losses else None,
        "best_epoch": best_epoch,
        "best_epoch_loss": best_loss if best_loss < float("inf") else None,
        "epoch_losses": epoch_losses,
    }
