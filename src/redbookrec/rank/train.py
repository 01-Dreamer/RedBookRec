from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from redbookrec.data.id_mapping import map_raw
from redbookrec.data.load_qilin import read_dataset_split
from redbookrec.data.preprocess_rec import read_recommendation
from redbookrec.rank.dataset import RankDataset, build_rank_train_frame, dense_dim
from redbookrec.rank.loss import build_pos_weight, click_bce_loss
from redbookrec.rank.sim import SIMRanker
from redbookrec.recall.dataset import NOTE_DENSE_COLS, USER_DENSE_COLS, build_note_features, build_user_features, build_user_map
from redbookrec.utils.config import get_device
from redbookrec.utils.io import read_json


def _save_checkpoint(
    path: Path,
    model: SIMRanker,
    cfg: dict,
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
            "num_users": model.user_embedding.num_embeddings,
            "num_notes": model.note_embedding.num_embeddings,
            "dense_dim": model.dense_dim,
            "embed_dim": model.embed_dim,
            "hidden_dims": list(cfg["model"].get("hidden_dims", [256, 128, 64])),
            "dropout": float(cfg["model"].get("dropout", 0.1)),
            "max_history_len": int(cfg["model"].get("last_n", 20)),
            "user_map": user_map,
            "note_map": note_map,
            "user_features": user_features,
            "user_dense_cols": USER_DENSE_COLS,
            "note_dense_cols": NOTE_DENSE_COLS,
            "best_epoch": int(epoch),
            "best_epoch_loss": float(epoch_loss),
        },
        path,
    )


def train_rank(cfg: dict, smoke_test: bool = False) -> dict:
    device = torch.device(get_device(cfg["train"].get("device", "auto")))
    max_requests = cfg["train"].get("max_requests")
    rec_df = read_recommendation(cfg["data"]["dataset_dir"], "recommendation_train", max_requests=max_requests)
    samples = build_rank_train_frame(rec_df)
    max_train = cfg["train"].get("max_train_samples")
    if smoke_test or cfg["train"].get("smoke_test", False):
        max_train = min(int(max_train or 20000), 20000)
    if max_train:
        samples = samples.head(int(max_train))
    if samples.empty:
        raise RuntimeError("no rank samples found")

    note_map = read_json(cfg["data"].get("note_id_map_path", "data_cache/notes/note_id_map.json"))["raw_to_model"]
    samples = samples[samples["note_idx"].map(lambda x: map_raw(note_map, x) > 0)].copy()
    if samples.empty:
        raise RuntimeError("no rank samples can be mapped to note ids")
    user_map = build_user_map(samples["user_idx"].astype(int).tolist())

    note_df = pd.read_parquet(cfg["data"]["note_text_path"])
    needed_notes = set(samples["note_idx"].astype("int64").tolist())
    note_df = note_df[note_df["note_idx"].astype("int64").isin(needed_notes)]
    note_features = build_note_features(note_df)
    try:
        user_df = read_dataset_split(cfg["data"]["dataset_dir"], "user_feat")
    except Exception:
        user_df = pd.DataFrame()
    user_features = build_user_features(user_df)

    dataset = RankDataset(
        samples,
        note_map,
        user_map,
        note_features,
        user_features,
        max_history_len=int(cfg["model"].get("last_n", 20)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 128)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
    )
    model = SIMRanker(
        num_users=max(user_map.values(), default=0) + 1,
        num_notes=max(note_map.values(), default=0) + 1,
        dense_dim=dense_dim(),
        embed_dim=int(cfg["model"].get("embed_dim", 64)),
        hidden_dims=list(cfg["model"].get("hidden_dims", [256, 128, 64])),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 0.001)),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    pos = float(samples["label_click"].sum())
    neg = float(len(samples) - pos)
    pos_weight = build_pos_weight(pos, neg, device)

    ckpt_path = Path(cfg["paths"]["checkpoint"])
    best_loss = float("inf")
    best_epoch = 0
    epoch_losses: list[float] = []
    model.train()
    for epoch in range(int(cfg["train"].get("epochs", 1))):
        running = 0.0
        seen = 0
        pbar = tqdm(loader, desc=f"train_sim epoch={epoch + 1}", leave=False)
        for batch in pbar:
            logits = model(
                batch["user_id"].to(device),
                batch["note_id"].to(device),
                batch["history_note_ids"].to(device),
                batch["user_cat"].to(device),
                batch["note_type"].to(device),
                batch["note_tax"].to(device),
                batch["dense"].to(device),
            )
            label = batch["label"].to(device)
            loss = click_bce_loss(logits, label, pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu()) * int(label.numel())
            seen += int(label.numel())
            pbar.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        epoch_loss = running / max(1, seen)
        epoch_losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            _save_checkpoint(ckpt_path, model, cfg, user_map, note_map, user_features, best_epoch, best_loss)
            print(f"epoch={epoch + 1} avg_loss={epoch_loss:.6f} saved_best={ckpt_path}")
        else:
            print(f"epoch={epoch + 1} avg_loss={epoch_loss:.6f} best_epoch={best_epoch} best_loss={best_loss:.6f}")

    return {
        "checkpoint": str(ckpt_path),
        "train_samples": int(len(samples)),
        "positive_samples": int(pos),
        "negative_samples": int(neg),
        "best_epoch": best_epoch,
        "best_epoch_loss": best_loss if best_loss < float("inf") else None,
        "epoch_losses": epoch_losses,
    }
