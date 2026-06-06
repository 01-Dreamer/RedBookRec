from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from redbookrec.data.id_mapping import map_raw
from redbookrec.data.load_qilin import read_dataset_split
from redbookrec.recall.dataset import (
    NOTE_DENSE_COLS,
    USER_DENSE_COLS,
    QilinRecallDataset,
    TEXT_BUCKETS,
    TEXT_MAX_LEN,
    build_note_features,
    build_user_features,
    build_user_map,
)
from redbookrec.recall.loss import grouped_info_nce_loss
from redbookrec.recall.text_embedding import encode_texts, load_text_embeddings
from redbookrec.recall.two_tower import DualTowerRecall
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
            "text_vocab_size": int(cfg["model"].get("text_vocab_size", TEXT_BUCKETS)),
            "text_max_len": int(cfg["model"].get("text_max_len", TEXT_MAX_LEN)),
            "text_emb_dim": int(model.text_emb_dim),
            "use_text_emb": bool(cfg["model"].get("use_text_emb", False)),
            "loss_type": "qilin_grouped_info_nce",
            "negative_samples": int(cfg["train"].get("negative_samples", 18)),
            "best_epoch": int(epoch),
            "best_epoch_loss": float(epoch_loss),
        },
        path,
    )


def train_recall_model(cfg: dict, smoke_test: bool = False) -> dict:
    device = torch.device(get_device(cfg["train"].get("device", "auto")))
    use_text_emb = bool(cfg["model"].get("use_text_emb", False))
    all_samples = pd.read_parquet(cfg["data"]["train_samples_path"])
    if "label" not in all_samples:
        all_samples["label"] = 1.0
    if "query" not in all_samples:
        all_samples["query"] = ""
    all_samples["label"] = pd.to_numeric(all_samples["label"], errors="coerce").fillna(0.0)
    samples = all_samples[all_samples["label"] > 0].copy()
    max_train = cfg["train"].get("max_train_samples")
    if smoke_test or cfg["train"].get("smoke_test", False):
        max_train = min(int(max_train or 20000), 20000)
    if max_train:
        samples = samples.head(int(max_train))
    note_map = read_json(cfg["data"]["note_id_map_path"])["raw_to_model"]
    all_samples = all_samples[all_samples["note_idx"].map(lambda x: map_raw(note_map, x) > 0)].copy()
    user_map = build_user_map(samples["user_idx"].astype(int).tolist())
    note_col = "note_idx" if "note_idx" in samples else "pos_note_idx"
    samples = samples[samples[note_col].map(lambda x: map_raw(note_map, x) > 0)].copy()
    if samples.empty:
        raise RuntimeError("no positive recall samples can be mapped to note ids")

    neg_samples = all_samples[all_samples["label"] <= 0].copy()
    negative_pools = {
        int(request_idx): group["note_idx"].dropna().astype("int64").tolist()
        for request_idx, group in neg_samples.groupby("request_idx")
    }
    global_negative_pool = neg_samples["note_idx"].dropna().astype("int64").drop_duplicates().tolist()

    needed_notes = set(int(x) for x in all_samples["note_idx"].dropna().astype("int64").tolist())
    if "recent_clicked_note_idxs" in samples:
        for hist in samples["recent_clicked_note_idxs"]:
            if isinstance(hist, list):
                needed_notes.update(int(x) for x in hist if str(x).lstrip("-").isdigit())
    all_note_df = pd.read_parquet(cfg["data"]["note_text_path"])
    note_text_emb = None
    note_text_emb_tensor = None
    note_text_pos_map: dict[int, int] = {}
    query_text_embeddings: dict[int, np.ndarray] = {}
    text_emb_dim = int(cfg["model"].get("text_emb_dim", 768))
    if use_text_emb:
        note_text_emb = load_text_embeddings(cfg["data"]["note_text_emb_path"])
        if note_text_emb is None:
            raise RuntimeError("note text embeddings not found. Run: python scripts/prepare_text_embeddings.py --config configs/recall.yaml")
        if len(note_text_emb) != len(all_note_df):
            raise RuntimeError(
                f"note_text_emb rows={len(note_text_emb)} does not match note_text rows={len(all_note_df)}. "
                "Regenerate note embeddings with scripts/prepare_text_embeddings.py."
            )
        text_emb_dim = int(note_text_emb.shape[1])
        note_text_emb_tensor = torch.from_numpy(note_text_emb.astype("float32", copy=False))
        note_text_pos_map = {int(raw): pos for pos, raw in enumerate(all_note_df["note_idx"].astype("int64").tolist())}
        query_rows = samples[["request_idx", "query"]].drop_duplicates("request_idx")
        query_emb = encode_texts(
            query_rows["query"].fillna("").astype(str).tolist(),
            model_name_or_path=cfg["model"].get("text_encoder_name_or_path", "../model/bert-base-chinese/"),
            batch_size=int(cfg["model"].get("text_encoder_batch_size", 64)),
            max_length=int(cfg["model"].get("text_encoder_max_length", 256)),
            device=cfg["train"].get("device", "auto"),
        )
        query_text_embeddings = {
            int(request_idx): query_emb[i]
            for i, request_idx in enumerate(query_rows["request_idx"].astype("int64").tolist())
        }
    note_df = all_note_df[all_note_df["note_idx"].astype("int64").isin(needed_notes)]
    note_features = build_note_features(note_df)
    try:
        user_df = read_dataset_split(cfg["data"]["dataset_dir"], "user_feat")
    except Exception:
        user_df = pd.DataFrame()
    user_features = build_user_features(user_df)

    negative_samples = int(cfg["train"].get("negative_samples", 18))
    dataset = QilinRecallDataset(
        samples,
        note_map,
        user_map,
        note_features,
        user_features,
        cfg["model"]["max_history_len"],
        negative_pools=negative_pools,
        global_negative_pool=global_negative_pool,
        negative_samples=negative_samples,
        query_text_embeddings=query_text_embeddings,
        note_text_pos_map=note_text_pos_map,
        text_emb_dim=text_emb_dim,
    )
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
        text_vocab_size=int(cfg["model"].get("text_vocab_size", TEXT_BUCKETS)),
        text_emb_dim=text_emb_dim,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    pos = float(samples["label"].sum())
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
            query_text_ids = batch["query_text_ids"].to(device)
            query_text_emb = batch["query_text_emb"].to(device) if use_text_emb else None
            note_id = note_id.reshape(-1)
            note_text_emb_batch = None
            if use_text_emb and note_text_emb_tensor is not None:
                note_pos = batch["note_text_pos"].reshape(-1)
                note_text_emb_batch = torch.zeros(note_pos.size(0), text_emb_dim, dtype=torch.float32)
                valid = note_pos >= 0
                if bool(valid.any()):
                    note_text_emb_batch[valid] = note_text_emb_tensor[note_pos[valid]]
                note_text_emb_batch = note_text_emb_batch.to(device)
            note_type = batch["note_type"].to(device).reshape(-1)
            note_tax = batch["note_tax"].to(device).reshape(-1, 3)
            note_dense = batch["note_dense"].to(device).reshape(-1, len(NOTE_DENSE_COLS))
            note_text_ids = batch["note_text_ids"].to(device).reshape(-1, int(cfg["model"].get("text_max_len", TEXT_MAX_LEN)))
            user_emb = model.encode_user(user_id, hist, user_cat, user_dense, query_text_ids, query_text_emb)
            note_emb = model.encode_note(note_id, note_type, note_tax[:, 0], note_tax[:, 1], note_tax[:, 2], note_dense, note_text_ids, note_text_emb_batch)
            if user_emb.size(0) < 2:
                continue
            passages_per_query = negative_samples + 1
            loss = grouped_info_nce_loss(user_emb, note_emb, passages_per_query, model.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            running += float(loss.detach().cpu()) * int(user_emb.size(0))
            seen += int(user_emb.size(0))
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
        "negative_samples": int(len(neg_samples)),
        "loss_type": "qilin_grouped_info_nce",
        "sampled_negatives_per_positive": negative_samples,
        "in_batch_negatives_per_sample": max(0, int(cfg["train"]["batch_size"]) - 1),
        "loss": losses[-1] if losses else None,
        "best_epoch": best_epoch,
        "best_epoch_loss": best_loss if best_loss < float("inf") else None,
        "epoch_losses": epoch_losses,
    }
