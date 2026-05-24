from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import RankerDataset
from src.data.io import get_device, read_json, seed_everything, write_json
from src.models.din_ranker import DINRanker
from src.models.losses import multitask_loss


def train_ranker(config: dict[str, Any]) -> Path:
    seed_everything(int(config["seed"]))
    processed = Path(config["paths"]["processed_dir"])
    model_dir = Path(config["paths"]["models_dir"]) / "ranker"
    model_dir.mkdir(parents=True, exist_ok=True)
    mappings = read_json(processed / "mappings" / "id_mappings.json")
    train = pd.read_parquet(processed / "samples" / "train_interactions.parquet")
    dataset = RankerDataset(train, max_samples=int(config.get("max_train_samples", 160000)))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, drop_last=len(dataset) > 1)
    device = get_device(config)
    model = DINRanker(
        num_users=len(mappings["user2idx"]),
        num_items=len(mappings["note2idx"]),
        num_outputs=len(dataset.label_names),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dims=list(config["hidden_dims"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    history = []
    for epoch in range(int(config["epochs"])):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"ranker epoch {epoch + 1}"):
            optimizer.zero_grad()
            preds = model(
                batch["user_idx"].to(device),
                batch["item_idx"].to(device),
                batch["history_idx"].to(device),
                batch["position"].to(device),
                batch["recall_score"].to(device),
            )
            loss = multitask_loss(preds, batch["labels"].to(device), dataset.label_names)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["user_idx"])
        epoch_loss = total_loss / max(len(dataset), 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch={epoch + 1} loss={epoch_loss:.4f}")

    checkpoint = model_dir / "din_ranker.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_users": len(mappings["user2idx"]),
            "num_items": len(mappings["note2idx"]),
            "num_outputs": len(dataset.label_names),
            "label_names": dataset.label_names,
            "embedding_dim": int(config["embedding_dim"]),
            "hidden_dims": list(config["hidden_dims"]),
            "dropout": float(config["dropout"]),
        },
        checkpoint,
    )
    write_json({"train_history": history, "label_names": dataset.label_names}, model_dir / "train_metrics.json")
    return checkpoint
