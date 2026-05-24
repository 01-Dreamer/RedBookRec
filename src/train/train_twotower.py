from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import TwoTowerTrainDataset
from src.data.io import get_device, read_json, seed_everything, write_json
from src.models.two_tower import TwoTowerModel, inbatch_softmax_loss


def train_twotower(config: dict[str, Any]) -> Path:
    seed_everything(int(config["seed"]))
    processed = Path(config["paths"]["processed_dir"])
    model_dir = Path(config["paths"]["models_dir"]) / "twotower"
    model_dir.mkdir(parents=True, exist_ok=True)
    mappings = read_json(processed / "mappings" / "id_mappings.json")
    train = pd.read_parquet(processed / "samples" / "train_interactions.parquet")

    dataset = TwoTowerTrainDataset(train, max_samples=int(config.get("max_train_samples", 120000)))
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=0, drop_last=len(dataset) > 1)
    device = get_device(config)
    model = TwoTowerModel(
        num_users=len(mappings["user2idx"]),
        num_items=len(mappings["note2idx"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))

    history = []
    for epoch in range(int(config["epochs"])):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"twotower epoch {epoch + 1}"):
            optimizer.zero_grad()
            logits = model(
                batch["user_idx"].to(device),
                batch["history_idx"].to(device),
                batch["item_idx"].to(device),
            )
            loss = inbatch_softmax_loss(logits)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch["user_idx"])
        epoch_loss = total_loss / max(len(dataset), 1)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})
        print(f"epoch={epoch + 1} loss={epoch_loss:.4f}")

    checkpoint = model_dir / "twotower.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "num_users": len(mappings["user2idx"]),
            "num_items": len(mappings["note2idx"]),
            "embedding_dim": int(config["embedding_dim"]),
            "hidden_dim": int(config["hidden_dim"]),
        },
        checkpoint,
    )
    write_json({"train_history": history}, model_dir / "train_metrics.json")
    return checkpoint
