from __future__ import annotations

from pathlib import Path

import torch


def train_rank_placeholder(cfg: dict) -> dict:
    path = Path(cfg["paths"]["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "sim_placeholder", "config": cfg}, path)
    return {"checkpoint": str(path)}
