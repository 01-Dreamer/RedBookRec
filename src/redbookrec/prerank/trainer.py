from __future__ import annotations

from pathlib import Path

import torch


def train_prerank_placeholder(cfg: dict) -> dict:
    path = Path(cfg["paths"]["checkpoint"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "dcn_lite_placeholder", "config": cfg}, path)
    return {"checkpoint": str(path)}
