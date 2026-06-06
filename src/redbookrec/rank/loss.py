from __future__ import annotations

import torch
from torch.nn import functional as F


def click_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)


def build_pos_weight(pos: float, neg: float, device: torch.device) -> torch.Tensor | None:
    if neg <= 0:
        return None
    return torch.tensor([float(neg) / max(float(pos), 1.0)], dtype=torch.float32, device=device)
