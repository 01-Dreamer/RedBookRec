from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_loss(user_emb: torch.Tensor, note_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = user_emb @ note_emb.T / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def binary_recall_loss(
    user_emb: torch.Tensor,
    note_emb: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = (user_emb * note_emb).sum(dim=-1) / float(temperature)
    return F.binary_cross_entropy_with_logits(logits, labels.float(), pos_weight=pos_weight)
