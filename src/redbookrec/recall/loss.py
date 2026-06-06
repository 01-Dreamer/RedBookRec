from __future__ import annotations

import torch
from torch.nn import functional as F


def info_nce_loss(user_emb: torch.Tensor, note_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = user_emb @ note_emb.T / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def grouped_info_nce_loss(
    user_emb: torch.Tensor,
    note_emb: torch.Tensor,
    passages_per_query: int,
    temperature: float,
) -> torch.Tensor:
    logits = user_emb @ note_emb.T / float(temperature)
    labels = torch.arange(logits.size(0), device=logits.device) * int(passages_per_query)
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


def build_pos_weight(pos: float, neg: float, device: torch.device) -> torch.Tensor | None:
    if neg <= 0:
        return None
    return torch.tensor([float(neg) / max(float(pos), 1.0)], dtype=torch.float32, device=device)
