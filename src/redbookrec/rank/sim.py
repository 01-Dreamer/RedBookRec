from __future__ import annotations

import torch
from torch import nn


class SIMRanker(nn.Module):
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 3 + 1, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, target: torch.Tensor, lastn_interest: torch.Tensor, topk_interest: torch.Tensor, dense_score: torch.Tensor) -> torch.Tensor:
        x = torch.cat([target, lastn_interest, topk_interest, dense_score.unsqueeze(-1)], dim=-1)
        return self.mlp(x).squeeze(-1)
