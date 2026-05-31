from __future__ import annotations

import torch
from torch import nn


class CrossLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * torch.sum(x * self.weight, dim=-1, keepdim=True) + self.bias + x


class DCN(nn.Module):
    def __init__(self, num_users: int, num_notes: int, dense_dim: int = 4, embed_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(num_notes, embed_dim, padding_idx=0)
        input_dim = embed_dim * 2 + dense_dim
        self.cross = CrossLayer(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, user_id: torch.Tensor, note_id: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([self.user_embedding(user_id), self.note_embedding(note_id), dense], dim=-1)
        x1 = self.cross(x0, x0)
        return self.mlp(torch.cat([x0, x1], dim=-1)).squeeze(-1)
