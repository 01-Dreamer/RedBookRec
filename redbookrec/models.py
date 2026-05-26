from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for dim in hidden_dims:
        layers.extend([nn.Linear(last, dim), nn.ReLU()])
        if dropout:
            layers.append(nn.Dropout(dropout))
        last = dim
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class TwoTower(nn.Module):
    def __init__(self, num_users: int, num_notes: int, embedding_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.model_kind = "twotower"
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(num_notes, embedding_dim, padding_idx=0)
        self.user_mlp = mlp(embedding_dim * 2, hidden_dims, embedding_dim)
        self.note_mlp = mlp(embedding_dim, hidden_dims, embedding_dim)

    def encode_user(self, user_id: torch.Tensor, history_seq: torch.Tensor | None = None) -> torch.Tensor:
        user = self.user_embedding(user_id)
        if history_seq is None:
            hist = torch.zeros_like(user)
        else:
            emb = self.note_embedding(history_seq)
            mask = (history_seq != 0).float().unsqueeze(-1)
            hist = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.nn.functional.normalize(self.user_mlp(torch.cat([user, hist], dim=-1)), dim=-1)

    def encode_note(self, note_id: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.note_mlp(self.note_embedding(note_id)), dim=-1)

    def forward(self, user_id: torch.Tensor, note_id: torch.Tensor, history_seq: torch.Tensor | None = None) -> torch.Tensor:
        return (self.encode_user(user_id, history_seq) * self.encode_note(note_id)).sum(dim=-1)


class DCNLite(nn.Module):
    def __init__(self, num_users: int, num_notes: int, dense_dim: int, embedding_dim: int, hidden_dims: list[int], dropout: float):
        super().__init__()
        self.model_kind = "dcn"
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(num_notes, embedding_dim, padding_idx=0)
        input_dim = embedding_dim * 2 + dense_dim
        self.cross_weight = nn.Parameter(torch.randn(input_dim) * 0.01)
        self.cross_bias = nn.Parameter(torch.zeros(input_dim))
        self.mlp = mlp(input_dim * 2, hidden_dims, 1, dropout)

    def forward(self, user_id: torch.Tensor, note_id: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        x0 = torch.cat([self.user_embedding(user_id), self.note_embedding(note_id), dense], dim=-1)
        cross = x0 * torch.sum(x0 * self.cross_weight, dim=-1, keepdim=True) + self.cross_bias + x0
        return self.mlp(torch.cat([x0, cross], dim=-1)).squeeze(-1)


class SIMRanker(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_notes: int,
        dense_dim: int,
        embedding_dim: int,
        hidden_dims: list[int],
        dropout: float,
        output_dim: int = 1,
    ):
        super().__init__()
        self.model_kind = "sim"
        self.user_embedding = nn.Embedding(num_users, embedding_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(num_notes, embedding_dim, padding_idx=0)
        input_dim = embedding_dim * 5 + dense_dim
        self.mlp = mlp(input_dim, hidden_dims, output_dim, dropout)

    def _attend(self, target: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        emb = self.note_embedding(seq)
        mask = (seq != 0).float()
        scores = (emb * target.unsqueeze(1)).sum(dim=-1)
        scores = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1) * mask
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        return torch.sum(emb * weights.unsqueeze(-1), dim=1)

    def forward(
        self,
        user_id: torch.Tensor,
        note_id: torch.Tensor,
        dense: torch.Tensor,
        lastn_seq: torch.Tensor,
        topk_seq: torch.Tensor,
    ) -> torch.Tensor:
        user = self.user_embedding(user_id)
        target = self.note_embedding(note_id)
        lastn = self._attend(target, lastn_seq)
        topk = self._attend(target, topk_seq)
        x = torch.cat([user, target, lastn, topk, user * target, dense], dim=-1)
        return self.mlp(x).squeeze(-1)
