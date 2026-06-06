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


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(last, dim))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        last = dim
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class DCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_notes: int,
        dense_dim: int = 49,
        embed_dim: int = 32,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        num_user_cat: int = 4,
        num_note_types: int = 16,
        num_taxonomy: int = 4096,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(num_notes, embed_dim, padding_idx=0)
        self.user_cat_embeddings = nn.ModuleList([nn.Embedding(128, embed_dim, padding_idx=0) for _ in range(num_user_cat)])
        self.note_type_embedding = nn.Embedding(num_note_types, embed_dim, padding_idx=0)
        self.tax1_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax2_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax3_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.dense_proj = nn.Sequential(nn.Linear(dense_dim, embed_dim), nn.ReLU())
        input_dim = embed_dim * (2 + num_user_cat + 1 + 3 + 1)
        self.cross = CrossLayer(input_dim)
        self.mlp = mlp(input_dim * 2, hidden_dims or [128, 64], 1, dropout)
        self.dense_dim = int(dense_dim)
        self.embed_dim = int(embed_dim)
        self.num_user_cat = int(num_user_cat)

    def forward(
        self,
        user_id: torch.Tensor,
        note_id: torch.Tensor,
        user_cat: torch.Tensor,
        note_type: torch.Tensor,
        note_tax: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        cat_embs = [
            emb(user_cat[:, i].clamp_min(0).clamp_max(emb.num_embeddings - 1))
            for i, emb in enumerate(self.user_cat_embeddings)
        ]
        x0 = torch.cat(
            [
                self.user_embedding(user_id),
                self.note_embedding(note_id),
                *cat_embs,
                self.note_type_embedding(note_type.clamp_min(0).clamp_max(self.note_type_embedding.num_embeddings - 1)),
                self.tax1_embedding(note_tax[:, 0].clamp_min(0).clamp_max(self.tax1_embedding.num_embeddings - 1)),
                self.tax2_embedding(note_tax[:, 1].clamp_min(0).clamp_max(self.tax2_embedding.num_embeddings - 1)),
                self.tax3_embedding(note_tax[:, 2].clamp_min(0).clamp_max(self.tax3_embedding.num_embeddings - 1)),
                self.dense_proj(dense.float()),
            ],
            dim=-1,
        )
        x1 = self.cross(x0, x0)
        return self.mlp(torch.cat([x0, x1], dim=-1)).squeeze(-1)
