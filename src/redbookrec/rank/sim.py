from __future__ import annotations

import torch
from torch import nn


def mlp(input_dim: int, hidden_dims: list[int], output_dim: int, dropout: float = 0.0) -> nn.Sequential:
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


class SIMRanker(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_notes: int,
        dense_dim: int,
        embed_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        num_user_cat: int = 4,
        num_note_types: int = 16,
        num_taxonomy: int = 4096,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.note_embedding = nn.Embedding(num_notes, embed_dim, padding_idx=0)
        self.user_cat_embeddings = nn.ModuleList([nn.Embedding(128, embed_dim, padding_idx=0) for _ in range(num_user_cat)])
        self.note_type_embedding = nn.Embedding(num_note_types, embed_dim, padding_idx=0)
        self.tax1_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax2_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax3_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.dense_proj = nn.Sequential(nn.Linear(dense_dim, embed_dim), nn.ReLU())
        input_dim = embed_dim * (1 + 1 + 2 + num_user_cat + 1 + 3 + 1)
        self.mlp = mlp(input_dim, hidden_dims or [256, 128, 64], 1, dropout)
        self.dense_dim = int(dense_dim)
        self.embed_dim = int(embed_dim)

    def _history_interest(self, history_note_ids: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hist_emb = self.note_embedding(history_note_ids)
        mask = (history_note_ids != 0).float().unsqueeze(-1)
        mean_interest = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        attn_logits = (hist_emb * target.unsqueeze(1)).sum(dim=-1).masked_fill(history_note_ids == 0, -1e9)
        attn = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        attn_interest = (hist_emb * attn).sum(dim=1)
        has_history = (history_note_ids != 0).any(dim=1, keepdim=True)
        attn_interest = torch.where(has_history, attn_interest, mean_interest)
        return mean_interest, attn_interest

    def forward(
        self,
        user_id: torch.Tensor,
        note_id: torch.Tensor,
        history_note_ids: torch.Tensor,
        user_cat: torch.Tensor,
        note_type: torch.Tensor,
        note_tax: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        user = self.user_embedding(user_id)
        target = self.note_embedding(note_id)
        mean_interest, attn_interest = self._history_interest(history_note_ids, target)
        cat_embs = [
            emb(user_cat[:, i].clamp_min(0).clamp_max(emb.num_embeddings - 1))
            for i, emb in enumerate(self.user_cat_embeddings)
        ]
        x = torch.cat(
            [
                user,
                target,
                mean_interest,
                attn_interest,
                *cat_embs,
                self.note_type_embedding(note_type.clamp_min(0).clamp_max(self.note_type_embedding.num_embeddings - 1)),
                self.tax1_embedding(note_tax[:, 0].clamp_min(0).clamp_max(self.tax1_embedding.num_embeddings - 1)),
                self.tax2_embedding(note_tax[:, 1].clamp_min(0).clamp_max(self.tax2_embedding.num_embeddings - 1)),
                self.tax3_embedding(note_tax[:, 2].clamp_min(0).clamp_max(self.tax3_embedding.num_embeddings - 1)),
                self.dense_proj(dense.float()),
            ],
            dim=-1,
        )
        return self.mlp(x).squeeze(-1)
