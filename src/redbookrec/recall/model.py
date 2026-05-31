from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


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


class UserTower(nn.Module):
    def __init__(self, num_users: int, note_embedding: nn.Embedding, embed_dim: int, user_dense_dim: int = 42, dropout: float = 0.1):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.history_embedding = note_embedding
        self.user_cat_embeddings = nn.ModuleList([nn.Embedding(128, embed_dim, padding_idx=0) for _ in range(4)])
        self.user_dense_proj = nn.Sequential(nn.Linear(user_dense_dim, embed_dim), nn.ReLU())
        self.encoder = mlp(embed_dim * 7, [embed_dim * 3, embed_dim * 2], embed_dim, dropout)

    def forward(
        self,
        user_id: torch.Tensor,
        recent_clicked_note_ids: torch.Tensor,
        user_cat: torch.Tensor | None = None,
        user_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user = self.user_embedding(user_id)
        hist_emb = self.history_embedding(recent_clicked_note_ids)
        mask = (recent_clicked_note_ids != 0).float().unsqueeze(-1)
        hist = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if user_cat is None:
            user_cat = torch.zeros(user_id.size(0), 4, dtype=torch.long, device=user_id.device)
        if user_dense is None:
            user_dense = torch.zeros(user_id.size(0), self.user_dense_proj[0].in_features, device=user_id.device)
        cat_embs = [
            emb(user_cat[:, i].clamp_min(0).clamp_max(emb.num_embeddings - 1))
            for i, emb in enumerate(self.user_cat_embeddings)
        ]
        dense = self.user_dense_proj(user_dense.float())
        return F.normalize(self.encoder(torch.cat([user, hist, *cat_embs, dense], dim=-1)), dim=-1)


class NoteTower(nn.Module):
    def __init__(
        self,
        note_embedding: nn.Embedding,
        embed_dim: int,
        num_note_types: int = 16,
        num_taxonomy: int = 4096,
        note_dense_dim: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.note_embedding = note_embedding
        self.type_embedding = nn.Embedding(num_note_types, embed_dim, padding_idx=0)
        self.tax1_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax2_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax3_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.note_dense_proj = nn.Sequential(nn.Linear(note_dense_dim, embed_dim), nn.ReLU())
        self.encoder = mlp(embed_dim * 6, [embed_dim * 3, embed_dim * 2], embed_dim, dropout)

    def forward(
        self,
        note_id: torch.Tensor,
        note_type: torch.Tensor | None = None,
        tax1: torch.Tensor | None = None,
        tax2: torch.Tensor | None = None,
        tax3: torch.Tensor | None = None,
        note_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(note_id)
        note_type = zeros if note_type is None else note_type
        tax1 = zeros if tax1 is None else tax1
        tax2 = zeros if tax2 is None else tax2
        tax3 = zeros if tax3 is None else tax3
        if note_dense is None:
            note_dense = torch.zeros(note_id.size(0), self.note_dense_proj[0].in_features, device=note_id.device)
        x = torch.cat(
            [
                self.note_embedding(note_id),
                self.type_embedding(note_type.clamp_min(0).clamp_max(self.type_embedding.num_embeddings - 1)),
                self.tax1_embedding(tax1.clamp_min(0).clamp_max(self.tax1_embedding.num_embeddings - 1)),
                self.tax2_embedding(tax2.clamp_min(0).clamp_max(self.tax2_embedding.num_embeddings - 1)),
                self.tax3_embedding(tax3.clamp_min(0).clamp_max(self.tax3_embedding.num_embeddings - 1)),
                self.note_dense_proj(note_dense.float()),
            ],
            dim=-1,
        )
        return F.normalize(self.encoder(x), dim=-1)


class DualTowerRecall(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_notes: int,
        embed_dim: int = 64,
        dropout: float = 0.1,
        temperature: float = 0.05,
        user_dense_dim: int = 42,
        note_dense_dim: int = 4,
    ):
        super().__init__()
        self.note_embedding = nn.Embedding(num_notes, embed_dim, padding_idx=0)
        self.user_tower = UserTower(num_users, self.note_embedding, embed_dim, user_dense_dim=user_dense_dim, dropout=dropout)
        self.note_tower = NoteTower(self.note_embedding, embed_dim, note_dense_dim=note_dense_dim, dropout=dropout)
        self.temperature = float(temperature)
        self.num_users = int(num_users)
        self.num_notes = int(num_notes)
        self.embed_dim = int(embed_dim)
        self.user_dense_dim = int(user_dense_dim)
        self.note_dense_dim = int(note_dense_dim)

    def encode_user(
        self,
        user_id: torch.Tensor,
        history_note_ids: torch.Tensor,
        user_cat: torch.Tensor | None = None,
        user_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.user_tower(user_id, history_note_ids, user_cat, user_dense)

    def encode_note(
        self,
        note_id: torch.Tensor,
        note_type: torch.Tensor | None = None,
        tax1: torch.Tensor | None = None,
        tax2: torch.Tensor | None = None,
        tax3: torch.Tensor | None = None,
        note_dense: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.note_tower(note_id, note_type, tax1, tax2, tax3, note_dense)

    def forward(
        self,
        user_id: torch.Tensor,
        history_note_ids: torch.Tensor,
        note_id: torch.Tensor,
        user_cat: torch.Tensor | None = None,
        user_dense: torch.Tensor | None = None,
        note_type: torch.Tensor | None = None,
        note_tax: torch.Tensor | None = None,
        note_dense: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if note_tax is None:
            note_tax = torch.zeros(note_id.size(0), 3, dtype=torch.long, device=note_id.device)
        return self.encode_user(user_id, history_note_ids, user_cat, user_dense), self.encode_note(
            note_id,
            note_type,
            note_tax[:, 0],
            note_tax[:, 1],
            note_tax[:, 2],
            note_dense,
        )
