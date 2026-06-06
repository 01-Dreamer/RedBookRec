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


def masked_mean_embedding(embedding: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
    emb = embedding(ids)
    mask = (ids != 0).float().unsqueeze(-1)
    return (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class UserTower(nn.Module):
    def __init__(
        self,
        num_users: int,
        note_embedding: nn.Embedding,
        text_embedding: nn.Embedding,
        embed_dim: int,
        user_dense_dim: int = 42,
        text_emb_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim, padding_idx=0)
        self.history_embedding = note_embedding
        self.text_embedding = text_embedding
        self.user_cat_embeddings = nn.ModuleList([nn.Embedding(128, embed_dim, padding_idx=0) for _ in range(4)])
        self.user_dense_proj = nn.Sequential(nn.Linear(user_dense_dim, embed_dim), nn.ReLU())
        self.query_text_proj = nn.Sequential(nn.Linear(text_emb_dim, embed_dim), nn.ReLU())
        self.encoder = mlp(embed_dim * 8, [embed_dim * 3, embed_dim * 2], embed_dim, dropout)

    def forward(
        self,
        user_id: torch.Tensor,
        recent_clicked_note_ids: torch.Tensor,
        user_cat: torch.Tensor | None = None,
        user_dense: torch.Tensor | None = None,
        query_text_ids: torch.Tensor | None = None,
        query_text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user = self.user_embedding(user_id)
        hist_emb = self.history_embedding(recent_clicked_note_ids)
        mask = (recent_clicked_note_ids != 0).float().unsqueeze(-1)
        hist = (hist_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if query_text_ids is None:
            query_text_ids = torch.zeros(user_id.size(0), 1, dtype=torch.long, device=user_id.device)
        query_text = masked_mean_embedding(self.text_embedding, query_text_ids)
        if query_text_emb is not None:
            query_text = query_text + self.query_text_proj(query_text_emb.float())
        if user_cat is None:
            user_cat = torch.zeros(user_id.size(0), 4, dtype=torch.long, device=user_id.device)
        if user_dense is None:
            user_dense = torch.zeros(user_id.size(0), self.user_dense_proj[0].in_features, device=user_id.device)
        cat_embs = [
            emb(user_cat[:, i].clamp_min(0).clamp_max(emb.num_embeddings - 1))
            for i, emb in enumerate(self.user_cat_embeddings)
        ]
        dense = self.user_dense_proj(user_dense.float())
        return F.normalize(self.encoder(torch.cat([user, hist, query_text, *cat_embs, dense], dim=-1)), dim=-1)


class NoteTower(nn.Module):
    def __init__(
        self,
        note_embedding: nn.Embedding,
        text_embedding: nn.Embedding,
        embed_dim: int,
        num_note_types: int = 16,
        num_taxonomy: int = 4096,
        note_dense_dim: int = 2,
        text_emb_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.note_embedding = note_embedding
        self.text_embedding = text_embedding
        self.type_embedding = nn.Embedding(num_note_types, embed_dim, padding_idx=0)
        self.tax1_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax2_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.tax3_embedding = nn.Embedding(num_taxonomy, embed_dim, padding_idx=0)
        self.note_dense_proj = nn.Sequential(nn.Linear(note_dense_dim, embed_dim), nn.ReLU())
        self.note_text_proj = nn.Sequential(nn.Linear(text_emb_dim, embed_dim), nn.ReLU())
        self.encoder = mlp(embed_dim * 7, [embed_dim * 3, embed_dim * 2], embed_dim, dropout)

    def forward(
        self,
        note_id: torch.Tensor,
        note_type: torch.Tensor | None = None,
        tax1: torch.Tensor | None = None,
        tax2: torch.Tensor | None = None,
        tax3: torch.Tensor | None = None,
        note_dense: torch.Tensor | None = None,
        note_text_ids: torch.Tensor | None = None,
        note_text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        zeros = torch.zeros_like(note_id)
        note_type = zeros if note_type is None else note_type
        tax1 = zeros if tax1 is None else tax1
        tax2 = zeros if tax2 is None else tax2
        tax3 = zeros if tax3 is None else tax3
        if note_dense is None:
            note_dense = torch.zeros(note_id.size(0), self.note_dense_proj[0].in_features, device=note_id.device)
        if note_text_ids is None:
            note_text_ids = torch.zeros(note_id.size(0), 1, dtype=torch.long, device=note_id.device)
        note_text = masked_mean_embedding(self.text_embedding, note_text_ids)
        if note_text_emb is not None:
            note_text = note_text + self.note_text_proj(note_text_emb.float())
        x = torch.cat(
            [
                self.note_embedding(note_id),
                note_text,
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
        note_dense_dim: int = 2,
        text_vocab_size: int = 50000,
        text_emb_dim: int = 768,
    ):
        super().__init__()
        self.note_embedding = nn.Embedding(num_notes, embed_dim, padding_idx=0)
        self.text_embedding = nn.Embedding(text_vocab_size, embed_dim, padding_idx=0)
        self.user_tower = UserTower(
            num_users,
            self.note_embedding,
            self.text_embedding,
            embed_dim,
            user_dense_dim=user_dense_dim,
            text_emb_dim=text_emb_dim,
            dropout=dropout,
        )
        self.note_tower = NoteTower(
            self.note_embedding,
            self.text_embedding,
            embed_dim,
            note_dense_dim=note_dense_dim,
            text_emb_dim=text_emb_dim,
            dropout=dropout,
        )
        self.temperature = float(temperature)
        self.num_users = int(num_users)
        self.num_notes = int(num_notes)
        self.embed_dim = int(embed_dim)
        self.user_dense_dim = int(user_dense_dim)
        self.note_dense_dim = int(note_dense_dim)
        self.text_vocab_size = int(text_vocab_size)
        self.text_emb_dim = int(text_emb_dim)

    def encode_user(
        self,
        user_id: torch.Tensor,
        history_note_ids: torch.Tensor,
        user_cat: torch.Tensor | None = None,
        user_dense: torch.Tensor | None = None,
        query_text_ids: torch.Tensor | None = None,
        query_text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.user_tower(user_id, history_note_ids, user_cat, user_dense, query_text_ids, query_text_emb)

    def encode_note(
        self,
        note_id: torch.Tensor,
        note_type: torch.Tensor | None = None,
        tax1: torch.Tensor | None = None,
        tax2: torch.Tensor | None = None,
        tax3: torch.Tensor | None = None,
        note_dense: torch.Tensor | None = None,
        note_text_ids: torch.Tensor | None = None,
        note_text_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.note_tower(note_id, note_type, tax1, tax2, tax3, note_dense, note_text_ids, note_text_emb)

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
        query_text_ids: torch.Tensor | None = None,
        note_text_ids: torch.Tensor | None = None,
        query_text_emb: torch.Tensor | None = None,
        note_text_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if note_tax is None:
            note_tax = torch.zeros(note_id.size(0), 3, dtype=torch.long, device=note_id.device)
        return self.encode_user(user_id, history_note_ids, user_cat, user_dense, query_text_ids, query_text_emb), self.encode_note(
            note_id,
            note_type,
            note_tax[:, 0],
            note_tax[:, 1],
            note_tax[:, 2],
            note_dense,
            note_text_ids,
            note_text_emb,
        )
