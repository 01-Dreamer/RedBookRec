from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from redbookrec.prerank.dataset import PrerankDataset, build_prerank_infer_frame
from redbookrec.prerank.dcn import DCN
from redbookrec.recall.dataset import build_note_features
from redbookrec.utils.config import get_device


def _load_model(cfg: dict, device: torch.device) -> tuple[DCN, dict]:
    ckpt = torch.load(cfg["paths"]["checkpoint"], map_location=device)
    model = DCN(
        num_users=int(ckpt["num_users"]),
        num_notes=int(ckpt["num_notes"]),
        dense_dim=int(ckpt["dense_dim"]),
        embed_dim=int(ckpt["embed_dim"]),
        hidden_dims=list(ckpt.get("hidden_dims", [128, 64])),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def infer_prerank(cfg: dict) -> pd.DataFrame:
    device = torch.device(get_device(cfg["train"].get("device", "auto")))
    model, ckpt = _load_model(cfg, device)
    df = build_prerank_infer_frame(pd.read_parquet(cfg["infer"]["input_path"]))
    if df.empty:
        out = pd.DataFrame(columns=["request_idx", "user_idx", "note_idx", "hybrid_score", "dcn_score", "dcn_rank", "label_click"])
    else:
        note_df = pd.read_parquet(cfg["data"]["note_text_path"])
        needed_notes = set(df["note_idx"].astype("int64").tolist())
        note_df = note_df[note_df["note_idx"].astype("int64").isin(needed_notes)]
        note_features = build_note_features(note_df)
        dataset = PrerankDataset(df, ckpt["note_map"], ckpt["user_map"], note_features, ckpt.get("user_features", {}))
        loader = DataLoader(dataset, batch_size=int(cfg["infer"].get("batch_size", 8192)), shuffle=False, num_workers=0)
        scores: list[float] = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="infer_dcn", leave=False):
                logits = model(
                    batch["user_id"].to(device),
                    batch["note_id"].to(device),
                    batch["user_cat"].to(device),
                    batch["note_type"].to(device),
                    batch["note_tax"].to(device),
                    batch["dense"].to(device),
                )
                scores.extend(torch.sigmoid(logits).cpu().tolist())
        df["dcn_score"] = scores
        df = df.sort_values(["request_idx", "dcn_score"], ascending=[True, False])
        df["dcn_rank"] = df.groupby("request_idx").cumcount() + 1
        df = df[df["dcn_rank"] <= int(cfg["infer"].get("top_k", 200))]
        out = df[["request_idx", "user_idx", "note_idx", "hybrid_score", "dcn_score", "dcn_rank", "label_click"]].copy()
    out_path = Path(cfg["infer"]["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
