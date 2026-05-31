from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def infer_prerank(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["infer"]["input_path"])
    df = pd.read_parquet(path)
    base = df["hybrid_score"].astype(float).fillna(0.0)
    rank_bonus = 1.0 / np.log2(df["hybrid_rank"].astype(float).fillna(1.0) + 2.0)
    df["dcn_score"] = 0.85 * base + 0.15 * rank_bonus
    df = df.sort_values(["request_idx", "dcn_score"], ascending=[True, False])
    df["dcn_rank"] = df.groupby("request_idx").cumcount() + 1
    df = df[df["dcn_rank"] <= int(cfg["infer"].get("top_k", 200))]
    out = df[["request_idx", "user_idx", "note_idx", "hybrid_score", "dcn_score", "dcn_rank", "label_click"]].copy()
    out_path = Path(cfg["infer"]["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
