from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def infer_rank(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(cfg["infer"]["input_path"])
    note_mod = (df["note_idx"].astype("int64") % 997).astype(float) / 997.0
    df["sim_score"] = 0.9 * df["dcn_score"].astype(float).fillna(0.0) + 0.1 * note_mod
    df = df.sort_values(["request_idx", "sim_score"], ascending=[True, False])
    df["sim_rank"] = df.groupby("request_idx").cumcount() + 1
    df = df[df["sim_rank"] <= int(cfg["infer"].get("top_k", 50))]
    out = df[["request_idx", "user_idx", "note_idx", "dcn_score", "sim_score", "sim_rank", "label_click"]].copy()
    out_path = Path(cfg["infer"]["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out
