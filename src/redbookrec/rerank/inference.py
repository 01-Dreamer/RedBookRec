from __future__ import annotations

from pathlib import Path

import pandas as pd

from redbookrec.rerank.dpp import greedy_dpp


def run_dpp(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(cfg["infer"]["input_path"])
    try:
        note_meta = pd.read_parquet(
            cfg["data"]["note_text_path"],
            columns=["note_idx", "note_type", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id", "content_length"],
        )
    except Exception:
        note_meta = pd.DataFrame(columns=["note_idx", "note_type", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id", "content_length"])
    frames = [
        greedy_dpp(
            group,
            note_meta,
            int(cfg["rerank"].get("top_k", 10)),
            float(cfg["rerank"].get("lambda_diversity", 0.2)),
            score_col=str(cfg["rerank"].get("score_col", "sim_score")),
        )
        for _, group in df.groupby("request_idx")
    ]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    keep = ["request_idx", "user_idx", "note_idx", "sim_score", "dpp_score", "final_rank", "label_click"]
    out = out[keep] if not out.empty else pd.DataFrame(columns=keep)
    path = Path(cfg["infer"]["output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
