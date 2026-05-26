from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, save_run_config, write_json
from redbookrec.data import expand_recommendation, load_mapping, map_with_default
from redbookrec.features import add_note_features
from redbookrec.metrics import evaluate_ranked_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    args = parser.parse_args()
    cfg = build_config(args, "evaluate")
    ensure_dirs(cfg)
    save_run_config(cfg, "evaluate")
    processed = Path(cfg["paths"]["processed_dir"])
    if not (processed / "mappings" / "user_id_map.parquet").exists():
        raise SystemExit("Run scripts/prepare_data.py first.")

    interactions = expand_recommendation(cfg, "recommendation_test")
    user_map = load_mapping(processed / "mappings" / "user_id_map.parquet", "raw_user_idx", "user_id")
    note_map = load_mapping(processed / "mappings" / "note_id_map.parquet", "raw_note_idx", "note_id")
    interactions["user_id"] = map_with_default(interactions["raw_user_idx"], user_map, 0)
    interactions["note_id"] = map_with_default(interactions["raw_note_idx"], note_map, 0)
    notes = pd.read_parquet(processed / "notes.parquet")
    notes = add_note_features(notes)
    scores = notes[["note_id", "popularity_score"]]
    eval_df = interactions.merge(scores, on="note_id", how="left")
    eval_df["score"] = eval_df["popularity_score"].fillna(0.0)
    eval_df = eval_df.sort_values(["request_idx", "score"], ascending=[True, False])
    groups = [g["click_label"].astype(int).tolist() for _, g in eval_df.groupby("request_idx")]
    ks = [int(args.top_k)] if args.top_k else [int(k) for k in cfg["evaluation"]["top_k"]]
    metrics = evaluate_ranked_groups(groups, ks)
    metrics["eval_rows"] = int(len(eval_df))
    metrics["eval_requests"] = int(eval_df["request_idx"].nunique())
    out = Path(cfg["paths"]["metrics_dir"]) / "eval_summary.json"
    write_json(out, metrics)
    print(metrics)
    print(f"saved_metrics={out}")


if __name__ == "__main__":
    main()
