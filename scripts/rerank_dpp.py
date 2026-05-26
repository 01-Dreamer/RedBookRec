from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, save_run_config
from redbookrec.rerank import greedy_dpp


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    args = parser.parse_args()
    cfg = build_config(args, "rerank")
    ensure_dirs(cfg)
    save_run_config(cfg, "rerank")

    notes = pd.read_parquet(Path(cfg["paths"]["processed_dir"]) / "notes.parquet")
    top_k = int(args.top_k or cfg["rerank"]["final_top_k"])
    candidates = notes.sort_values("popularity_score", ascending=False).head(max(top_k * 5, top_k)).copy()
    candidates["score"] = candidates["popularity_score"]
    out = greedy_dpp(candidates, score_col="score", top_k=top_k, diversity_weight=float(cfg["rerank"]["diversity_weight"]))
    path = Path(cfg["paths"]["processed_dir"]) / "sample_dpp_rerank.parquet"
    out.to_parquet(path, index=False)
    print(f"saved_rerank_sample={path} rows={len(out)}")


if __name__ == "__main__":
    main()
