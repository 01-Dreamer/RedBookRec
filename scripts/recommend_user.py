from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, save_run_config
from redbookrec.recommend import recommend


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--user-id", "--user_id", dest="user_id", type=int, default=0)
    parser.add_argument("--query", default="")
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=20)
    parser.add_argument("--random-user", "--random_user", dest="random_user", action="store_true")
    args = parser.parse_args()
    cfg = build_config(args, "recommend")
    ensure_dirs(cfg)
    save_run_config(cfg, "recommend")

    processed = Path(cfg["paths"]["processed_dir"])
    user_id = int(args.user_id)
    if args.random_user:
        users = pd.read_parquet(processed / "users.parquet")
        user_id = int(users.sample(1, random_state=int(cfg["runtime"]["seed"]))["user_id"].iloc[0])
    recs = recommend(cfg, user_id=user_id, query=args.query, top_k=int(args.top_k))
    cols = [c for c in ["rerank_position", "note_id", "raw_note_idx", "note_title", "score", "recall_source", "taxonomy1_id"] if c in recs.columns]
    print(f"user_id={user_id} query={args.query!r}")
    print(recs[cols].to_string(index=False, max_colwidth=40))


if __name__ == "__main__":
    main()
