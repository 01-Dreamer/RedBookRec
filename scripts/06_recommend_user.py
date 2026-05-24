from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cli import add_config_arguments, load_config_with_overrides
from src.data.io import save_parquet, setup_logging
from src.rerank.diversity import diversity_rerank
from src.service.recommender import build_candidate_samples, score_with_ranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    parser.add_argument("--user_id", type=int, default=None)
    parser.add_argument("--random_user", action="store_true")
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def snippet(text: str, n: int = 80) -> str:
    text = " ".join(str(text or "").split())
    return text[:n] + ("..." if len(text) > n else "")


def main() -> None:
    setup_logging()
    args = parse_args()
    config = load_config_with_overrides(args, ["configs/base.yaml", "configs/ranker.yaml"])
    processed = Path(config["paths"]["processed_dir"])
    outputs = Path(config["paths"]["outputs_dir"]) / "recommendations"
    outputs.mkdir(parents=True, exist_ok=True)

    profiles = pd.read_parquet(processed / "features" / "user_profiles.parquet")
    notes = pd.read_parquet(processed / "features" / "note_features.parquet")
    recalls_path = processed / "recalls" / "merged_recall.parquet"
    if not recalls_path.exists():
        raise FileNotFoundError("Missing data_processed/recalls/merged_recall.parquet. Run scripts/02_build_recall.py first.")
    recalls = pd.read_parquet(recalls_path)

    if args.random_user or args.user_id is None:
        valid_users = sorted(set(recalls["user_id"]).intersection(set(profiles["user_id"])))
        user_id = random.choice(valid_users)
    else:
        user_id = args.user_id
    user_profile = profiles[profiles["user_id"] == user_id]
    if user_profile.empty:
        raise ValueError(f"user_id={user_id} not found in processed user profiles.")
    history = [int(x) for x in user_profile.iloc[0]["history_note_ids"]]
    candidates = recalls[recalls["user_id"] == user_id].copy()
    if candidates.empty:
        raise ValueError(f"No recall candidates for user_id={user_id}.")

    samples = build_candidate_samples(candidates, config)
    ranked = score_with_ranker(samples, config)
    final = diversity_rerank(ranked, notes, history, top_k=args.top_k)
    save_parquet(final, outputs / f"user_{user_id}_recommendations.parquet")

    note_lookup = notes.set_index("note_id")
    print(f"\nuser_id: {user_id}")
    print(f"profile: positives={int(user_profile.iloc[0]['positive_count'])}, impressions={int(user_profile.iloc[0]['impression_count'])}")
    print(f"top_categories: {user_profile.iloc[0]['top_categories']}")
    print("\nrecent history:")
    for note_id in history[-5:]:
        if note_id in note_lookup.index:
            print(f"  - {note_id}: {snippet(note_lookup.loc[note_id, 'title'], 42)}")
        else:
            print(f"  - {note_id}")

    print("\nTop recommendations:")
    display_cols = ["note_id", "final_score", "recall_source", "rerank_reason", "title", "content"]
    for rank, row in enumerate(final[display_cols].itertuples(index=False), start=1):
        print(f"{rank:02d}. note_id={row.note_id} score={row.final_score:.4f} source={row.recall_source}")
        print(f"    title: {snippet(row.title, 60)}")
        print(f"    text : {snippet(row.content, 90)}")
        print(f"    rerank: {row.rerank_reason}")


if __name__ == "__main__":
    main()
