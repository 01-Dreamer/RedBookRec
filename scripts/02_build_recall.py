from __future__ import annotations

import pickle
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cli import add_config_arguments, load_config_with_overrides
from src.data.io import save_parquet, setup_logging
from src.recall.itemcf import ItemCFRecall
from src.recall.merge import merge_recall_frames
from src.recall.popular import PopularRecall
from src.recall.text_recall import TextRecall


def histories_from_profiles(profiles: pd.DataFrame) -> dict[int, list[int]]:
    return {int(row.user_id): [int(x) for x in row.history_note_ids] for row in profiles.itertuples(index=False)}


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args()
    config = load_config_with_overrides(args, ["configs/base.yaml", "configs/recall.yaml"])
    processed = Path(config["paths"]["processed_dir"])
    train = pd.read_parquet(processed / "samples" / "train_interactions.parquet")
    notes = pd.read_parquet(processed / "features" / "note_features.parquet")
    profiles = pd.read_parquet(processed / "features" / "user_profiles.parquet")
    histories = histories_from_profiles(profiles)

    print("Fitting PopularRecall...", flush=True)
    popular = PopularRecall(top_k=int(config["popular"]["top_k"])).fit(notes, train)
    print("Fitting ItemCFRecall...", flush=True)
    itemcf = ItemCFRecall(
        top_k=int(config["itemcf"]["top_k"]),
        max_history_len=int(config["itemcf"]["max_history_len"]),
        max_neighbors_per_item=int(config["itemcf"]["max_neighbors_per_item"]),
    ).fit(train)

    rows = []
    print("Generating popular/itemcf candidates...", flush=True)
    for row in profiles.itertuples(index=False):
        history = histories[int(row.user_id)]
        rows.extend([{"user_id": int(row.user_id), **item} for item in popular.recommend(history)])
        rows.extend([{"user_id": int(row.user_id), **item} for item in itemcf.recommend(history)])
    recall_frames = [pd.DataFrame(rows)]

    if config["text"].get("enabled", True):
        print("Fitting TextRecall...", flush=True)
        text_users = (
            train.groupby("user_id")["query"]
            .agg(lambda x: " ".join(str(v) for v in x.dropna().head(3)))
            .reset_index()
            .head(int(config["text"]["max_users"]))
        )
        text_notes = notes.sort_values("popularity_score", ascending=False).head(int(config["text"].get("max_notes", len(notes))))
        text_recall = TextRecall(top_k=int(config["text"]["top_k"]), max_features=int(config["text"]["max_features"])).fit(text_notes)
        recall_frames.append(text_recall.recommend_for_queries(text_users, histories))

    print("Merging recall candidates...", flush=True)
    merged = merge_recall_frames(recall_frames, top_k_per_user=int(config["merge"]["top_k_per_user"]))
    save_parquet(merged, processed / "recalls" / "merged_recall.parquet")
    save_parquet(recall_frames[0], processed / "recalls" / "classic_recall.parquet")
    with (processed / "recalls" / "popular_itemcf.pkl").open("wb") as f:
        pickle.dump({"popular": popular, "itemcf": itemcf}, f)
    print(f"Saved merged recall rows: {len(merged)}")


if __name__ == "__main__":
    main()
