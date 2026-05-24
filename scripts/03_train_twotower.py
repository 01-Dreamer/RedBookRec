from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cli import add_config_arguments, load_config_with_overrides
from src.data.io import get_device, read_json, save_parquet, setup_logging
from src.models.two_tower import TwoTowerModel
from src.recall.merge import merge_recall_frames
from src.recall.twotower_recall import TwoTowerRecall
from src.train.train_twotower import train_twotower


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, include_training=True)
    args = parser.parse_args()
    config = load_config_with_overrides(args, ["configs/base.yaml", "configs/twotower.yaml"])
    checkpoint = train_twotower(config)

    processed = Path(config["paths"]["processed_dir"])
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = TwoTowerModel(ckpt["num_users"], ckpt["num_items"], ckpt["embedding_dim"], ckpt["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    device = get_device(config)
    notes = pd.read_parquet(processed / "features" / "note_features.parquet")
    profiles = pd.read_parquet(processed / "features" / "user_profiles.parquet").head(int(config["num_eval_users"]))
    train = pd.read_parquet(processed / "samples" / "train_interactions.parquet")
    user_idx_map = train[["user_id", "user_idx_internal", "history_idx_internal"]].drop_duplicates("user_id")
    profiles = profiles.merge(user_idx_map, on="user_id", how="left")
    mappings = read_json(processed / "mappings" / "id_mappings.json")

    tw_recall = TwoTowerRecall(model, device).recommend(
        profiles.dropna(subset=["user_idx_internal"]),
        notes,
        mappings,
        top_k=int(config["recall_top_k"]),
    )
    save_parquet(tw_recall, processed / "recalls" / "twotower_recall.parquet")
    frames = [tw_recall]
    classic_path = processed / "recalls" / "merged_recall.parquet"
    if classic_path.exists():
        frames.append(pd.read_parquet(classic_path))
    merged = merge_recall_frames(frames, top_k_per_user=300)
    save_parquet(merged, processed / "recalls" / "merged_recall.parquet")
    print(f"Saved TwoTower checkpoint: {checkpoint}")
    print(f"Saved TwoTower recall rows: {len(tw_recall)}")


if __name__ == "__main__":
    main()
