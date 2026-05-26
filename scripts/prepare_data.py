from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, save_run_config
from redbookrec.data import (
    build_mapping,
    build_user_history,
    expand_recommendation,
    map_history,
    map_with_default,
    read_notes,
    read_users,
    save_json,
    save_mapping,
)
from redbookrec.features import add_note_features, join_training_features


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = build_config(args, "prepare")
    ensure_dirs(cfg)
    save_run_config(cfg, "prepare")

    processed = Path(cfg["paths"]["processed_dir"])
    mappings_dir = processed / "mappings"
    default_user = int(cfg["id_mapping"]["default_user_id"])
    default_note = int(cfg["id_mapping"]["default_note_id"])

    print("loading recommendation_train...")
    interactions = expand_recommendation(cfg, "recommendation_train")
    print(f"expanded_interactions={len(interactions)}")

    print("loading notes/users...")
    notes = read_notes(cfg).rename(columns={"note_idx": "raw_note_idx"})
    users = read_users(cfg).rename(columns={"user_idx": "raw_user_idx"})

    raw_user_values = pd.concat([users["raw_user_idx"], interactions["raw_user_idx"]], ignore_index=True)
    raw_note_values = [notes["raw_note_idx"], interactions["raw_note_idx"]]
    history_values = interactions["history_raw_note_idxs"].explode().dropna().astype("int64")
    raw_note_values.append(history_values)
    user_map = build_mapping(raw_user_values, default_user)
    note_map = build_mapping(pd.concat(raw_note_values, ignore_index=True), default_note)

    users["user_id"] = map_with_default(users["raw_user_idx"], user_map, default_user)
    notes["note_id"] = map_with_default(notes["raw_note_idx"], note_map, default_note)
    interactions["user_id"] = map_with_default(interactions["raw_user_idx"], user_map, default_user)
    interactions["note_id"] = map_with_default(interactions["raw_note_idx"], note_map, default_note)
    max_history = int(cfg["limits"].get("max_history") or 200)
    interactions["history_note_ids"] = interactions["history_raw_note_idxs"].map(
        lambda x: map_history(x, note_map, default_note, max_history)
    )

    notes = add_note_features(notes)
    training_features = join_training_features(interactions, notes, users)
    user_history = build_user_history(interactions, max_history=max_history)

    processed.mkdir(parents=True, exist_ok=True)
    notes.to_parquet(processed / "notes.parquet", index=False)
    users.to_parquet(processed / "users.parquet", index=False)
    interactions.to_parquet(processed / "interactions.parquet", index=False)
    training_features.to_parquet(processed / "training_features.parquet", index=False)
    user_history.to_parquet(processed / "user_history.parquet", index=False)
    save_mapping(mappings_dir / "user_id_map.parquet", "raw_user_idx", "user_id", user_map)
    save_mapping(mappings_dir / "note_id_map.parquet", "raw_note_idx", "note_id", note_map)
    save_json(
        processed / "prep_stats.json",
        {
            "interactions": int(len(interactions)),
            "notes": int(len(notes)),
            "users": int(len(users)),
            "mapped_users": int(len(user_map)),
            "mapped_notes": int(len(note_map)),
            "default_user_id": default_user,
            "default_note_id": default_note,
        },
    )
    print(f"saved processed artifacts to {processed}")


if __name__ == "__main__":
    main()
