from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.io import ensure_dirs, read_json, read_parquet_dir, save_parquet, write_json
from src.data.parser import LABEL_COLUMNS, parse_notes_frame, parse_recommendation_frame, parse_user_frame

LOGGER = logging.getLogger(__name__)


def load_recommendation_interactions(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = config["paths"]["dataset_dir"]
    history_max_len = int(config["data"]["history_max_len"])
    train_raw = read_parquet_dir(dataset_dir, "recommendation_train")
    test_raw = read_parquet_dir(dataset_dir, "recommendation_test")

    debug = config.get("debug", {})
    if debug.get("enabled", True):
        max_users = int(debug.get("max_users", 5000))
        users = pd.concat([train_raw["user_idx"], test_raw["user_idx"]]).drop_duplicates().head(max_users)
        train_raw = train_raw[train_raw["user_idx"].isin(users)]
        test_raw = test_raw[test_raw["user_idx"].isin(users)]

    train = parse_recommendation_frame(train_raw, "train", history_max_len)
    test = parse_recommendation_frame(test_raw, "test", history_max_len)

    if debug.get("enabled", True):
        max_interactions = int(debug.get("max_interactions", 200000))
        train = train.head(max_interactions)
        test = test.head(max(20000, max_interactions // 5))

    return train, test


def load_notes_for_interactions(config: dict[str, Any], interactions: pd.DataFrame) -> pd.DataFrame:
    dataset_dir = config["paths"]["dataset_dir"]
    note_columns = [
        "note_idx",
        "note_title",
        "note_content",
        "note_type",
        "taxonomy1_id",
        "taxonomy2_id",
        "taxonomy3_id",
        "imp_num",
        "click_num",
        "like_num",
        "collect_num",
        "comment_num",
        "share_num",
        "view_time",
    ]
    LOGGER.info("Loading note metadata columns from local parquet files")
    raw = read_parquet_dir(dataset_dir, "notes", columns=note_columns)
    notes = parse_notes_frame(raw)
    notes["popularity_score"] = compute_popularity(notes)

    debug = config.get("debug", {})
    if debug.get("enabled", True):
        max_notes = int(debug.get("max_notes", 50000))
        important = set(interactions["note_id"].dropna().astype(int).tolist())
        for history in interactions["history_note_ids"].head(50000):
            important.update(int(x) for x in history)
        top_notes = notes.sort_values("popularity_score", ascending=False).head(max_notes)["note_id"].tolist()
        keep = set(top_notes) | important
        notes = notes[notes["note_id"].isin(keep)].head(max_notes + len(important))
    return notes.reset_index(drop=True)


def compute_popularity(notes: pd.DataFrame) -> pd.Series:
    imp = np.log1p(notes.get("imp_num", 0.0).astype(float))
    click = np.log1p(notes.get("click_num", 0.0).astype(float))
    engage = np.log1p(
        notes.get("like_num", 0.0).astype(float)
        + notes.get("collect_num", 0.0).astype(float)
        + notes.get("comment_num", 0.0).astype(float)
        + notes.get("share_num", 0.0).astype(float)
    )
    ctr = (notes.get("click_num", 0.0).astype(float) + 1.0) / (notes.get("imp_num", 0.0).astype(float) + 10.0)
    return 0.45 * click + 0.35 * engage + 0.15 * imp + 0.05 * ctr


def build_mappings(train: pd.DataFrame, test: pd.DataFrame, notes: pd.DataFrame) -> dict[str, dict[str, int]]:
    users = sorted(set(train["user_id"].astype(int)) | set(test["user_id"].astype(int)))
    note_ids = set(notes["note_id"].astype(int))
    for df in [train, test]:
        note_ids.update(df["note_id"].astype(int).tolist())
        for history in df["history_note_ids"]:
            note_ids.update(int(x) for x in history)
    notes_sorted = sorted(note_ids)
    return {
        "user2idx": {str(user_id): idx + 1 for idx, user_id in enumerate(users)},
        "note2idx": {str(note_id): idx + 1 for idx, note_id in enumerate(notes_sorted)},
        "idx2user": {str(idx + 1): user_id for idx, user_id in enumerate(users)},
        "idx2note": {str(idx + 1): note_id for idx, note_id in enumerate(notes_sorted)},
    }


def add_index_columns(df: pd.DataFrame, mappings: dict[str, dict[str, int]], history_max_len: int) -> pd.DataFrame:
    out = df.copy()
    user2idx = mappings["user2idx"]
    note2idx = mappings["note2idx"]
    out["user_idx_internal"] = out["user_id"].astype(str).map(user2idx).fillna(0).astype(int)
    out["note_idx_internal"] = out["note_id"].astype(str).map(note2idx).fillna(0).astype(int)

    def encode_history(history: list[int]) -> list[int]:
        encoded = [note2idx.get(str(int(x)), 0) for x in history][-history_max_len:]
        return [0] * (history_max_len - len(encoded)) + encoded

    out["history_idx_internal"] = out["history_note_ids"].map(encode_history)
    valid_page_time = out["page_time"].clip(lower=0, upper=float(300.0))
    denom = valid_page_time.quantile(0.95) or 1.0
    out["page_time_norm"] = (valid_page_time / denom).clip(0, 1)
    return out


def build_user_profiles(train: pd.DataFrame, notes: pd.DataFrame) -> pd.DataFrame:
    note_meta = notes.set_index("note_id")[["title", "category", "topic"]].to_dict("index")
    rows = []
    for user_id, grp in tqdm(train.groupby("user_id"), desc="user profiles"):
        histories = []
        for history in grp["history_note_ids"].head(3):
            histories.extend(int(x) for x in history)
        positives = grp.loc[grp["click"] > 0, "note_id"].astype(int).tolist()
        history = list(dict.fromkeys((histories + positives)[-50:]))
        categories = [note_meta.get(n, {}).get("category", "") for n in history]
        top_categories = pd.Series([c for c in categories if c]).value_counts().head(5).index.tolist()
        rows.append(
            {
                "user_id": int(user_id),
                "history_note_ids": history[-50:],
                "top_categories": top_categories,
                "positive_count": int((grp["click"] > 0).sum()),
                "impression_count": int(len(grp)),
            }
        )
    return pd.DataFrame(rows)


def prepare_data(config: dict[str, Any]) -> None:
    processed = Path(config["paths"]["processed_dir"])
    ensure_dirs(
        processed / "samples",
        processed / "features",
        processed / "mappings",
        processed / "schema",
        processed / "recalls",
    )
    train, test = load_recommendation_interactions(config)
    all_interactions = pd.concat([train, test], ignore_index=True)
    notes = load_notes_for_interactions(config, all_interactions)
    users = parse_user_frame(read_parquet_dir(config["paths"]["dataset_dir"], "user_feat"))
    mappings = build_mappings(train, test, notes)
    history_max_len = int(config["data"]["history_max_len"])
    train = add_index_columns(train, mappings, history_max_len)
    test = add_index_columns(test, mappings, history_max_len)
    user_profiles = build_user_profiles(train, notes)

    save_parquet(train, processed / "samples" / "train_interactions.parquet")
    save_parquet(test, processed / "samples" / "test_interactions.parquet")
    save_parquet(notes, processed / "features" / "note_features.parquet")
    save_parquet(users, processed / "features" / "user_features.parquet")
    save_parquet(user_profiles, processed / "features" / "user_profiles.parquet")
    write_json(mappings, processed / "mappings" / "id_mappings.json")
    write_json(
        {
            "train_rows": len(train),
            "test_rows": len(test),
            "notes": len(notes),
            "users": len(users),
            "labels": [label for label in LABEL_COLUMNS if label in train.columns],
        },
        processed / "schema" / "processed_summary.json",
    )
    LOGGER.info("Prepared data: train=%s test=%s notes=%s", len(train), len(test), len(notes))
