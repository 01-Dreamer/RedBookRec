from __future__ import annotations

import numpy as np
import pandas as pd


def stable_bucket(value: str, buckets: int = 10000) -> int:
    if not value:
        return 0
    total = 0
    for ch in str(value):
        total = (total * 131 + ord(ch)) % buckets
    return total + 1


def add_note_features(notes: pd.DataFrame) -> pd.DataFrame:
    notes = notes.copy()
    for col in [
        "imp_num",
        "click_num",
        "like_num",
        "collect_num",
        "comment_num",
        "share_num",
        "content_length",
        "image_num",
        "video_duration",
    ]:
        if col not in notes:
            notes[col] = 0.0
        notes[col] = notes[col].fillna(0.0).astype(float)
    notes["ctr"] = (notes["click_num"] + 1.0) / (notes["imp_num"] + 10.0)
    notes["engagement_score"] = np.log1p(
        notes["click_num"]
        + 2 * notes["like_num"]
        + 3 * notes["collect_num"]
        + 2 * notes["comment_num"]
        + notes["share_num"]
    )
    notes["popularity_score"] = notes["engagement_score"] + np.log1p(notes["imp_num"]) * notes["ctr"]
    text_cols = ["note_title", "note_content", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]
    for col in text_cols:
        if col not in notes:
            notes[col] = ""
        notes[col] = notes[col].fillna("").astype(str)
    notes["search_text"] = notes[text_cols].agg(" ".join, axis=1)
    notes["title_len"] = notes["note_title"].str.len().fillna(0).astype(float)
    notes["taxonomy1_bucket"] = notes["taxonomy1_id"].map(stable_bucket).astype(float)
    notes["taxonomy2_bucket"] = notes["taxonomy2_id"].map(stable_bucket).astype(float)
    notes["taxonomy3_bucket"] = notes["taxonomy3_id"].map(stable_bucket).astype(float)
    return notes


def join_training_features(interactions: pd.DataFrame, notes: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    note_cols = [
        "note_id",
        "raw_note_idx",
        "note_type",
        "content_length",
        "image_num",
        "video_duration",
        "ctr",
        "popularity_score",
        "engagement_score",
        "note_title",
        "note_content",
        "taxonomy1_id",
        "taxonomy2_id",
        "taxonomy3_id",
        "title_len",
        "taxonomy1_bucket",
        "taxonomy2_bucket",
        "taxonomy3_bucket",
    ]
    note_cols = [c for c in note_cols if c in notes.columns]
    out = interactions.merge(notes[note_cols], on=["note_id", "raw_note_idx"], how="left")
    user_cols = ["user_id", "raw_user_idx", "fans_num", "follows_num"]
    user_cols.extend([c for c in users.columns if c.startswith("dense_feat")])
    user_cols = [c for c in user_cols if c in users.columns]
    out = out.merge(users[user_cols], on=["user_id", "raw_user_idx"], how="left")
    out["query_len"] = out["query"].fillna("").astype(str).str.len().astype(float)
    out["query_title_overlap"] = [
        char_overlap(query, title) for query, title in zip(out["query"].fillna(""), out.get("note_title", "").fillna(""))
    ]
    out["query_content_overlap"] = [
        char_overlap(query, content) for query, content in zip(out["query"].fillna(""), out.get("note_content", "").fillna(""))
    ]
    return out


def char_overlap(a: str, b: str) -> float:
    a_set = {c for c in str(a) if not c.isspace()}
    b_set = {c for c in str(b) if not c.isspace()}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set))


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "position",
        "page_time",
        "note_type",
        "content_length",
        "image_num",
        "video_duration",
        "ctr",
        "popularity_score",
        "engagement_score",
        "title_len",
        "taxonomy1_bucket",
        "taxonomy2_bucket",
        "taxonomy3_bucket",
        "query_len",
        "query_title_overlap",
        "query_content_overlap",
        "fans_num",
        "follows_num",
    ]
    preferred.extend([c for c in df.columns if c.startswith("dense_feat")])
    return [c for c in preferred if c in df.columns]
