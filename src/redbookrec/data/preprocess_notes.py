from __future__ import annotations

from pathlib import Path

import pandas as pd

from redbookrec.data.id_mapping import build_id_map
from redbookrec.data.load_qilin import read_dataset_split
from redbookrec.utils.io import write_json


NOTE_COLUMNS = [
    "note_idx",
    "note_title",
    "note_content",
    "note_type",
    "content_length",
    "commercial_flag",
    "taxonomy1_id",
    "taxonomy2_id",
    "taxonomy3_id",
    "image_num",
    "video_duration",
]


def clean_taxonomy(series: pd.Series) -> pd.Series:
    return series.fillna("UNK").astype(str).replace({"": "UNK", "nan": "UNK", "None": "UNK"})


def prepare_notes(cfg: dict, max_notes: int | None = None) -> pd.DataFrame:
    dataset_dir = cfg["data"]["dataset_dir"]
    output_path = Path(cfg["data"]["note_text_path"])
    map_path = Path(cfg["data"]["note_id_map_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = read_dataset_split(dataset_dir, "notes", columns=NOTE_COLUMNS, max_rows=max_notes)
    df["note_idx"] = pd.to_numeric(df["note_idx"], errors="coerce").fillna(-1).astype("int64")
    df = df[df["note_idx"] >= 0].copy()
    df["note_title"] = df["note_title"].fillna("").astype(str)
    df["note_content"] = df["note_content"].fillna("").astype(str)
    for col in ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]:
        df[col] = clean_taxonomy(df[col])
    for col in ["note_type", "content_length", "commercial_flag", "image_num", "video_duration"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["note_text"] = df["note_title"] + " [SEP] " + df["note_content"]

    raw_to_model = build_id_map(df["note_idx"].tolist())
    df["note_id"] = df["note_idx"].map(lambda x: raw_to_model.get(str(int(x)), 0)).astype("int64")
    keep_cols = [
        "note_idx",
        "note_id",
        "note_title",
        "note_content",
        "note_text",
        "note_type",
        "content_length",
        "commercial_flag",
        "taxonomy1_id",
        "taxonomy2_id",
        "taxonomy3_id",
        "image_num",
        "video_duration",
    ]
    df[keep_cols].to_parquet(output_path, index=False)
    write_json(
        map_path,
        {
            "raw_to_model": raw_to_model,
            "model_to_raw": {str(v): int(k) for k, v in raw_to_model.items()},
            "padding_id": 0,
            "num_notes": len(raw_to_model) + 1,
        },
    )
    return df[keep_cols]
