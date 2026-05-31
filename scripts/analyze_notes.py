from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from redbookrec.data.load_qilin import read_dataset_split
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--max-notes", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cols = ["note_idx", "note_title", "note_content", "note_type", "taxonomy1_id", "taxonomy2_id", "taxonomy3_id", "content_length"]
    df = read_dataset_split(cfg["data"]["dataset_dir"], "notes", columns=cols, max_rows=args.max_notes)
    print(f"notes_rows={len(df)}")
    print(f"columns={list(df.columns)}")
    print("missing_rate:")
    print(df.isna().mean().sort_values(ascending=False).to_string())
    print("note_type_distribution:")
    print(df["note_type"].value_counts(dropna=False).head(20).to_string())
    print("taxonomy_missing:")
    for col in ["taxonomy1_id", "taxonomy2_id", "taxonomy3_id"]:
        s = df[col].fillna("nan").astype(str)
        print(f"{col}: {(s.isin(['', 'nan', 'None'])).mean():.4f}")
    text_len = df["note_title"].fillna("").astype(str).str.len() + df["note_content"].fillna("").astype(str).str.len()
    print("text_length:")
    print(text_len.describe().to_string())


if __name__ == "__main__":
    main()
