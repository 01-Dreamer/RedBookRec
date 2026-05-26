from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, save_run_config
from redbookrec.search import build_search_index, build_tfidf_index, save_index


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=None)
    args = parser.parse_args()
    cfg = build_config(args, "search")
    ensure_dirs(cfg)
    save_run_config(cfg, "search")

    notes_path = Path(cfg["paths"]["processed_dir"]) / "notes.parquet"
    if not notes_path.exists():
        raise SystemExit("Run scripts/prepare_data.py first.")
    notes = pd.read_parquet(notes_path)
    max_notes = cfg.get("limits", {}).get("max_notes")
    if max_notes is not None:
        notes = notes.head(int(max_notes))
    index = build_search_index(notes, cfg)
    out = Path(cfg["paths"]["indexes_dir"]) / "search_index.joblib"
    save_index(index, out)
    # Keep a TF-IDF-only artifact for quick ablations and backward compatibility.
    tfidf_out = Path(cfg["paths"]["indexes_dir"]) / "search_tfidf.joblib"
    save_index(build_tfidf_index(notes, cfg), tfidf_out)
    print(f"saved_search_index={out} saved_tfidf_index={tfidf_out} notes={len(notes)}")


if __name__ == "__main__":
    main()
