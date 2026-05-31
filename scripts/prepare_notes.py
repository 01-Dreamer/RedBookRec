from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.data.preprocess_notes import prepare_notes
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--max-notes", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = prepare_notes(cfg, max_notes=args.max_notes)
    print(f"saved_notes={len(df)} path={cfg['data']['note_text_path']}")


if __name__ == "__main__":
    main()
