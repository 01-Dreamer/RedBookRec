from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.search_recall.merger import merge_recall
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = merge_recall(cfg)
    print(f"saved_hybrid_rows={len(out)} path={cfg['infer']['hybrid_output_path']}")


if __name__ == "__main__":
    main()
