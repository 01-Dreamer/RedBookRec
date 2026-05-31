from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.recall.inference import infer_recall
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--max-notes", type=int, default=None)
    parser.add_argument("--max-requests", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = infer_recall(cfg, max_notes=args.max_notes, max_requests=args.max_requests)
    print(f"saved_recall_rows={len(out)} path={cfg['infer']['dual_output_path']}")


if __name__ == "__main__":
    main()
