from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from redbookrec.evaluation.metrics import evaluate_stage
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--stage", choices=["recall", "search_recall", "hybrid_recall", "prerank", "rank", "rerank"], default="recall")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    metrics = evaluate_stage(cfg, args.stage)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
