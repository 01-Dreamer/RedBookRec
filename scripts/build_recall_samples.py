from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.data.sample_builder import build_recall_samples
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--max-requests", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train, test = build_recall_samples(cfg, max_requests=args.max_requests)
    print(f"train_positive_samples={len(train)} path={cfg['data']['train_samples_path']}")
    print(f"test_positive_samples={len(test)} path={cfg['data']['test_samples_path']}")


if __name__ == "__main__":
    main()
