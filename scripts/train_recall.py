from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.recall.trainer import train_recall_model
from redbookrec.utils.config import load_config
from redbookrec.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["train"]["max_train_samples"] = args.max_train_samples
    set_seed(int(cfg.get("seed", 2025)))
    result = train_recall_model(cfg, smoke_test=args.smoke_test)
    print(result)


if __name__ == "__main__":
    main()
