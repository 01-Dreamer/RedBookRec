from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.prerank.train import train_prerank
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dcn.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.max_train_samples is not None:
        cfg["train"]["max_train_samples"] = args.max_train_samples
    print(train_prerank(cfg, smoke_test=args.smoke_test))


if __name__ == "__main__":
    main()
