from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.rank.trainer import train_rank_placeholder
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sim.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(train_rank_placeholder(cfg))


if __name__ == "__main__":
    main()
