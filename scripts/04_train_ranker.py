from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cli import add_config_arguments, load_config_with_overrides
from src.data.io import setup_logging
from src.train.train_ranker import train_ranker


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser, include_training=True)
    args = parser.parse_args()
    config = load_config_with_overrides(args, ["configs/base.yaml", "configs/ranker.yaml"])
    checkpoint = train_ranker(config)
    print(f"Saved DIN ranker checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
