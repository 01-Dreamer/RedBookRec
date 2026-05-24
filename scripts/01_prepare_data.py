from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cli import add_config_arguments, load_config_with_overrides
from src.data.io import setup_logging, seed_everything
from src.data.preprocess import prepare_data


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args()
    config = load_config_with_overrides(args, ["configs/base.yaml"])
    seed_everything(int(config["seed"]))
    prepare_data(config)


if __name__ == "__main__":
    main()
