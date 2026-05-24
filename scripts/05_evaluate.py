from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.cli import add_config_arguments, load_config_with_overrides
from src.data.io import setup_logging
from src.eval.evaluator import evaluate


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    add_config_arguments(parser)
    args = parser.parse_args()
    config = load_config_with_overrides(args, ["configs/base.yaml", "configs/ranker.yaml"])
    results = evaluate(config)
    for model, metrics in results.items():
        compact = ", ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:6])
        print(f"{model}: {compact}")


if __name__ == "__main__":
    main()
