from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.rerank.inference import run_dpp
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpp.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = run_dpp(cfg)
    print(f"saved_rerank_rows={len(out)} path={cfg['infer']['output_path']}")


if __name__ == "__main__":
    main()
