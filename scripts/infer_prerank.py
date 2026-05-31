from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.prerank.inference import infer_prerank
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dcn_lite.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = infer_prerank(cfg)
    print(f"saved_prerank_rows={len(out)} path={cfg['infer']['output_path']}")


if __name__ == "__main__":
    main()
