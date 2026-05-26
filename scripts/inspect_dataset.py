from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, save_run_config, write_json


def inspect_file(path: Path, sample_rows: int = 3) -> dict:
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    df = pd.read_parquet(path).head(sample_rows)
    sample = df.astype(str).to_dict(orient="records")
    return {
        "path": str(path),
        "rows": meta.num_rows,
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "sample": sample,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = build_config(args, "inspect")
    ensure_dirs(cfg)
    save_run_config(cfg, "inspect")

    dataset_dir = Path(cfg["paths"]["dataset_dir"])
    files = sorted(dataset_dir.glob("**/*.parquet"))
    summary = [inspect_file(path) for path in files]
    out = Path(cfg["paths"]["processed_dir"]) / "schema_summary.json"
    write_json(out, summary)
    print(f"inspected_files={len(summary)} output={out}")
    for item in summary:
        print(f"{item['path']} rows={item['rows']} cols={len(item['columns'])}")


if __name__ == "__main__":
    main()
