from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.io import project_path, setup_logging, write_json
from src.data.schema import inspect_parquet_file


def main() -> None:
    setup_logging()
    dataset_dir = project_path("dataset")
    output_dir = project_path("outputs")
    sample_dir = output_dir / "sample_rows"
    sample_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in sorted(dataset_dir.glob("*/*.parquet")):
        info, sample = inspect_parquet_file(path)
        summaries.append(info)
        subset_name = path.parent.name
        sample_path = sample_dir / f"{subset_name}_{path.stem}.json"
        sample.to_json(sample_path, orient="records", force_ascii=False, indent=2, default_handler=str)
        print(f"{path}: rows={info['rows']} columns={len(info['columns'])}")
        print("  columns:", ", ".join(info["columns"]))
        if info["nested_or_list_columns"]:
            print("  nested/list:", ", ".join(info["nested_or_list_columns"].keys()))

    write_json({"files": summaries}, output_dir / "schema_summary.json")
    print(f"\nSaved schema summary to {output_dir / 'schema_summary.json'}")
    print(f"Saved sample rows to {sample_dir}")


if __name__ == "__main__":
    main()
