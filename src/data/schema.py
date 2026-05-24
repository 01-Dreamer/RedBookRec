from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from src.data.io import json_default


def summarize_value(value: Any) -> dict[str, Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        first = value[0] if value else None
        return {
            "kind": "list",
            "length": len(value),
            "first_type": type(first).__name__ if first is not None else None,
            "first_value": json_default(first) if first is not None else None,
        }
    if isinstance(value, dict):
        return {"kind": "dict", "keys": list(value.keys())[:20]}
    return {"kind": type(value).__name__, "sample": json_default(value)}


def inspect_parquet_file(path: Path, sample_rows: int = 3) -> tuple[dict[str, Any], pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    rows = parquet.metadata.num_rows if parquet.metadata is not None else None
    if parquet.num_row_groups:
        table = parquet.read_row_group(0).slice(0, sample_rows)
    else:
        table = pq.read_table(path).slice(0, sample_rows)
    sample = table.to_pandas()
    nested_summary: dict[str, Any] = {}
    for col in sample.columns:
        non_null = sample[col].dropna()
        if len(non_null) == 0:
            continue
        summary = summarize_value(non_null.iloc[0])
        if summary["kind"] in {"list", "dict"}:
            nested_summary[col] = summary
    info = {
        "path": str(path),
        "rows": rows,
        "columns": list(sample.columns),
        "dtypes": {col: str(dtype) for col, dtype in sample.dtypes.items()},
        "arrow_schema": str(schema),
        "nested_or_list_columns": nested_summary,
    }
    return info, sample
