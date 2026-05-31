from __future__ import annotations

import ast
import glob
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
except Exception:
    pass


def parse_nested(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "as_py"):
        return parse_nested(value.as_py())
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                return parse_nested(parsed)
            except Exception:
                pass
        warnings.warn(f"failed to parse nested value: {text[:80]}", RuntimeWarning)
        return []
    try:
        if pd.isna(value):
            return []
    except Exception:
        pass
    return []


def read_parquet_files(paths: list[str | Path], columns: list[str] | None = None, max_rows: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = None if max_rows is None else int(max_rows)
    for path in paths:
        df = pd.read_parquet(path, columns=columns)
        if remaining is not None:
            df = df.head(max(0, remaining))
            remaining -= len(df)
        frames.append(df)
        if remaining is not None and remaining <= 0:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def split_paths(dataset_dir: str | Path, split: str) -> list[str]:
    pattern = str(Path(dataset_dir) / split / "*.parquet")
    return sorted(glob.glob(pattern))


def read_dataset_split(
    dataset_dir: str | Path,
    split: str,
    columns: list[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    paths = split_paths(dataset_dir, split)
    if not paths:
        raise FileNotFoundError(f"no parquet files found for split={split} under {dataset_dir}")
    return read_parquet_files(paths, columns=columns, max_rows=max_rows)


def normalize_detail(item: Any) -> dict[str, Any]:
    if hasattr(item, "as_py"):
        item = item.as_py()
    if isinstance(item, dict):
        return item
    if hasattr(item, "_asdict"):
        return item._asdict()
    try:
        return dict(item)
    except Exception:
        return {}
