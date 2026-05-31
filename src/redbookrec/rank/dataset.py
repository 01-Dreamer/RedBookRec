from __future__ import annotations

import pandas as pd


def read_rank_candidates(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
