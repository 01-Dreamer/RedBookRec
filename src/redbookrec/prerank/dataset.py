from __future__ import annotations

import pandas as pd

from redbookrec.data.preprocess_rec import expand_recommendation_requests


def build_prerank_train_frame(rec_df: pd.DataFrame, max_samples: int | None = None) -> pd.DataFrame:
    df = expand_recommendation_requests(rec_df)
    if max_samples:
        df = df.head(int(max_samples))
    return df
