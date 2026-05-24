from __future__ import annotations

import pandas as pd


def summarize_user_profile(row: pd.Series) -> str:
    categories = row.get("top_categories", [])
    return f"positive_count={row.get('positive_count', 0)}, top_categories={categories}"
