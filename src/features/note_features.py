from __future__ import annotations

import pandas as pd


def note_text(row: pd.Series) -> str:
    return f"{row.get('title', '')} {row.get('content', '')}".strip()
