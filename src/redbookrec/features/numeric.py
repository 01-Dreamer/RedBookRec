from __future__ import annotations

import math


def safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except Exception:
        return default
