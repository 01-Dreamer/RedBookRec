from __future__ import annotations

import numpy as np


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.ones_like(scores, dtype="float32")
    return ((scores - lo) / (hi - lo)).astype("float32")
