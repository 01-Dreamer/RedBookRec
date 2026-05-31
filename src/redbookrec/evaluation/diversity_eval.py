from __future__ import annotations


def unique_ratio(values: list) -> float:
    return len(set(values)) / max(1, len(values))
