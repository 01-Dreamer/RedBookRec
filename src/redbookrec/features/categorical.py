from __future__ import annotations


def bucketize_string(value: str, buckets: int = 10000) -> int:
    if value is None or str(value) in {"", "nan", "None"}:
        return 0
    return abs(hash(str(value))) % max(1, buckets - 1) + 1
