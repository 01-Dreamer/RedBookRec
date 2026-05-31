from __future__ import annotations

from typing import Iterable


def build_id_map(values: Iterable[int], existing: dict[str, int] | None = None) -> dict[str, int]:
    mapping = dict(existing or {})
    next_id = max(mapping.values(), default=0) + 1
    for value in values:
        try:
            raw = int(value)
        except Exception:
            continue
        if raw < 0:
            continue
        key = str(raw)
        if key not in mapping:
            mapping[key] = next_id
            next_id += 1
    return mapping


def map_raw(mapping: dict[str, int], value: int | float | str | None, default: int = 0) -> int:
    try:
        return int(mapping.get(str(int(value)), default))
    except Exception:
        return int(default)


def map_sequence(mapping: dict[str, int], values: list, max_len: int, default: int = 0) -> list[int]:
    mapped = [map_raw(mapping, x, default) for x in values]
    mapped = [x for x in mapped if x > 0]
    return mapped[-int(max_len) :]
