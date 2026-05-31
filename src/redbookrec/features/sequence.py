from __future__ import annotations


def pad_sequence(values: list[int], max_len: int, pad: int = 0) -> list[int]:
    values = [int(v) for v in values if int(v) > 0][-int(max_len) :]
    return [pad] * (int(max_len) - len(values)) + values
