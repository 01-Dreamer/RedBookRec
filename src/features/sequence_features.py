from __future__ import annotations


def pad_or_truncate(sequence: list[int], max_len: int, pad_value: int = 0) -> list[int]:
    seq = list(sequence)[-max_len:]
    return [pad_value] * (max_len - len(seq)) + seq
