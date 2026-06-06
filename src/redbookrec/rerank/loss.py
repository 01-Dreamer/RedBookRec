from __future__ import annotations

from typing import Any


def item_similarity(a: dict[str, Any], b: dict[str, Any]) -> float:
    if not a or not b:
        return 0.0
    score = 0.0
    if a.get("tax1") != "UNK" and a.get("tax1") == b.get("tax1"):
        score += 0.45
    if a.get("tax2") != "UNK" and a.get("tax2") == b.get("tax2"):
        score += 0.25
    if a.get("tax3") != "UNK" and a.get("tax3") == b.get("tax3"):
        score += 0.15
    if a.get("note_type") == b.get("note_type"):
        score += 0.10
    if a.get("length_bucket") == b.get("length_bucket"):
        score += 0.05
    return min(1.0, score)


def dpp_objective_score(relevance: float, diversity_penalty: float, lambda_diversity: float) -> float:
    return float(relevance) - float(lambda_diversity) * float(diversity_penalty)
