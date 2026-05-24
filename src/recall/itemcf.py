from __future__ import annotations

from collections import Counter, defaultdict
from math import log, sqrt

import pandas as pd
from tqdm import tqdm


class ItemCFRecall:
    def __init__(self, top_k: int = 200, max_history_len: int = 30, max_neighbors_per_item: int = 80) -> None:
        self.top_k = top_k
        self.max_history_len = max_history_len
        self.max_neighbors_per_item = max_neighbors_per_item
        self.similar_items: dict[int, list[tuple[int, float]]] = {}

    def fit(self, train: pd.DataFrame) -> "ItemCFRecall":
        cooccur: dict[int, Counter] = defaultdict(Counter)
        item_freq: Counter = Counter()
        user_sequences: dict[int, list[int]] = defaultdict(list)
        for row in train.itertuples(index=False):
            seq = [int(x) for x in row.history_note_ids[-self.max_history_len :]]
            if float(row.click) > 0:
                seq.append(int(row.note_id))
            if seq:
                user_sequences[int(row.user_id)].extend(seq[-self.max_history_len :])

        for seq in tqdm(user_sequences.values(), desc="itemcf cooccur"):
            seq = list(dict.fromkeys(seq[-self.max_history_len :]))
            for item in seq:
                item_freq[item] += 1
            weight = 1.0 / log(len(seq) + 2.0)
            for i, item_i in enumerate(seq):
                for item_j in seq[i + 1 :]:
                    cooccur[item_i][item_j] += weight
                    cooccur[item_j][item_i] += weight

        similar: dict[int, list[tuple[int, float]]] = {}
        for item_i, neighbors in cooccur.items():
            scored = []
            for item_j, cij in neighbors.items():
                denom = sqrt(item_freq[item_i] * item_freq[item_j]) or 1.0
                scored.append((item_j, float(cij / denom)))
            similar[item_i] = sorted(scored, key=lambda x: x[1], reverse=True)[: self.max_neighbors_per_item]
        self.similar_items = similar
        return self

    def recommend(self, history: list[int], top_k: int | None = None) -> list[dict]:
        top_k = top_k or self.top_k
        seen = set(int(x) for x in history)
        scores: Counter = Counter()
        recent = [int(x) for x in history[-self.max_history_len :]]
        for age, item in enumerate(reversed(recent)):
            decay = 1.0 / (1.0 + 0.05 * age)
            for neighbor, sim in self.similar_items.get(item, []):
                if neighbor not in seen:
                    scores[neighbor] += sim * decay
        return [
            {"note_id": int(note_id), "recall_score": float(score), "recall_source": "itemcf"}
            for note_id, score in scores.most_common(top_k)
        ]
