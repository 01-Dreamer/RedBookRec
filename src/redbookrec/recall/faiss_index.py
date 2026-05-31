from __future__ import annotations

import numpy as np


def search_topk(query_emb: np.ndarray, item_emb: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import faiss

        index = faiss.IndexFlatIP(item_emb.shape[1])
        index.add(item_emb.astype("float32"))
        scores, indices = index.search(query_emb.astype("float32"), int(top_k))
        return scores, indices
    except Exception:
        scores = query_emb @ item_emb.T
        k = min(int(top_k), scores.shape[1])
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part_scores, axis=1)
        return np.take_along_axis(part_scores, order, axis=1), np.take_along_axis(idx, order, axis=1)
