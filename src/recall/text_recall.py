from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm


class TextRecall:
    def __init__(self, top_k: int = 100, max_features: int = 50000) -> None:
        self.top_k = top_k
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=max_features)
        self.note_ids: np.ndarray | None = None
        self.matrix: sparse.csr_matrix | None = None

    def fit(self, notes: pd.DataFrame) -> "TextRecall":
        text = notes["text"].fillna("").astype(str)
        self.matrix = self.vectorizer.fit_transform(text)
        self.note_ids = notes["note_id"].astype(int).to_numpy()
        return self

    def recommend_for_queries(self, user_queries: pd.DataFrame, histories: dict[int, list[int]], top_k: int | None = None) -> pd.DataFrame:
        if self.matrix is None or self.note_ids is None:
            raise RuntimeError("TextRecall must be fitted before recommending.")
        top_k = top_k or self.top_k
        rows = []
        for row in tqdm(user_queries.itertuples(index=False), total=len(user_queries), desc="text recall"):
            user_id = int(row.user_id)
            query = str(row.query or "")
            if not query:
                continue
            q_vec = self.vectorizer.transform([query])
            scores = linear_kernel(q_vec, self.matrix).ravel()
            if len(scores) == 0:
                continue
            seen = set(histories.get(user_id, []))
            candidate_idx = np.argpartition(-scores, kth=min(top_k * 3, len(scores) - 1))[: top_k * 3]
            ranked = candidate_idx[np.argsort(-scores[candidate_idx])]
            count = 0
            for idx in ranked:
                note_id = int(self.note_ids[idx])
                if note_id in seen:
                    continue
                rows.append(
                    {
                        "user_id": user_id,
                        "note_id": note_id,
                        "recall_score": float(scores[idx]),
                        "recall_source": "text",
                    }
                )
                count += 1
                if count >= top_k:
                    break
        return pd.DataFrame(rows)
