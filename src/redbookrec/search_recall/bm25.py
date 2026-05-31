from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfSearchIndex:
    def __init__(self, max_features: int = 50000):
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), max_features=max_features)
        self.matrix = None

    def fit(self, texts: pd.Series) -> "TfidfSearchIndex":
        self.matrix = self.vectorizer.fit_transform(texts.fillna("").astype(str))
        return self

    def search(self, query: str, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.matrix is None:
            raise RuntimeError("index is not fitted")
        q = self.vectorizer.transform([query or ""])
        scores = (q @ self.matrix.T).toarray()[0]
        if scores.size == 0:
            return np.array([], dtype="float32"), np.array([], dtype="int64")
        k = min(int(top_k), scores.size)
        idx = np.argpartition(-scores, kth=k - 1)[:k]
        order = np.argsort(-scores[idx])
        idx = idx[order]
        return scores[idx].astype("float32"), idx.astype("int64")
