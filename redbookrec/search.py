from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text: str) -> list[str]:
    text = (text or "").strip().lower()
    if not text:
        return []
    chars = [c for c in text if not c.isspace()]
    unigrams = chars
    bigrams = ["".join(chars[i : i + 2]) for i in range(max(0, len(chars) - 1))]
    return unigrams + bigrams


def build_bm25_index(notes: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    docs = [tokenize(text) for text in notes["search_text"].fillna("").astype(str)]
    doc_freq: dict[str, int] = {}
    term_freqs: list[dict[str, int]] = []
    doc_lens = []
    for tokens in docs:
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        for token in tf:
            doc_freq[token] = doc_freq.get(token, 0) + 1
        term_freqs.append(tf)
        doc_lens.append(len(tokens))
    n_docs = max(1, len(docs))
    idf = {term: float(np.log(1 + (n_docs - df + 0.5) / (df + 0.5))) for term, df in doc_freq.items()}
    return {
        "type": "bm25",
        "term_freqs": term_freqs,
        "doc_lens": np.asarray(doc_lens, dtype=np.float32),
        "avgdl": float(np.mean(doc_lens)) if doc_lens else 0.0,
        "idf": idf,
        "note_ids": notes["note_id"].astype(int).to_numpy(),
        "raw_note_idxs": notes["raw_note_idx"].astype(int).to_numpy(),
    }


def bm25_scores(index: dict[str, Any], query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokens = tokenize(query)
    scores = np.zeros(len(index["term_freqs"]), dtype=np.float32)
    if not tokens or len(scores) == 0:
        return scores
    avgdl = max(float(index["avgdl"]), 1.0)
    doc_lens = index["doc_lens"]
    query_terms = set(tokens)
    for term in query_terms:
        idf = index["idf"].get(term)
        if idf is None:
            continue
        for i, tf in enumerate(index["term_freqs"]):
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            denom = freq + k1 * (1 - b + b * doc_lens[i] / avgdl)
            scores[i] += idf * (freq * (k1 + 1)) / denom
    return scores


def build_tfidf_index(notes: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    search_cfg = cfg.get("search", {})
    vectorizer = TfidfVectorizer(
        analyzer=search_cfg.get("analyzer", "char"),
        ngram_range=tuple(search_cfg.get("ngram_range", [1, 2])),
        min_df=search_cfg.get("min_df", 2),
        max_features=search_cfg.get("max_features", 200000),
    )
    matrix = vectorizer.fit_transform(notes["search_text"].fillna("").astype(str))
    return {
        "type": "tfidf",
        "vectorizer": vectorizer,
        "matrix": matrix,
        "note_ids": notes["note_id"].astype(int).to_numpy(),
        "raw_note_idxs": notes["raw_note_idx"].astype(int).to_numpy(),
    }


def build_search_index(notes: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "hybrid_bm25_tfidf",
        "bm25": build_bm25_index(notes, cfg),
        "tfidf": build_tfidf_index(notes, cfg),
    }


def save_index(index: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(index, path)


def load_index(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)


def search(index: dict[str, Any], query: str, top_k: int) -> pd.DataFrame:
    index_type = index.get("type", "tfidf")
    if index_type == "hybrid_bm25_tfidf":
        bm25 = index["bm25"]
        tfidf = index["tfidf"]
        bm25_score = bm25_scores(bm25, query)
        q = tfidf["vectorizer"].transform([query or ""])
        tfidf_score = (tfidf["matrix"] @ q.T).toarray().ravel().astype(np.float32)
        if bm25_score.max() > bm25_score.min():
            bm25_norm = (bm25_score - bm25_score.min()) / (bm25_score.max() - bm25_score.min())
        else:
            bm25_norm = bm25_score
        if tfidf_score.max() > tfidf_score.min():
            tfidf_norm = (tfidf_score - tfidf_score.min()) / (tfidf_score.max() - tfidf_score.min())
        else:
            tfidf_norm = tfidf_score
        scores = 0.6 * bm25_norm + 0.4 * tfidf_norm
        note_ids = bm25["note_ids"]
        raw_note_idxs = bm25["raw_note_idxs"]
        source = "search_bm25_tfidf"
    elif index_type == "bm25":
        scores = bm25_scores(index, query)
        note_ids = index["note_ids"]
        raw_note_idxs = index["raw_note_idxs"]
        source = "search_bm25"
    else:
        q = index["vectorizer"].transform([query or ""])
        scores = (index["matrix"] @ q.T).toarray().ravel()
        note_ids = index["note_ids"]
        raw_note_idxs = index["raw_note_idxs"]
        source = "search_tfidf"
    if scores.size == 0:
        return pd.DataFrame(columns=["note_id", "raw_note_idx", "score", "recall_source"])
    top_k = min(int(top_k), scores.size)
    idx = scores.argsort()[-top_k:][::-1]
    return pd.DataFrame(
        {
            "note_id": note_ids[idx],
            "raw_note_idx": raw_note_idxs[idx],
            "score": scores[idx],
            "recall_source": source,
        }
    )
