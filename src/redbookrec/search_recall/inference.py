from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from redbookrec.data.load_qilin import parse_nested
from redbookrec.data.preprocess_rec import clicked_set, read_recommendation
from redbookrec.search_recall.bm25 import TfidfSearchIndex


def run_search_recall(cfg: dict, max_notes: int | None = None, max_requests: int | None = None) -> pd.DataFrame:
    note_df = pd.read_parquet(cfg["data"]["note_text_path"])
    if max_notes is None:
        max_notes = cfg.get("infer", {}).get("max_notes")
    if max_notes:
        note_df = note_df.head(int(max_notes))
    index = TfidfSearchIndex().fit(note_df["note_text"])
    title_map = dict(zip(note_df["note_idx"].astype(int), note_df["note_title"].fillna("").astype(str)))

    if max_requests is None:
        max_requests = cfg.get("infer", {}).get("max_requests")
    req_df = read_recommendation(cfg["data"]["dataset_dir"], "recommendation_test", max_requests=max_requests)
    top_k = min(int(cfg["infer"].get("top_k", 1000)), len(note_df))
    raw_note_idxs = note_df["note_idx"].astype("int64").to_numpy()
    rows: list[dict] = []
    for req in tqdm(req_df.itertuples(index=False), total=len(req_df), desc="search_recall", leave=False):
        query = (getattr(req, "query", "") or "").strip()
        if not query:
            history = parse_nested(getattr(req, "recent_clicked_note_idxs", []))[-5:]
            query = " ".join(title_map.get(int(x), "") for x in history if str(x).lstrip("-").isdigit())
        scores, idx = index.search(query, top_k)
        clicked = clicked_set(parse_nested(getattr(req, "rec_result_details_with_idx", [])))
        for rank, pos in enumerate(idx, start=1):
            note_idx = int(raw_note_idxs[int(pos)])
            rows.append(
                {
                    "request_idx": int(getattr(req, "request_idx")),
                    "user_idx": int(getattr(req, "user_idx")),
                    "note_idx": note_idx,
                    "search_score": float(scores[rank - 1]),
                    "search_rank": rank,
                    "label_click": int(note_idx in clicked),
                    "source": "tfidf_search",
                }
            )
    out = pd.DataFrame(rows)
    path = Path(cfg["infer"]["search_output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
