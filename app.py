from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.io import load_config
from src.rerank.diversity import diversity_rerank
from src.service.recommender import build_candidate_samples, score_with_ranker


def main() -> None:
    st.set_page_config(page_title="RedBookRec", layout="wide")
    st.title("RedBookRec Offline Feed Recommender")
    config = load_config("configs/base.yaml", "configs/ranker.yaml")
    processed = Path(config["paths"]["processed_dir"])
    profiles = pd.read_parquet(processed / "features" / "user_profiles.parquet")
    notes = pd.read_parquet(processed / "features" / "note_features.parquet")
    recalls = pd.read_parquet(processed / "recalls" / "merged_recall.parquet")

    user_id = st.sidebar.selectbox("User", sorted(recalls["user_id"].unique().tolist()))
    top_k = st.sidebar.slider("Top K", 5, 50, 20)
    profile = profiles[profiles["user_id"] == user_id].iloc[0]
    candidates = recalls[recalls["user_id"] == user_id]
    ranked = score_with_ranker(build_candidate_samples(candidates, config), config)
    final = diversity_rerank(ranked, notes, profile["history_note_ids"], top_k=top_k)

    st.subheader(f"User {user_id}")
    st.write({"top_categories": profile["top_categories"], "positive_count": int(profile["positive_count"])})
    st.subheader("Recommendations")
    st.dataframe(final[["note_id", "title", "final_score", "recall_source", "rerank_reason"]], use_container_width=True)


if __name__ == "__main__":
    main()
