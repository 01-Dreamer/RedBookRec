from __future__ import annotations

import argparse

from redbookrec.config import build_config
from redbookrec.recommend import recommend


def main() -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        raise SystemExit("streamlit is not installed. Install it or use scripts/recommend_user.py.")

    args = argparse.Namespace(
        config=None,
        debug=True,
        full=False,
        device=None,
        batch_size=None,
        epochs=None,
        num_workers=None,
        max_users=None,
        max_notes=None,
        max_interactions=None,
        mixed_precision=False,
        run_id=None,
        top_k=None,
    )
    cfg = build_config(args, "recommend")
    st.set_page_config(page_title="RedBookRec", layout="wide")
    st.title("RedBookRec")
    user_id = st.number_input("user_id", min_value=0, value=0, step=1)
    query = st.text_input("query", "")
    top_k = st.slider("top_k", 5, 50, 20)
    if st.button("Recommend"):
        recs = recommend(cfg, user_id=int(user_id), query=query, top_k=int(top_k))
        show_cols = [c for c in ["rerank_position", "note_title", "score", "recall_source", "taxonomy1_id"] if c in recs.columns]
        st.dataframe(recs[show_cols], use_container_width=True)


if __name__ == "__main__":
    main()
