from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from redbookrec.data.load_qilin import read_dataset_split
from redbookrec.data.preprocess_rec import expand_recommendation_requests
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--max-requests", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cols = ["recent_clicked_note_idxs", "request_idx", "session_idx", "user_idx", "query", "rec_result_details_with_idx"]
    req = read_dataset_split(cfg["data"]["dataset_dir"], "recommendation_train", columns=cols, max_rows=args.max_requests)
    exp = expand_recommendation_requests(req)
    print(f"requests={len(req)}")
    print(f"expanded_candidates={len(exp)}")
    if not exp.empty:
        print("candidates_per_request:")
        print(exp.groupby("request_idx").size().describe().to_string())
        print(f"click_rate={exp['label_click'].mean():.6f}")
        print("label_counts:")
        print(exp["label_click"].value_counts().to_string())
        note_cols = ["note_idx"]
        notes = read_dataset_split(cfg["data"]["dataset_dir"], "notes", columns=note_cols, max_rows=None)
        coverage = exp["note_idx"].isin(set(notes["note_idx"].astype(int))).mean()
        print(f"candidate_note_coverage_in_notes={coverage:.6f}")


if __name__ == "__main__":
    main()
