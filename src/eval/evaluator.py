from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.io import save_parquet, write_json
from src.eval.metrics import binary_metrics, ranking_metrics
from src.rerank.diversity import diversity_rerank
from src.service.recommender import build_candidate_samples, score_with_ranker


def positive_items(test: pd.DataFrame) -> dict[int, set[int]]:
    clicked = test[test["click"] > 0]
    return clicked.groupby("user_id")["note_id"].agg(lambda x: set(int(v) for v in x)).to_dict()


def evaluate(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    processed = Path(config["paths"]["processed_dir"])
    outputs = Path(config["paths"]["outputs_dir"]) / "metrics"
    outputs.mkdir(parents=True, exist_ok=True)
    test = pd.read_parquet(processed / "samples" / "test_interactions.parquet")
    notes = pd.read_parquet(processed / "features" / "note_features.parquet")
    max_users = config.get("evaluation", {}).get("max_users")
    if max_users:
        eval_users = test["user_id"].drop_duplicates().head(int(max_users))
        test = test[test["user_id"].isin(eval_users)]
    positives = positive_items(test)
    results: dict[str, dict[str, float]] = {}

    recall_path = processed / "recalls" / "merged_recall.parquet"
    if recall_path.exists():
        recalls = pd.read_parquet(recall_path)
        if max_users:
            recalls = recalls[recalls["user_id"].isin(positives.keys())]
        recall_scored = recalls.rename(columns={"merged_recall_score": "final_score"})
        for name, mask in {
            "PopularRecall": recalls["recall_source"].str.contains("popular", na=False),
            "ItemCFRecall": recalls["recall_source"].str.contains("itemcf", na=False),
            "TwoTowerRecall": recalls["recall_source"].str.contains("twotower", na=False),
            "MergedRecall": pd.Series(True, index=recalls.index),
        }.items():
            subset = recall_scored[mask].copy()
            if len(subset):
                results[name] = ranking_metrics(subset, positives, score_col="final_score")

        per_user = int(config.get("evaluation", {}).get("ranker_candidates_per_user", 80))
        ranker_candidates = recalls.sort_values(["user_id", "merged_recall_score"], ascending=[True, False]).groupby("user_id").head(per_user)
        ranker_samples = build_candidate_samples(ranker_candidates, config)
        ranked = score_with_ranker(ranker_samples, config)
        if len(ranked):
            results["DINRanker"] = ranking_metrics(ranked, positives, score_col="final_score")
            reranked_rows = []
            profiles = pd.read_parquet(processed / "features" / "user_profiles.parquet").set_index("user_id")
            for user_id, grp in ranked.groupby("user_id"):
                history = profiles.loc[user_id, "history_note_ids"] if user_id in profiles.index else []
                reranked_rows.append(diversity_rerank(grp, notes, history, top_k=50))
            reranked = pd.concat(reranked_rows, ignore_index=True) if reranked_rows else pd.DataFrame()
            if len(reranked):
                results["DINRanker+Rerank"] = ranking_metrics(reranked, positives, score_col="final_score")
                save_parquet(reranked, processed / "recalls" / "ranked_reranked.parquet")
            save_parquet(ranked, processed / "recalls" / "ranked_candidates.parquet")

    direct_test = test.copy()
    direct_test["merged_recall_score"] = 0.0
    direct_scored = score_with_ranker(direct_test, config)
    if "p_click" in direct_scored.columns:
        results.setdefault("DINRanker", {}).update(binary_metrics(direct_scored["click"], direct_scored["p_click"]))

    summary = pd.DataFrame.from_dict(results, orient="index").reset_index(names="model")
    write_json(results, outputs / "eval_summary.json")
    summary.to_csv(outputs / "eval_summary.csv", index=False)
    return results
