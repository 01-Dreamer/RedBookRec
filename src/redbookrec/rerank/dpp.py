from __future__ import annotations

import pandas as pd


def greedy_dpp(group: pd.DataFrame, note_meta: pd.DataFrame, top_k: int, lambda_diversity: float) -> pd.DataFrame:
    meta = note_meta.set_index("note_idx") if "note_idx" in note_meta else pd.DataFrame()
    candidates = group.sort_values("sim_score", ascending=False).to_dict("records")
    selected: list[dict] = []
    selected_tax: list[str] = []
    while candidates and len(selected) < int(top_k):
        best_i = 0
        best_score = None
        for i, row in enumerate(candidates):
            tax = ""
            if not meta.empty and int(row["note_idx"]) in meta.index:
                tax = str(meta.loc[int(row["note_idx"]), "taxonomy1_id"])
            max_sim = 1.0 if tax and tax in selected_tax else 0.0
            dpp_score = float(row["sim_score"]) - float(lambda_diversity) * max_sim
            if best_score is None or dpp_score > best_score:
                best_i = i
                best_score = dpp_score
        chosen = candidates.pop(best_i)
        chosen["dpp_score"] = float(best_score or chosen["sim_score"])
        if not meta.empty and int(chosen["note_idx"]) in meta.index:
            selected_tax.append(str(meta.loc[int(chosen["note_idx"]), "taxonomy1_id"]))
        selected.append(chosen)
    out = pd.DataFrame(selected)
    out["final_rank"] = range(1, len(out) + 1)
    return out
