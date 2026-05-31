from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from redbookrec.data.id_mapping import build_id_map
from redbookrec.data.load_qilin import normalize_detail, parse_nested
from redbookrec.data.preprocess_rec import REC_COLUMNS, read_recommendation
from redbookrec.utils.io import read_json, write_json


def _exposure_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base in df.itertuples(index=False):
        details = parse_nested(getattr(base, "rec_result_details_with_idx", []))
        history = parse_nested(getattr(base, "recent_clicked_note_idxs", []))
        for item in details:
            detail = normalize_detail(item)
            try:
                note_idx = int(detail.get("note_idx"))
            except Exception:
                continue
            label = int(float(detail.get("click", 0) or 0) > 0)
            rows.append(
                {
                    "request_idx": int(getattr(base, "request_idx", -1)),
                    "session_idx": int(getattr(base, "session_idx", -1)),
                    "user_idx": int(getattr(base, "user_idx", -1)),
                    "recent_clicked_note_idxs": history,
                    "note_idx": note_idx,
                    "pos_note_idx": note_idx,
                    "position": int(detail.get("position", 0) or 0),
                    "label": label,
                }
            )
    return rows


def _extend_note_map(map_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    data = read_json(map_path) if map_path.exists() else {"raw_to_model": {}, "padding_id": 0}
    raw_to_model = data.get("raw_to_model", {})
    values: list[int] = []
    for row in rows:
        values.append(int(row.get("note_idx", row.get("pos_note_idx"))))
        values.extend(int(x) for x in row.get("recent_clicked_note_idxs", []) if str(x).lstrip("-").isdigit())
    raw_to_model = build_id_map(values, raw_to_model)
    data["raw_to_model"] = raw_to_model
    data["model_to_raw"] = {str(v): int(k) for k, v in raw_to_model.items()}
    data["num_notes"] = len(raw_to_model) + 1
    data["padding_id"] = 0
    write_json(map_path, data)
    return data


def build_recall_samples(cfg: dict, max_requests: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_dir = cfg["data"]["dataset_dir"]
    train_df = read_recommendation(dataset_dir, "recommendation_train", max_requests=max_requests)
    test_df = read_recommendation(dataset_dir, "recommendation_test", max_requests=max_requests)
    train_rows = _exposure_rows(train_df)
    test_rows = _exposure_rows(test_df)

    map_path = Path(cfg["data"]["note_id_map_path"])
    _extend_note_map(map_path, train_rows + test_rows)

    train = pd.DataFrame(train_rows)
    test = pd.DataFrame(test_rows)
    train_path = Path(cfg["data"]["train_samples_path"])
    test_path = Path(cfg["data"]["test_samples_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return train, test
