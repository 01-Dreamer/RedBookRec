from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from redbookrec.data.id_mapping import map_raw, map_sequence
from redbookrec.data.load_qilin import parse_nested
from redbookrec.data.preprocess_rec import clicked_set, read_recommendation
from redbookrec.recall.faiss_index import search_topk
from redbookrec.recall.model import DualTowerRecall
from redbookrec.utils.config import get_device


def _load_model(cfg: dict, device: torch.device) -> tuple[DualTowerRecall, dict]:
    ckpt = torch.load(cfg["paths"]["recall_checkpoint"], map_location=device)
    model = DualTowerRecall(
        num_users=int(ckpt["num_users"]),
        num_notes=int(ckpt["num_notes"]),
        embed_dim=int(ckpt["embed_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
        temperature=float(ckpt.get("temperature", 0.05)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def infer_recall(cfg: dict, max_notes: int | None = None, max_requests: int | None = None) -> pd.DataFrame:
    device = torch.device(get_device(cfg["train"].get("device", "auto")))
    model, ckpt = _load_model(cfg, device)
    note_df = pd.read_parquet(cfg["data"]["note_text_path"])
    if max_notes is None:
        max_notes = cfg.get("infer", {}).get("max_notes")
    if max_notes:
        note_df = note_df.head(int(max_notes))

    if max_requests is None:
        max_requests = cfg.get("infer", {}).get("max_requests")
    req_df = read_recommendation(cfg["data"]["dataset_dir"], "recommendation_test", max_requests=max_requests)
    note_map = ckpt["note_map"]
    clicked_for_eval: set[int] = set()
    for req in req_df.itertuples(index=False):
        clicked_for_eval.update(clicked_set(parse_nested(getattr(req, "rec_result_details_with_idx", []))))
    present = set(note_df["note_idx"].astype(int))
    extras = [
        {"note_idx": raw, "note_id": map_raw(note_map, raw)}
        for raw in sorted(clicked_for_eval - present)
        if map_raw(note_map, raw) > 0
    ]
    if extras:
        note_df = pd.concat([note_df, pd.DataFrame(extras)], ignore_index=True)
    note_ids = note_df["note_id"].astype("int64").to_numpy()
    note_tensor = torch.tensor(note_ids, dtype=torch.long, device=device)
    note_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(note_tensor), 8192):
            note_vecs.append(model.encode_note(note_tensor[start : start + 8192]).cpu().numpy())
    note_emb = np.vstack(note_vecs).astype("float32")
    Path(cfg["infer"]["note_emb_path"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(Path(cfg["infer"]["note_emb_path"]), note_emb)
    top_k = min(int(cfg["infer"].get("top_k", 1000)), len(note_df))
    rows: list[dict] = []
    user_map = ckpt["user_map"]
    max_history_len = int(ckpt["max_history_len"])

    user_embs: list[np.ndarray] = []
    request_meta: list[dict] = []
    with torch.no_grad():
        for req in tqdm(req_df.itertuples(index=False), total=len(req_df), desc="encode_requests", leave=False):
            user_id = map_raw(user_map, getattr(req, "user_idx", 0))
            hist = map_sequence(note_map, parse_nested(getattr(req, "recent_clicked_note_idxs", [])), max_history_len)
            hist = [0] * (max_history_len - len(hist)) + hist
            u = torch.tensor([user_id], dtype=torch.long, device=device)
            h = torch.tensor([hist], dtype=torch.long, device=device)
            user_embs.append(model.encode_user(u, h).cpu().numpy()[0])
            clicked = clicked_set(parse_nested(getattr(req, "rec_result_details_with_idx", [])))
            request_meta.append(
                {
                    "request_idx": int(getattr(req, "request_idx")),
                    "user_idx": int(getattr(req, "user_idx")),
                    "clicked": clicked,
                }
            )
    if not user_embs:
        out = pd.DataFrame(columns=["request_idx", "user_idx", "note_idx", "recall_score", "recall_rank", "label_click", "source"])
    else:
        scores, indices = search_topk(np.vstack(user_embs).astype("float32"), note_emb, top_k)
        raw_note_idxs = note_df["note_idx"].astype("int64").to_numpy()
        for i, meta in enumerate(request_meta):
            clicked = meta["clicked"]
            for rank, pos in enumerate(indices[i], start=1):
                note_idx = int(raw_note_idxs[int(pos)])
                rows.append(
                    {
                        "request_idx": meta["request_idx"],
                        "user_idx": meta["user_idx"],
                        "note_idx": note_idx,
                        "recall_score": float(scores[i][rank - 1]),
                        "recall_rank": rank,
                        "label_click": int(note_idx in clicked),
                        "source": "dual_tower",
                    }
                )
        out = pd.DataFrame(rows)
    path = Path(cfg["infer"]["dual_output_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out
