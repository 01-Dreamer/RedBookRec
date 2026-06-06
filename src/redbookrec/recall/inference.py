from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from redbookrec.data.id_mapping import map_raw, map_sequence
from redbookrec.data.load_qilin import parse_nested
from redbookrec.data.preprocess_rec import clicked_set, read_recommendation
from redbookrec.recall.dataset import NOTE_DENSE_COLS, TEXT_BUCKETS, TEXT_MAX_LEN, USER_DENSE_COLS, build_note_features, text_to_ids
from redbookrec.recall.faiss_index import search_topk
from redbookrec.recall.text_embedding import encode_texts, load_text_embeddings
from redbookrec.recall.two_tower import DualTowerRecall
from redbookrec.utils.config import get_device


def _load_model(cfg: dict, device: torch.device) -> tuple[DualTowerRecall, dict]:
    ckpt = torch.load(cfg["paths"]["recall_checkpoint"], map_location=device)
    model = DualTowerRecall(
        num_users=int(ckpt["num_users"]),
        num_notes=int(ckpt["num_notes"]),
        embed_dim=int(ckpt["embed_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
        temperature=float(ckpt.get("temperature", 0.05)),
        user_dense_dim=int(ckpt.get("user_dense_dim", len(USER_DENSE_COLS))),
        note_dense_dim=int(ckpt.get("note_dense_dim", len(NOTE_DENSE_COLS))),
        text_vocab_size=int(ckpt.get("text_vocab_size", TEXT_BUCKETS)),
        text_emb_dim=int(ckpt.get("text_emb_dim", cfg["model"].get("text_emb_dim", 768))),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def infer_recall(cfg: dict, max_notes: int | None = None, max_requests: int | None = None) -> pd.DataFrame:
    device = torch.device(get_device(cfg["train"].get("device", "auto")))
    model, ckpt = _load_model(cfg, device)
    use_text_emb = bool(ckpt.get("use_text_emb", cfg["model"].get("use_text_emb", False)))
    all_note_df = pd.read_parquet(cfg["data"]["note_text_path"])
    note_text_emb = load_text_embeddings(cfg["data"]["note_text_emb_path"]) if use_text_emb else None
    if use_text_emb:
        if note_text_emb is None:
            raise RuntimeError("note text embeddings not found. Run: python scripts/prepare_text_embeddings.py --config configs/recall.yaml")
        if len(note_text_emb) != len(all_note_df):
            raise RuntimeError("note_text_emb row count does not match note_text parquet. Regenerate text embeddings.")
    note_text_pos_map = {int(raw): pos for pos, raw in enumerate(all_note_df["note_idx"].astype("int64").tolist())} if use_text_emb else {}
    note_df = all_note_df
    if max_notes is None:
        max_notes = cfg.get("infer", {}).get("max_notes")
    if max_notes:
        note_df = all_note_df.head(int(max_notes)).copy()

    if max_requests is None:
        max_requests = cfg.get("infer", {}).get("max_requests")
    req_df = read_recommendation(cfg["data"]["dataset_dir"], "recommendation_test", max_requests=max_requests)
    note_map = ckpt["note_map"]
    clicked_for_eval: set[int] = set()
    for req in req_df.itertuples(index=False):
        clicked_for_eval.update(clicked_set(parse_nested(getattr(req, "rec_result_details_with_idx", []))))
    present = set(note_df["note_idx"].astype(int))
    extras = sorted(clicked_for_eval - present)
    if extras:
        extra_df = all_note_df[all_note_df["note_idx"].astype(int).isin(extras)].copy()
        if not extra_df.empty:
            note_df = pd.concat([note_df, extra_df], ignore_index=True)
    note_feature_map = build_note_features(note_df)
    note_feat_rows = [note_feature_map.get(str(int(raw)), {}) for raw in note_df["note_idx"].astype(int)]
    note_ids = note_df["note_id"].astype("int64").to_numpy()
    note_tensor = torch.tensor(note_ids, dtype=torch.long, device=device)
    note_type_tensor = torch.tensor([int(x.get("note_type", 0)) for x in note_feat_rows], dtype=torch.long, device=device)
    note_tax_tensor = torch.tensor([x.get("tax", [0, 0, 0]) for x in note_feat_rows], dtype=torch.long, device=device)
    note_dense_tensor = torch.tensor([x.get("dense", [0.0] * len(NOTE_DENSE_COLS)) for x in note_feat_rows], dtype=torch.float32, device=device)
    note_text_tensor = torch.tensor([x.get("text_ids", [0] * int(ckpt.get("text_max_len", TEXT_MAX_LEN))) for x in note_feat_rows], dtype=torch.long, device=device)
    note_text_emb_tensor = None
    if use_text_emb and note_text_emb is not None:
        text_emb_dim = int(ckpt.get("text_emb_dim", cfg["model"].get("text_emb_dim", note_text_emb.shape[1])))
        note_text_emb_rows = np.zeros((len(note_df), text_emb_dim), dtype="float32")
        for i, raw in enumerate(note_df["note_idx"].astype("int64").tolist()):
            pos = note_text_pos_map.get(int(raw), -1)
            if pos >= 0:
                note_text_emb_rows[i] = note_text_emb[pos]
        note_text_emb_tensor = torch.tensor(note_text_emb_rows, dtype=torch.float32, device=device)
    note_vecs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(note_tensor), 8192):
            end = start + 8192
            note_vecs.append(
                model.encode_note(
                    note_tensor[start:end],
                    note_type_tensor[start:end],
                    note_tax_tensor[start:end, 0],
                    note_tax_tensor[start:end, 1],
                    note_tax_tensor[start:end, 2],
                    note_dense_tensor[start:end],
                    note_text_tensor[start:end],
                    note_text_emb_tensor[start:end] if note_text_emb_tensor is not None else None,
                )
                .cpu()
                .numpy()
            )
    note_emb = np.vstack(note_vecs).astype("float32")
    Path(cfg["infer"]["note_emb_path"]).parent.mkdir(parents=True, exist_ok=True)
    np.save(Path(cfg["infer"]["note_emb_path"]), note_emb)
    top_k = min(int(cfg["infer"].get("top_k", 1000)), len(note_df))
    rows: list[dict] = []
    user_map = ckpt["user_map"]
    user_features = ckpt.get("user_features", {})
    max_history_len = int(ckpt["max_history_len"])

    user_embs: list[np.ndarray] = []
    request_meta: list[dict] = []
    query_emb_map: dict[int, np.ndarray] = {}
    if use_text_emb and len(req_df):
        query_emb = encode_texts(
            req_df["query"].fillna("").astype(str).tolist(),
            model_name_or_path=cfg["model"].get("text_encoder_name_or_path", "../model/bert-base-chinese/"),
            batch_size=int(cfg["model"].get("text_encoder_batch_size", 64)),
            max_length=int(cfg["model"].get("text_encoder_max_length", 256)),
            device=cfg["train"].get("device", "auto"),
        )
        query_emb_map = {int(req.request_idx): query_emb[i] for i, req in enumerate(req_df.itertuples(index=False))}
    with torch.no_grad():
        for req in tqdm(req_df.itertuples(index=False), total=len(req_df), desc="encode_requests", leave=False):
            user_id = map_raw(user_map, getattr(req, "user_idx", 0))
            hist = map_sequence(note_map, parse_nested(getattr(req, "recent_clicked_note_idxs", [])), max_history_len)
            hist = [0] * (max_history_len - len(hist)) + hist
            u = torch.tensor([user_id], dtype=torch.long, device=device)
            h = torch.tensor([hist], dtype=torch.long, device=device)
            user_feat = user_features.get(str(int(getattr(req, "user_idx"))), {})
            user_cat = torch.tensor([user_feat.get("cat", [0, 0, 0, 0])], dtype=torch.long, device=device)
            user_dense = torch.tensor([user_feat.get("dense", [0.0] * len(USER_DENSE_COLS))], dtype=torch.float32, device=device)
            query_text = torch.tensor([text_to_ids(getattr(req, "query", ""), max_len=int(ckpt.get("text_max_len", TEXT_MAX_LEN)))], dtype=torch.long, device=device)
            query_dense = None
            if use_text_emb:
                query_dense = torch.tensor([query_emb_map.get(int(getattr(req, "request_idx")), np.zeros(int(ckpt.get("text_emb_dim", cfg["model"].get("text_emb_dim", 768))), dtype="float32"))], dtype=torch.float32, device=device)
            user_embs.append(model.encode_user(u, h, user_cat, user_dense, query_text, query_dense).cpu().numpy()[0])
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
