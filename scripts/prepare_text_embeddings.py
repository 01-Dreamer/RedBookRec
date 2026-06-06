from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from redbookrec.recall.text_embedding import encode_texts, save_text_embeddings
from redbookrec.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recall.yaml")
    parser.add_argument("--max-notes", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    encoder_path = model_cfg.get("text_encoder_name_or_path", "../model/bert-base-chinese/")
    note_df = pd.read_parquet(cfg["data"]["note_text_path"], columns=["note_text"])
    if args.max_notes:
        note_df = note_df.head(int(args.max_notes))
    emb = encode_texts(
        note_df["note_text"].fillna("").astype(str).tolist(),
        model_name_or_path=encoder_path,
        batch_size=int(model_cfg.get("text_encoder_batch_size", 64)),
        max_length=int(model_cfg.get("text_encoder_max_length", 256)),
        device=cfg["train"].get("device", "auto"),
        fp16=bool(model_cfg.get("text_encoder_fp16", True)),
    )
    save_text_embeddings(cfg["data"]["note_text_emb_path"], emb)
    print(f"saved_note_text_emb_shape={emb.shape} path={cfg['data']['note_text_emb_path']}")


if __name__ == "__main__":
    main()
