from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, resolve_device, save_run_config, set_seed, write_json
from redbookrec.models import SIMRanker
from redbookrec.train_utils import (
    InteractionDataset,
    load_history,
    load_note_taxonomy,
    load_training_frame,
    make_loader,
    max_ids,
    prepare_dense,
    train_binary_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--sim-last-n", type=int, default=None)
    parser.add_argument("--sim-top-k", type=int, default=None)
    parser.add_argument("--sim-max-history", type=int, default=None)
    args = parser.parse_args()
    cfg = build_config(args, "sim")
    ensure_dirs(cfg)
    save_run_config(cfg, "sim")
    set_seed(int(cfg["runtime"]["seed"]))
    device = resolve_device(cfg["runtime"]["device"])

    processed = Path(cfg["paths"]["processed_dir"])
    df = load_training_frame(processed)
    if cfg["limits"].get("max_interactions") is not None:
        df = df.head(int(cfg["limits"]["max_interactions"]))
    df, dense_cols, _ = prepare_dense(df, processed, "sim")
    history = load_history(processed)
    note_taxonomy = load_note_taxonomy(processed)
    label_cols = [c for c in ["click_label", "like", "collect", "comment", "share"] if c in df.columns]
    dataset = InteractionDataset(
        df,
        dense_cols=dense_cols,
        label_cols=label_cols,
        history=history,
        note_taxonomy=note_taxonomy,
        last_n=int(cfg["rank"]["sim_last_n"]),
        top_k=int(cfg["rank"]["sim_top_k"]),
    )
    loader = make_loader(dataset, int(cfg["runtime"]["batch_size"]), True, int(cfg["runtime"]["num_workers"]))
    num_users, num_notes = max_ids(df)
    model = SIMRanker(
        num_users,
        num_notes,
        dense_dim=len(dense_cols),
        embedding_dim=int(cfg["twotower"]["embedding_dim"]),
        hidden_dims=list(cfg["rank"]["sim_hidden_dims"]),
        dropout=float(cfg["rank"]["dropout"]),
        output_dim=len(label_cols),
    )
    losses = train_binary_model(
        model,
        loader,
        device,
        int(cfg["runtime"]["epochs"]),
        float(cfg["rank"]["learning_rate"]),
        float(cfg["rank"]["weight_decay"]),
    )
    ckpt = Path(cfg["paths"]["checkpoints_dir"]) / "sim.pt"
    torch.save(
        {"model": model.state_dict(), "num_users": num_users, "num_notes": num_notes, "dense_cols": dense_cols, "label_cols": label_cols, "config": cfg},
        ckpt,
    )
    write_json(Path(cfg["paths"]["metrics_dir"]) / "sim_train.json", {"losses": losses, "checkpoint": str(ckpt)})
    print(f"saved_checkpoint={ckpt} dense_dim={len(dense_cols)} losses={losses}")


if __name__ == "__main__":
    main()
