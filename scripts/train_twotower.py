from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from redbookrec.config import add_common_args, build_config, ensure_dirs, resolve_device, save_run_config, set_seed, write_json
from redbookrec.models import TwoTower
from redbookrec.train_utils import InteractionDataset, load_history, make_loader, max_ids, train_binary_model


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    cfg = build_config(args, "twotower")
    ensure_dirs(cfg)
    save_run_config(cfg, "twotower")
    set_seed(int(cfg["runtime"]["seed"]))
    device = resolve_device(cfg["runtime"]["device"])

    processed = Path(cfg["paths"]["processed_dir"])
    df = pd.read_parquet(processed / "interactions.parquet")
    if cfg["limits"].get("max_interactions") is not None:
        df = df.head(int(cfg["limits"]["max_interactions"]))
    history = load_history(processed)
    dataset = InteractionDataset(
        df,
        dense_cols=[],
        label_col="click_label",
        history=history,
        last_n=int(cfg.get("rank", {}).get("sim_last_n", 20)),
    )
    loader = make_loader(dataset, int(cfg["runtime"]["batch_size"]), True, int(cfg["runtime"]["num_workers"]))
    num_users, num_notes = max_ids(df)
    model = TwoTower(
        num_users=num_users,
        num_notes=num_notes,
        embedding_dim=int(cfg["twotower"]["embedding_dim"]),
        hidden_dims=list(cfg["twotower"]["hidden_dims"]),
    )
    losses = train_binary_model(
        model,
        loader,
        device,
        int(cfg["runtime"]["epochs"]),
        float(cfg["twotower"]["learning_rate"]),
        float(cfg["twotower"]["weight_decay"]),
    )
    ckpt = Path(cfg["paths"]["checkpoints_dir"]) / "twotower.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "num_users": num_users, "num_notes": num_notes, "config": cfg}, ckpt)

    notes = pd.read_parquet(processed / "notes.parquet")
    users = pd.read_parquet(processed / "users.parquet")
    top_k = int(cfg["twotower"].get("candidate_top_k", 200))
    model.eval().to(device)
    note_ids = torch.tensor(notes["note_id"].astype(int).values, dtype=torch.long, device=device)
    with torch.no_grad():
        note_vec = model.encode_note(note_ids)
        candidates = {}
        for user_id in users["user_id"].astype(int).head(int(cfg["limits"].get("max_users") or len(users))):
            u = torch.tensor([int(user_id)], dtype=torch.long, device=device)
            hist = history.get(int(user_id), [])[-int(cfg.get("rank", {}).get("sim_last_n", 20)) :]
            if len(hist) < int(cfg.get("rank", {}).get("sim_last_n", 20)):
                hist = [0] * (int(cfg.get("rank", {}).get("sim_last_n", 20)) - len(hist)) + hist
            hist_tensor = torch.tensor([hist], dtype=torch.long, device=device)
            scores = (model.encode_user(u, hist_tensor) @ note_vec.T).squeeze(0)
            k = min(top_k, scores.numel())
            vals, idx = torch.topk(scores, k=k)
            rows = []
            for score, pos in zip(vals.cpu().tolist(), idx.cpu().tolist()):
                note = notes.iloc[int(pos)]
                rows.append(
                    {
                        "note_id": int(note.note_id),
                        "raw_note_idx": int(note.raw_note_idx),
                        "score": float(score),
                        "recall_source": "twotower",
                    }
                )
            candidates[int(user_id)] = rows
    out = Path(cfg["paths"]["indexes_dir"]) / "twotower_candidates.joblib"
    joblib.dump(candidates, out)
    write_json(Path(cfg["paths"]["metrics_dir"]) / "twotower_train.json", {"losses": losses, "checkpoint": str(ckpt)})
    print(f"saved_checkpoint={ckpt} saved_candidates={out} losses={losses}")


if __name__ == "__main__":
    main()
