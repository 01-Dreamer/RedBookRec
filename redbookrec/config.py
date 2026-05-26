from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


STAGE_CONFIGS = {
    "prepare": [],
    "inspect": [],
    "search": ["configs/recall.yaml"],
    "twotower": ["configs/recall.yaml"],
    "dcn": ["configs/rank.yaml"],
    "sim": ["configs/rank.yaml"],
    "rerank": ["configs/rerank.yaml"],
    "evaluate": ["configs/recall.yaml", "configs/rank.yaml", "configs/rerank.yaml"],
    "recommend": ["configs/recall.yaml", "configs/rank.yaml", "configs/rerank.yaml"],
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(paths: Iterable[str | Path]) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for path in paths:
        cfg = deep_merge(cfg, load_yaml(path))
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", nargs="*", default=None, help="YAML config files to merge.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true", help="Run a small local smoke test.")
    mode.add_argument("--full", action="store_true", help="Run on full data.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-notes", type=int, default=None)
    parser.add_argument("--max-interactions", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser


def build_config(args: argparse.Namespace, stage: str) -> dict[str, Any]:
    if args.config:
        paths = list(args.config)
    else:
        paths = ["configs/base.yaml", *STAGE_CONFIGS.get(stage, [])]
        paths.append("configs/full.yaml" if args.full else "configs/debug.yaml")
    cfg = load_config(paths)

    if args.debug:
        cfg = deep_merge(cfg, load_yaml("configs/debug.yaml"))
        cfg.setdefault("runtime", {})["mode"] = "debug"
    if args.full:
        cfg = deep_merge(cfg, load_yaml("configs/full.yaml"))
        cfg.setdefault("runtime", {})["mode"] = "full"

    runtime = cfg.setdefault("runtime", {})
    limits = cfg.setdefault("limits", {})
    for attr, key in [
        ("device", "device"),
        ("batch_size", "batch_size"),
        ("epochs", "epochs"),
        ("num_workers", "num_workers"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            runtime[key] = value
    for attr, key in [
        ("max_users", "max_users"),
        ("max_notes", "max_notes"),
        ("max_interactions", "max_interactions"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            limits[key] = value
    if getattr(args, "mixed_precision", False):
        runtime["mixed_precision"] = True

    rank = cfg.setdefault("rank", {})
    for attr, key in [
        ("sim_last_n", "sim_last_n"),
        ("sim_top_k", "sim_top_k"),
        ("sim_max_history", "sim_max_history"),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            rank[key] = value

    search = cfg.setdefault("search", {})
    if getattr(args, "top_k", None) is not None:
        search["top_k"] = args.top_k
        cfg.setdefault("rerank", {})["final_top_k"] = args.top_k

    cfg["run_id"] = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg["config_files"] = [str(p) for p in paths]
    return cfg


def project_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])


def dataset_path(cfg: dict[str, Any], relative_key: str) -> Path:
    return Path(cfg["paths"]["dataset_dir"]) / cfg["data"][relative_key]


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in [
        "artifacts_dir",
        "processed_dir",
        "runs_dir",
        "indexes_dir",
        "checkpoints_dir",
        "metrics_dir",
        "logs_dir",
    ]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def save_run_config(cfg: dict[str, Any], stage: str) -> Path:
    run_dir = Path(cfg["paths"]["runs_dir"]) / cfg["run_id"] / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
