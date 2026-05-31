from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def update_config(cfg: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    for key, value in updates.items():
        if value is not None:
            out[key] = value
    return out


def get_nested(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_nested(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    cur = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def apply_cli_overrides(cfg: dict[str, Any], args: Any) -> dict[str, Any]:
    out = deepcopy(cfg)
    for name in ["max_notes", "max_requests", "max_train_samples", "max_eval_samples"]:
        value = getattr(args, name, None)
        if value is not None:
            set_nested(out, f"infer.{name}", value)
            set_nested(out, f"train.{name}", value)
    if getattr(args, "smoke_test", False):
        set_nested(out, "train.smoke_test", True)
    return out


def get_device(device: str):
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
