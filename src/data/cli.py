from __future__ import annotations

import argparse
from typing import Any

from src.data.io import deep_update, load_config


def add_config_arguments(parser: argparse.ArgumentParser, include_training: bool = False) -> argparse.ArgumentParser:
    parser.add_argument("--config", action="append", default=[], help="Extra yaml config file, loaded after defaults.")
    parser.add_argument("--full", action="store_true", help="Load configs/full.yaml and disable debug limits.")
    parser.add_argument("--device", default=None, help="Override runtime.device, for example cpu, cuda, or auto.")
    parser.add_argument("--debug", dest="debug", action="store_true", default=None, help="Enable debug data limits.")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug data limits.")
    parser.add_argument("--max-users", type=int, default=None, help="Override debug.max_users.")
    parser.add_argument("--max-notes", type=int, default=None, help="Override debug.max_notes.")
    parser.add_argument("--max-interactions", type=int, default=None, help="Override debug.max_interactions.")
    parser.add_argument("--max-eval-users", type=int, default=None, help="Override evaluation.max_users.")
    if include_training:
        parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
        parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
        parser.add_argument("--max-train-samples", type=int, default=None, help="Override max_train_samples.")
    return parser


def load_config_with_overrides(args: argparse.Namespace, defaults: list[str]) -> dict[str, Any]:
    config_paths = list(defaults)
    if getattr(args, "full", False):
        config_paths.append("configs/full.yaml")
    config_paths.extend(getattr(args, "config", []) or [])
    config = load_config(*config_paths)

    overrides: dict[str, Any] = {}
    if getattr(args, "device", None):
        overrides.setdefault("runtime", {})["device"] = args.device
    if getattr(args, "debug", None) is not None:
        overrides.setdefault("debug", {})["enabled"] = bool(args.debug)
    for arg_name, config_name in [
        ("max_users", "max_users"),
        ("max_notes", "max_notes"),
        ("max_interactions", "max_interactions"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides.setdefault("debug", {})[config_name] = value
    if getattr(args, "max_eval_users", None) is not None:
        overrides.setdefault("evaluation", {})["max_users"] = args.max_eval_users
    for arg_name, config_name in [
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("max_train_samples", "max_train_samples"),
    ]:
        value = getattr(args, arg_name, None)
        if value is not None:
            overrides[config_name] = value
    return deep_update(config, overrides)
