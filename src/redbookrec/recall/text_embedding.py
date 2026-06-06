from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from redbookrec.utils.config import get_device


def mean_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def encode_texts(
    texts: Iterable[str],
    model_name_or_path: str,
    batch_size: int = 64,
    max_length: int = 256,
    device: str = "auto",
    normalize: bool = True,
    fp16: bool = False,
) -> np.ndarray:
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("text encoder requires transformers. Install it or set model.use_text_emb=false.") from exc

    model_device = torch.device(get_device(device))
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(model_device)
    use_fp16 = bool(fp16 and model_device.type == "cuda")
    if use_fp16:
        model = model.half()
    model.eval()

    values = ["" if x is None else str(x) for x in texts]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(values), int(batch_size)), desc="encode_text", leave=False):
            batch = values[start : start + int(batch_size)]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**inputs)
            emb = mean_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            if normalize:
                emb = torch.nn.functional.normalize(emb, dim=-1)
            chunks.append(emb.cpu().numpy().astype("float32"))
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype="float32")


def load_text_embeddings(path: str | Path) -> np.ndarray | None:
    p = Path(path)
    if not p.exists():
        return None
    return np.load(p)


def save_text_embeddings(path: str | Path, embeddings: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, embeddings.astype("float32"))
