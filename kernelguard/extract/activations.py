"""Activation extraction (P3).

One forward pass per transcript with output_hidden_states=True. Caches every layer at the
last prompt token so the layer sweep (RQ1) costs nothing after the first pass.

Layer indexing convention used everywhere in this repo:
  hidden_states[0]  = embedding output
  hidden_states[l]  = residual stream after decoder block l (1-based), l in 1..N
A runtime hook that wants hidden_states[l] registers on model.model.layers[l-1].
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from kernelguard.data.schema import Item


@dataclass
class ModelBundle:
    model: torch.nn.Module
    tokenizer: object
    device: str
    n_layers: int
    hidden_dim: int

    @property
    def layers(self):
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise AttributeError("cannot resolve decoder layer stack")


def load_model(name: str, device: str | None = None, dtype: str = "auto") -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "auto":
        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype)
    model.to(device).eval()
    cfg = model.config
    return ModelBundle(model, tok, device, int(cfg.num_hidden_layers), int(cfg.hidden_size))


def render_prompt(mb: ModelBundle, item: Item) -> str:
    """Chat template + tool schema + generation prompt + optional assistant prefix."""
    tok = mb.tokenizer
    try:
        text = tok.apply_chat_template(item.messages(), tools=item.tools, tokenize=False, add_generation_prompt=True)
    except TypeError:  # template without tools support
        text = tok.apply_chat_template(item.messages(), tokenize=False, add_generation_prompt=True)
    return text + item.assistant_prefix


def _cache_key(model_name: str, item: Item) -> str:
    """Content key: depends only on what the model sees, so editing templates never invalidates other items."""
    payload = json.dumps(
        [model_name, item.system, item.tools, item.context_docs, item.user, item.assistant_prefix], sort_keys=True
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _legacy_cache_key(model_name: str, item: Item) -> str:
    """Pre-v3 key that included the item id. Kept only to migrate old cache files."""
    h = hashlib.sha1(f"{model_name}|{item.id}|{item.user}|{item.assistant_prefix}|{item.system}".encode()).hexdigest()
    return h[:16]


def cache_path(cache_dir: str | Path, model_name: str, item: Item) -> Path:
    """Path of the item's cached activations. Renames a legacy-keyed file in place if one exists."""
    cache = Path(cache_dir)
    p = cache / f"{_cache_key(model_name, item)}.npy"
    if not p.exists():
        old = cache / f"{_legacy_cache_key(model_name, item)}.npy"
        if old.exists():
            old.rename(p)
    return p


@torch.no_grad()
def extract_last_token_all_layers(mb: ModelBundle, item: Item) -> np.ndarray:
    """Returns [N+1, d] float32: residual stream at the final prompt token for every layer."""
    text = render_prompt(mb, item)
    enc = mb.tokenizer(text, return_tensors="pt").to(mb.device)
    out = mb.model(**enc, output_hidden_states=True)
    hs = torch.stack([h[0, -1, :] for h in out.hidden_states])  # [N+1, d]
    return hs.float().cpu().numpy()


@torch.no_grad()
def extract_batch_last_token_all_layers(mb: ModelBundle, items: list[Item]) -> np.ndarray:
    """Batched version of extract_last_token_all_layers. Returns [B, N+1, d] float32.

    Left padding puts every prompt's last real token at position -1. Position ids are rebuilt from the
    attention mask so each prompt sees the same positions it would see unpadded.
    """
    tok = mb.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    side = tok.padding_side
    tok.padding_side = "left"
    try:
        enc = tok([render_prompt(mb, it) for it in items], return_tensors="pt", padding=True).to(mb.device)
    finally:
        tok.padding_side = side
    pos = (enc.attention_mask.cumsum(-1) - 1).clamp(min=0)
    out = mb.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, position_ids=pos,
                   output_hidden_states=True)
    hs = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)  # [B, N+1, d]
    return hs.float().cpu().numpy()


def extract_many(
    mb: ModelBundle | None,
    items: Iterable[Item],
    model_name: str,
    cache_dir: str | Path | None = None,
    verbose: bool = False,
    batch_size: int = 1,
) -> tuple[np.ndarray, list[Item]]:
    """Returns ([n_items, N+1, d], items) using an .npy cache keyed by (model, prompt content).

    batch_size 1 runs one prompt per forward pass, which suits CPU. Use 8 to 32 on a GPU.
    """
    items = list(items)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    paths = [cache_path(cache, model_name, it) if cache else None for it in items]
    acts: list[np.ndarray | None] = [None] * len(items)
    todo = []
    for k, p in enumerate(paths):
        if p is not None and p.exists():
            acts[k] = np.load(p).astype(np.float32)
        else:
            todo.append(k)
    if todo and mb is None:
        raise ValueError(f"{len(todo)} items are not cached and no model was given")
    todo.sort(key=lambda k: len(items[k].user))  # similar lengths per batch means less padding
    done = 0
    for s in range(0, len(todo), batch_size):
        ks = todo[s:s + batch_size]
        if batch_size == 1:
            batch = [extract_last_token_all_layers(mb, items[ks[0]])]
        else:
            batch = list(extract_batch_last_token_all_layers(mb, [items[k] for k in ks]))
        for k, a in zip(ks, batch):
            a16 = a.astype(np.float16)
            if paths[k] is not None:
                np.save(paths[k], a16)
            acts[k] = a16.astype(np.float32)
        done += len(ks)
        if verbose and (done // 50 != (done - len(ks)) // 50 or done == len(todo)):
            print(f"  extracted {done}/{len(todo)} uncached ({len(items) - len(todo)} cached)", flush=True)
    return np.stack(acts), items
