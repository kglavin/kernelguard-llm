#!/usr/bin/env python3
"""Build a small slice of data/v1 for the preflight run.

Per policy and family: 2 bad requests, 2 benign execution requests, 1 question on each topic type,
all at wrapper 0. That gives 72 items with every family present, so the family split, the
outbound step and the runtime demo all run.

Usage: python scripts/make_preflight.py [data/v1/items.jsonl] [reports/preflight/items.jsonl]
"""
import sys
from pathlib import Path

from kernelguard.data.schema import load_jsonl, save_jsonl

TAKE = {"exec_mal": 2, "exec_ben": 2, "discuss_mal": 1, "discuss_ben": 1}

src = sys.argv[1] if len(sys.argv) > 1 else "data/v1/items.jsonl"
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/preflight/items.jsonl")
items = [i for i in load_jsonl(src) if i.meta.get("wrapper") == 0]
out, seen = [], {}
for it in items:
    k = (it.policy, it.family, it.condition)
    if seen.get(k, 0) < TAKE[it.condition]:
        seen[k] = seen.get(k, 0) + 1
        out.append(it)
dst.parent.mkdir(parents=True, exist_ok=True)
save_jsonl(out, dst)
print(f"wrote {len(out)} items to {dst}")
