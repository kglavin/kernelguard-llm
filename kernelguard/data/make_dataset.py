"""S1: build the labeled request set from seeds plus yaml templates.

Sources, merged per (policy, family, condition):
  kernelguard/data/seeds.jsonl     reviewed hand-written items, always included
  data/templates/<policy>.yaml     extra items per family; exec_mal lists are extended by hand
  data/templates/wrappers.yaml     phrasing wrappers applied to every item

Split stays by family, so held-out means held-out request type and phrasing.

Run: python -m kernelguard.data.make_dataset  ->  data/v1/items.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

from kernelguard.data.schema import CONDITIONS, Item, load_jsonl, save_jsonl

SEEDS = Path(__file__).parent / "seeds.jsonl"
TEMPLATES = Path("data/templates")
OUT = Path("data/v1/items.jsonl")


def load_templates(tdir: Path) -> tuple[list[str], list[tuple[str, str, str, str]]]:
    """Returns (wrappers, rows). wrappers = {"exec": [...], "discuss": [...]}; rows = (policy, family, condition, text)."""
    wrappers = yaml.safe_load(open(tdir / "wrappers.yaml"))
    if set(wrappers) != {"exec", "discuss"}:
        raise ValueError("wrappers.yaml must define exactly 'exec' and 'discuss' lists")
    rows = []
    for f in sorted(tdir.glob("*.yaml")):
        if f.name == "wrappers.yaml":
            continue
        doc = yaml.safe_load(open(f))
        policy = doc["policy"]
        for family, conds in (doc.get("families") or {}).items():
            for cond, texts in (conds or {}).items():
                if cond not in CONDITIONS:
                    raise ValueError(f"{f}: family {family}: unknown condition {cond}")
                for t in texts or []:
                    rows.append((policy, family, cond, t))
    return wrappers, rows


def build(seeds: list[Item], wrappers: dict[str, list[str]], rows: list[tuple[str, str, str, str]]) -> list[Item]:
    base = [(s.policy, s.family, s.condition, s.user, s.id) for s in seeds]
    for k, (p, fam, c, t) in enumerate(rows):
        base.append((p, fam, c, t, f"{p}-{fam}-{c}-t{k:03d}"))

    has_bad = {(p, fam) for p, fam, c, _, _ in base if c == "exec_mal"}
    for p, fam in sorted({(p, fam) for p, fam, _, _, _ in base} - has_bad):
        print(f"  skipping {p}/{fam}: no exec_mal items yet", file=sys.stderr)
    base = [b for b in base if (b[0], b[1]) in has_bad]

    out: list[Item] = []
    seen: set[tuple[str, str]] = set()
    for p, fam, c, text, bid in base:
        wset = wrappers["discuss"] if c.startswith("discuss") else wrappers["exec"]
        for wi, w in enumerate(wset):
            user = w.format(x=text, n=1000 + wi)
            if (p, user) in seen:
                continue
            seen.add((p, user))
            it = Item(id=f"{bid}-w{wi}", policy=p, condition=c, family=fam, user=user,
                      meta={"base": bid, "wrapper": wi})
            it.validate()
            out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=str(TEMPLATES))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    wrappers, rows = load_templates(Path(a.templates))
    items = build(load_jsonl(SEEDS), wrappers, rows)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(items, out)

    print(f"wrote {len(items)} items to {out}")
    for p in sorted({i.policy for i in items}):
        c = Counter(i.condition for i in items if i.policy == p)
        fams = sorted({i.family for i in items if i.policy == p})
        print(f"  {p:12s} {dict(c)} families={fams}")


if __name__ == "__main__":
    main()
