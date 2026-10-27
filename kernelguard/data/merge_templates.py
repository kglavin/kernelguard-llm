"""Merge items from an incoming yaml into the matching data/templates/<policy>.yaml.

The incoming file uses the same layout as a template: `policy:` plus `families:` with
condition lists. Items are appended to the existing lists. The target keeps its comments
and order, and items already present are skipped. Families must already exist in the target.

Usage:
  python -m kernelguard.data.merge_templates INCOMING.yaml [--templates DIR] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

from kernelguard.data.schema import CONDITIONS

# A word character directly before $VAR usually means a missing space.
MISSING_SPACE = re.compile(r"[A-Za-z0-9,]\$[A-Za-z_]")


def target_for(policy: str, tdir: Path) -> Path:
    for f in sorted(tdir.glob("*.yaml")):
        if f.name == "wrappers.yaml":
            continue
        d = yaml.safe_load(open(f)) or {}
        if d.get("policy") == policy:
            return f
    raise SystemExit(f"no template with policy {policy} in {tdir}")


def _quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def merge_text(text: str, family: str, cond: str, new_items: list[str]) -> str:
    """Insert new_items at the end of families.<family>.<cond> in template text."""
    lines = text.splitlines(keepends=True)
    fi = next((i for i, l in enumerate(lines) if l.rstrip("\n") == f"  {family}:"), None)
    if fi is None:
        raise ValueError(f"family {family} not found")
    end = len(lines)
    for j in range(fi + 1, len(lines)):
        l = lines[j]
        if l.strip() and not l.startswith("   "):
            end = j
            break
    ci = next(
        (i for i in range(fi + 1, end)
         if lines[i].startswith(f"    {cond}:")),
        None,
    )
    add = [f"      - {_quote(s)}\n" for s in new_items]
    if ci is None:
        k = end
        while k > fi + 1 and not lines[k - 1].strip():
            k -= 1
        lines[k:k] = [f"    {cond}:\n"] + add
    else:
        lines[ci] = lines[ci].replace(f"{cond}: []", f"{cond}:")
        k = ci + 1
        while k < end and lines[k].startswith("      ") and lines[k].strip():
            k += 1
        lines[k:k] = add
    return "".join(lines)


def merge(incoming: Path, tdir: Path, dry_run: bool = False) -> list[tuple[str, str, int, int]]:
    inc = yaml.safe_load(open(incoming)) or {}
    policy = inc.get("policy")
    if not policy:
        raise SystemExit(f"{incoming}: missing policy")
    target = target_for(policy, tdir)
    cur = yaml.safe_load(open(target)) or {}
    fams = cur.get("families") or {}
    text = target.read_text()
    if not text.endswith("\n"):
        text += "\n"

    report = []
    for fam, conds in (inc.get("families") or {}).items():
        if fam not in fams:
            raise SystemExit(f"{fam}: not a family in {target}. Add the family there first.")
        for cond, items in (conds or {}).items():
            if cond not in CONDITIONS:
                raise SystemExit(f"{fam}: unknown condition {cond}")
            have = set(fams[fam].get(cond) or [])
            new, seen = [], set()
            for n, s in enumerate(items or []):
                s = str(s).strip()
                if MISSING_SPACE.search(s):
                    print(f"  warn {fam}.{cond}[{n}]: possible missing space before $VAR", file=sys.stderr)
                if s in have or s in seen:
                    continue
                seen.add(s)
                new.append(s)
            if new:
                text = merge_text(text, fam, cond, new)
            report.append((fam, cond, len(items or []), len(new)))

    after = (yaml.safe_load(text) or {}).get("families") or {}
    for fam, conds in (inc.get("families") or {}).items():
        for cond, items in (conds or {}).items():
            missing = [s for s in (str(x).strip() for x in items or []) if s not in (after[fam].get(cond) or [])]
            if missing:
                raise SystemExit(f"merge check failed for {fam}.{cond}: {len(missing)} items missing")

    if not dry_run:
        target.write_text(text)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("incoming")
    ap.add_argument("--templates", default="data/templates")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    report = merge(Path(a.incoming), Path(a.templates), a.dry_run)
    verb = "would add" if a.dry_run else "added"
    for fam, cond, total, added in report:
        print(f"  {fam:12s} {cond:11s} {verb} {added} of {total}")


if __name__ == "__main__":
    main()
