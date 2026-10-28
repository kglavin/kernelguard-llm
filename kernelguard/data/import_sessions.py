"""Import benign execution requests from local Claude Code session files.

Each Bash tool call that carries a description becomes a pair: the description is the request and
the command is what the agent ran. Every pair is benign, since it comes from the user's own
development work. The pairs are written in two forms, which are scored separately:
  desc      the description only, phrased like the dataset's requests
  desc_cmd  the description followed by the command

The output holds private commands and possibly secrets. It goes under data/private/, which is
git-ignored, and is used only for evaluation and calibration, never for training.
Command and description text is never printed.

Usage:
  python -m kernelguard.data.import_sessions [--root ~/.claude/projects] [--out data/private/sessions.jsonl]
                                             [--exclude kernelguard] [--max-chars 600]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from kernelguard.data.schema import Item, save_jsonl

# Keyword tags, checked in order. They only set the policy field and the report breakdown.
TAGS = [
    ("sql", "SQL_MUTATION", re.compile(
        r"\b(psql|sqlite3|mysql|INSERT INTO|UPDATE \w+ SET|DELETE FROM|DROP|ALTER TABLE|CREATE TABLE|TRUNCATE)\b", re.I)),
    ("cred", "CRED_EXFIL", re.compile(
        r"(secret|token|password|passwd|\.env\b|credential|aws configure|gcloud auth|\.ssh/|vault |api[_-]?key)", re.I)),
    ("destructive", "SHELL_EXEC", re.compile(
        r"(\brm -r|\brm -f|\bgit reset --hard|\bgit clean|\bkill\b|\bpkill\b|docker (rm|system prune)|kubectl delete"
        r"|terraform destroy|\bdd\b|chmod -R|mkfs)", re.I)),
]


def tag(cmd: str) -> tuple[str, str]:
    for name, policy, rx in TAGS:
        if rx.search(cmd):
            return name, policy
    return "other", "SHELL_EXEC"


def bash_pairs(path: Path) -> list[tuple[str, str]]:
    """(description, command) for every Bash tool call with a description in one session file."""
    out = []
    for line in open(path, errors="ignore"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        cont = (d.get("message") or {}).get("content")
        if d.get("type") != "assistant" or not isinstance(cont, list):
            continue
        for x in cont:
            if isinstance(x, dict) and x.get("type") == "tool_use" and x.get("name") == "Bash":
                i = x.get("input") or {}
                desc, cmd = (i.get("description") or "").strip(), (i.get("command") or "").strip()
                if desc and cmd:
                    out.append((desc, cmd))
    return out


def build(root: Path, exclude: list[str], max_chars: int) -> tuple[list[Item], Counter]:
    """Items in both forms, deduplicated on (description, command), grouped by top-level session."""
    stats: Counter = Counter()
    seen = set()
    items = []
    for f in sorted(root.glob("*/**/*.jsonl")):
        rel = f.relative_to(root)
        project = rel.parts[0]
        if any(e in project for e in exclude):
            stats["files_excluded"] += 1
            continue
        stats["files"] += 1
        session = hashlib.sha1(f"{project}/{rel.parts[1].removesuffix('.jsonl')}".encode()).hexdigest()[:10]
        for desc, cmd in bash_pairs(f):
            stats["calls"] += 1
            if (desc, cmd) in seen:
                stats["duplicates"] += 1
                continue
            seen.add((desc, cmd))
            if len(cmd) > max_chars:
                stats["too_long"] += 1
                continue
            name, policy = tag(cmd)
            stats[f"tag_{name}"] += 1
            pid = hashlib.sha1(f"{desc}\n{cmd}".encode()).hexdigest()[:12]
            meta = {"base": f"sessions-{pid}", "session": session, "project": project, "tag": name}
            for form, user in (("desc", desc), ("desc_cmd", f"{desc}:\n{cmd}")):
                items.append(Item(id=f"sessions-{pid}-{form}", policy=policy, condition="exec_ben", family="sessions",
                                  user=user, meta={**meta, "form": form}))
    stats["pairs"] = len(items) // 2
    stats["sessions"] = len({i.meta["session"] for i in items})
    return items, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".claude/projects"))
    ap.add_argument("--out", default="data/private/sessions.jsonl")
    ap.add_argument("--exclude", nargs="*", default=["kernelguard"],
                    help="skip projects whose folder name contains any of these")
    ap.add_argument("--max-chars", type=int, default=600, help="drop commands longer than this")
    a = ap.parse_args()
    items, stats = build(Path(a.root).expanduser(), a.exclude, a.max_chars)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(items, out)
    for k in ("files", "files_excluded", "calls", "duplicates", "too_long", "pairs", "sessions"):
        print(f"  {k:15s} {stats[k]}")
    print("  tags           " + " ".join(f"{n}={stats[f'tag_{n}']}" for n, _, _ in TAGS) + f" other={stats['tag_other']}")
    print(f"wrote {len(items)} items to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
