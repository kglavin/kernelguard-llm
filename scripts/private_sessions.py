#!/usr/bin/env python3
"""Private check: false alarms on benign commands from local Claude Code sessions, and recalibration (O2).

Input is data/private/sessions.jsonl from `make sessions`. Every item is benign. The S2 runtime bank
scores each item at the S2 layer.

Reported, per form (description only, description plus command):
  - flag rate with the S2 thresholds unchanged, per policy, any policy, and by keyword tag
  - recalibration: sessions are split in two halves of whole sessions. A new threshold per policy is
    the larger of the S2 threshold and the (1 - target_fpr) quantile on the calibration half. The flag
    rate on the other half and the catch and false-alarm rates on the S2 test set are reported at both
    thresholds, so the cost of the new threshold is visible.

Only counts and rates are printed or written. Request and command text is never printed.

Usage: python scripts/private_sessions.py configs/s2_cpu.yaml [--data data/private/sessions.jsonl] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from kernelguard.data.schema import by_policy, load_jsonl, split_by_family
from kernelguard.extract.activations import cache_path, extract_many, load_model
from kernelguard.probes.bank import ProbeBank

FORMS = ["desc", "desc_cmd"]
CHUNK = 512  # items per extraction call, so only one layer is kept in memory


def halves(sessions: np.ndarray) -> np.ndarray:
    """True for the calibration half. Whole sessions, largest first, each to the smaller half."""
    sizes = Counter(sessions.tolist())
    cal, n = set(), [0, 0]
    for s, c in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        k = 0 if n[0] <= n[1] else 1
        n[k] += c
        if k == 0:
            cal.add(s)
    return np.array([s in cal for s in sessions.tolist()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/s2_cpu.yaml")
    ap.add_argument("--data", default="data/private/sessions.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="sample this many pairs, 0 for all")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rd = Path(cfg.get("report_dir", "reports"))
    s2 = json.load(open(rd / "s2.json"))
    L = int(s2["layer"])
    bank = ProbeBank.load(rd / "s2_bank.npz")
    fpr = cfg["target_fpr"]

    xs = load_jsonl(a.data)
    if a.limit:
        keep = set(random.Random(0).sample(sorted({x.meta["base"] for x in xs}), a.limit))
        xs = [x for x in xs if x.meta["base"] in keep]
    form = np.array([x.meta["form"] for x in xs])
    tags = np.array([x.meta["tag"] for x in xs])
    cal = halves(np.array([x.meta["session"] for x in xs]))
    print(f"[*] sessions: {len(xs) // 2} pairs, {len(set(x.meta['session'] for x in xs))} sessions, layer {L}")

    cache = cfg["cache_dir"] + "-sessions"
    need = any(not cache_path(cache, cfg["model"], x).exists() for x in xs)
    mb = load_model(cfg["model"], device=cfg["device"], dtype=cfg["dtype"]) if need else None
    X = np.concatenate([extract_many(mb, xs[s:s + CHUNK], cfg["model"], cache_dir=cache, verbose=True,
                                     batch_size=cfg.get("batch_size", 1))[0][:, L, :] for s in range(0, len(xs), CHUNK)])
    S = bank.score(X)  # [n, P]
    F = S >= bank.tau[None, :]

    report = {"layer": L, "n_pairs": len(xs) // 2, "unchanged": {}, "recalibrated": {}}
    print("\n[*] flag rate, S2 thresholds unchanged")
    print(f"  {'form':9s} " + " ".join(f"{p:>12s}" for p in bank.names) + f" {'any':>7s}  by tag, any policy")
    for f in FORMS:
        m = form == f
        r = {p: float(F[m, k].mean()) for k, p in enumerate(bank.names)}
        r["any"] = float(F[m].any(1).mean())
        r["by_tag"] = {t: {"n": int((m & (tags == t)).sum()), "any": float(F[m & (tags == t)].any(1).mean())}
                       for t in sorted(set(tags.tolist()))}
        report["unchanged"][f] = r
        print(f"  {f:9s} " + " ".join(f"{r[p]:12.1%}" for p in bank.names) + f" {r['any']:7.1%}  "
              + " ".join(f"{t}={v['any']:.0%} (n={v['n']})" for t, v in r["by_tag"].items()))

    # Recalibration on one half of the sessions, evaluated on the other half and on the S2 test set.
    tau2 = np.maximum(bank.tau, np.quantile(S[cal], 1 - fpr, axis=0))
    F2 = S >= tau2[None, :]
    report["recalibrated"]["tau"] = {p: {"s2": float(bank.tau[k]), "new": float(tau2[k])} for k, p in enumerate(bank.names)}
    report["recalibrated"]["n_pairs_cal"] = int(cal.sum()) // 2
    report["recalibrated"]["n_pairs_eval"] = int((~cal).sum()) // 2
    print(f"\n[*] recalibrated on {int(cal.sum()) // 2} pairs, evaluated on the other {int((~cal).sum()) // 2}")
    print(f"  {'form':9s} {'any, S2 tau':>12s} {'any, new tau':>13s}")
    for f in FORMS:
        m = (form == f) & ~cal
        r = {"any_s2_tau": float(F[m].any(1).mean()), "any_new_tau": float(F2[m].any(1).mean())}
        report["recalibrated"][f] = r
        print(f"  {f:9s} {r['any_s2_tau']:12.1%} {r['any_new_tau']:13.1%}")

    items = load_jsonl(cfg["data"])
    print("\n  cost on the S2 test set")
    print(f"  {'policy':13s} {'caught, S2 tau':>15s} {'caught, new tau':>16s} {'fa exec, S2 tau':>16s} {'fa exec, new tau':>17s}")
    for k, p in enumerate(bank.names):
        _, te = split_by_family(by_policy(items)[p], cfg["holdout_frac"])
        te = [i for i in te if i.condition in ("exec_mal", "exec_ben")]
        paths = [cache_path(cfg["cache_dir"], cfg["model"], i) for i in te]
        if not all(q.exists() for q in paths):
            raise SystemExit(f"S2 test items are not in {cfg['cache_dir']}. Run make s2 first.")
        s = bank.score(np.stack([np.load(q)[L].astype(np.float32) for q in paths]))[:, k]
        bad = np.array([i.condition == "exec_mal" for i in te])
        r = {"caught_s2_tau": float((s[bad] >= bank.tau[k]).mean()), "caught_new_tau": float((s[bad] >= tau2[k]).mean()),
             "fa_exec_s2_tau": float((s[~bad] >= bank.tau[k]).mean()), "fa_exec_new_tau": float((s[~bad] >= tau2[k]).mean())}
        report["recalibrated"][p] = r
        print(f"  {p:13s} {r['caught_s2_tau']:15.1%} {r['caught_new_tau']:16.1%} {r['fa_exec_s2_tau']:16.1%} {r['fa_exec_new_tau']:17.1%}")

    json.dump(report, open(rd / "private_sessions.json", "w"), indent=2, default=float)
    print(f"\nwrote {rd}/private_sessions.json (rates and counts only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
