#!/usr/bin/env python3
"""S2 re-score from the activation cache. No model load, no extraction.

- Family holdout from the config (0.5 holds out two families per policy).
- Scores bad vs all good, and bad vs benign execution only.
- Chooses one layer by leave-one-family-out over the training families. The test set is never used to pick it.
- Threshold is set on training negatives at target_fpr, then applied unchanged to the test set.
- Confidence intervals resample whole requests, so wrapper variants of one request move together.
- Saves the inbound probe bank at the chosen layer for the runtime gate.

Usage: python scripts/s2_eval.py configs/s2_cpu.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

from kernelguard.data.schema import by_policy, load_jsonl, split_by_family
from kernelguard.eval.metrics import auroc, tpr_at_fpr
from kernelguard.extract.activations import cache_path
from kernelguard.probes.bank import ProbeBank

CONDS = ["exec_mal", "exec_ben", "discuss_mal", "discuss_ben"]


def load_acts(items, model, cache_dir) -> np.ndarray:
    paths = [cache_path(cache_dir, model, it) for it in items]
    miss = sum(not p.exists() for p in paths)
    if miss:
        raise SystemExit(f"{miss} of {len(items)} items are not in {cache_dir}. Run make s2 first.")
    return np.stack([np.load(p) for p in paths])  # fp16 [n, N+1, d]


def fit_direction(bg: np.ndarray, X: np.ndarray, y: np.ndarray) -> ProbeBank:
    b = ProbeBank(X.shape[1])
    b.set_background(bg)
    b.add("p", b.direction_logistic(X[y == 1], X[y == 0]), 0.0)
    return b


def cluster_ci(y, s, groups, n=500, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    where = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n):
        b = np.concatenate([where[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        if len(set(y[b].tolist())) == 2:
            vals.append(auroc(y[b], s[b]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main(cfg_path: str) -> int:
    cfg = yaml.safe_load(open(cfg_path))
    items = load_jsonl(cfg["data"])
    A = load_acts(items, cfg["model"], cfg["cache_dir"])
    layers = list(range(1, A.shape[1]))
    frac, fpr = cfg["holdout_frac"], cfg["target_fpr"]

    y = np.array([i.label for i in items])
    cond = np.array([i.condition for i in items])
    fam = np.array([i.family for i in items])
    base = np.array([i.meta.get("base", i.id) for i in items])
    pos = {it.id: k for k, it in enumerate(items)}

    splits = {}
    for p, its in sorted(by_policy(items).items()):
        tr, te = split_by_family(its, frac)
        splits[p] = (np.array([pos[i.id] for i in tr]), np.array([pos[i.id] for i in te]))
    train_all = np.concatenate([tr for tr, _ in splits.values()])

    # Layer choice: leave one training family out, average AUROC, never touch the test families.
    lofo: dict[int, dict[str, float]] = {}
    test_by_layer: dict[int, dict[str, float]] = {}
    for L in layers:
        X = A[:, L, :].astype(np.float32)
        lofo[L], test_by_layer[L] = {}, {}
        for p, (tr, te) in splits.items():
            fold = []
            for f in sorted(set(fam[tr])):
                a, b = tr[fam[tr] != f], tr[fam[tr] == f]
                bank = fit_direction(X[a], X[a], y[a])
                fold.append(auroc(y[b], bank.score(X[b])[:, 0]))
            lofo[L][p] = float(np.mean(fold))
            bank = fit_direction(X[train_all], X[tr], y[tr])
            test_by_layer[L][p] = auroc(y[te], bank.score(X[te])[:, 0])
        print(f"  layer {L:2d}  lofo " + " ".join(f"{p[:5]}={v:.3f}" for p, v in lofo[L].items()), flush=True)
    best = max(layers, key=lambda L: np.mean(list(lofo[L].values())))

    # Final evaluation at the chosen layer, plus the runtime bank.
    X = A[:, best, :].astype(np.float32)
    bank_rt = ProbeBank(X.shape[1])
    bank_rt.set_background(X[train_all])
    report = {"config": cfg, "layer": best, "lofo": lofo, "test_auroc_by_layer": test_by_layer, "policies": {}}
    for p, (tr, te) in splits.items():
        v = bank_rt.direction_logistic(X[tr][y[tr] == 1], X[tr][y[tr] == 0])
        bank_rt.add(p, v, 0.0)
        s_tr = bank_rt.score(X[tr])[:, -1]
        s_te = bank_rt.score(X[te])[:, -1]
        tau = bank_rt.set_threshold_at_fpr(p, s_tr[y[tr] == 0], fpr)
        yt, ct, bt = y[te], cond[te], base[te]
        ex = np.isin(ct, ["exec_mal", "exec_ben"])
        report["policies"][p] = {
            "test_families": sorted(set(fam[te].tolist())),
            "n_test": int(len(te)),
            "n_test_requests": int(len(set(bt.tolist()))),
            "n_test_bad_requests": int(len(set(bt[yt == 1].tolist()))),
            "auroc_all": auroc(yt, s_te),
            "auroc_all_ci": cluster_ci(yt, s_te, bt),
            "auroc_exec": auroc(yt[ex], s_te[ex]),
            "auroc_exec_ci": cluster_ci(yt[ex], s_te[ex], bt[ex]),
            "tpr_at_test_fpr1%": tpr_at_fpr(yt, s_te, 0.01),
            "tau": tau,
            "flag_rate_at_tau": {c: float((s_te[ct == c] >= tau).mean()) for c in CONDS},
        }

    out = Path(cfg.get("report_dir", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    bank_rt.save(out / "s2_bank.npz")
    json.dump(report, open(out / "s2.json", "w"), indent=2, default=float)

    print(f"\nchosen layer {best} (leave-one-family-out on training families)")
    print(f"threshold set on training negatives at {fpr:.0%} false alarms, applied unchanged to test\n")
    hdr = f"{'policy':13s} {'test families':26s} {'bad req':>7s} {'AUROC all':>18s} {'AUROC exec':>18s} {'caught':>7s} {'fa exec':>7s} {'fa disc':>7s}"
    print(hdr)
    for p, r in report["policies"].items():
        f = r["flag_rate_at_tau"]
        ci_a, ci_e = r["auroc_all_ci"], r["auroc_exec_ci"]
        print(f"{p:13s} {','.join(r['test_families']):26s} {r['n_test_bad_requests']:7d} "
              f"{r['auroc_all']:.3f} [{ci_a[0]:.2f},{ci_a[1]:.2f}] {r['auroc_exec']:.3f} [{ci_e[0]:.2f},{ci_e[1]:.2f}] "
              f"{f['exec_mal']:7.0%} {f['exec_ben']:7.0%} {f['discuss_mal']:7.0%}")
    print("\ntest AUROC by layer (reference only, not used for the choice):")
    for L in layers:
        print(f"  {L:2d} " + " ".join(f"{p[:5]}={v:.3f}" for p, v in test_by_layer[L].items()))
    print(f"\nwrote {out}/s2.json and {out}/s2_bank.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/s2_cpu.yaml"))
