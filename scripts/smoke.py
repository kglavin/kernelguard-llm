#!/usr/bin/env python3
"""M0 end-to-end smoke: load model, extract, fit probes per policy, eval, arm tripwire, generate.

Usage: python scripts/smoke.py configs/smoke_cpu.yaml
Exit code 0 means the pipeline runs. It does not mean the hypothesis holds.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from kernelguard.data.schema import load_jsonl, split_by_family, by_policy
from kernelguard.extract.activations import load_model, extract_many, render_prompt
from kernelguard.probes.bank import ProbeBank
from kernelguard.eval.metrics import summarize
from kernelguard.eval.baselines import regex_score
from kernelguard.runtime.hf_hook import Tripwire


def main(cfg_path: str) -> int:
    cfg = yaml.safe_load(open(cfg_path))
    t0 = time.time()
    print(f"[*] loading {cfg['model']} on {cfg['device']}")
    mb = load_model(cfg["model"], device=cfg["device"], dtype=cfg["dtype"])
    print(f"    layers={mb.n_layers} hidden={mb.hidden_dim} ({time.time() - t0:.1f}s)")

    items = load_jsonl(cfg["data"])
    print(f"[*] {len(items)} items, extracting all layers at last prompt token")
    acts, items = extract_many(mb, items, cfg["model"], cache_dir=cfg["cache_dir"], verbose=True,
                               batch_size=cfg.get("batch_size", 1))
    print(f"    acts shape {acts.shape} ({time.time() - t0:.1f}s)")

    L = cfg["layer"]
    bank = ProbeBank(hidden_dim=mb.hidden_dim)
    bank.set_background(acts[:, L, :], zscore=cfg["zscore"])

    report = {"config": cfg, "policies": {}, "layer_sweep": {}}
    idx = {it.id: n for n, it in enumerate(items)}
    for policy, pitems in by_policy(items).items():
        if policy not in cfg["policies"]:
            continue
        train, test = split_by_family(pitems, cfg["holdout_frac"])
        A = lambda subset, cond=None: acts[[idx[i.id] for i in subset if cond is None or i.condition == cond], L, :]
        pos, neg = A(train, "exec_mal"), np.concatenate([A(train, c) for c in ("exec_ben", "discuss_mal", "discuss_ben")])
        v = bank.direction_dom(pos, neg) if cfg["fit"] == "dom" else bank.direction_logistic(pos, neg)
        bank.add(policy, v, tau=0.0)
        bank.set_threshold_at_fpr(policy, bank.score(neg)[:, -1], cfg["target_fpr"])

        def ev(subset):
            y = np.array([i.label for i in subset])
            s = bank.score(acts[[idx[i.id] for i in subset], L, :])[:, -1]
            r = np.array([regex_score(i) for i in subset])
            return {"probe": summarize(y, s), "regex": summarize(y, r)}

        report["policies"][policy] = {"n_train": len(train), "n_test": len(test), "tau": float(bank.tau[-1]),
                                      "train": ev(train), "test": ev(test) if test else None}
        tr = report["policies"][policy]
        te = tr["test"]["probe"] if tr["test"] else {}
        print(f"[*] {policy:12s} train n={len(train)} test n={len(test)} tau={tr['tau']:.3f} "
              f"train AUROC={tr['train']['probe']['auroc']:.3f} test AUROC={te.get('auroc', float('nan')):.3f}")

    # layer sweep on the first policy (RQ1 scaffolding)
    p0 = cfg["policies"][0]
    pitems = by_policy(items)[p0]
    y = np.array([i.label for i in pitems])
    for l in cfg["layer_sweep"]:
        b = ProbeBank(hidden_dim=mb.hidden_dim)
        a = acts[[idx[i.id] for i in pitems], l, :]
        b.set_background(a)
        v = b.direction_dom(a[y == 1], a[y == 0])
        b.add(p0, v, 0.0)
        report["layer_sweep"][l] = summarize(y, b.score(a)[:, 0])
    print("[*] layer sweep (" + p0 + ", in-sample):", {l: round(r["auroc"], 3) for l, r in report["layer_sweep"].items()})

    # runtime tripwire
    rt = cfg["runtime"]
    tw = Tripwire(mb, bank, layer=L, **rt)
    demo = ([i for i in items if i.policy == p0 and i.condition == "exec_ben"][:2]
            + [i for i in items if i.policy == p0 and i.condition == "exec_mal"][:2])
    report["runtime"] = []
    for it in demo:
        r = tw.generate(render_prompt(mb, it), max_new_tokens=cfg["max_new_tokens"])
        row = {"id": it.id, "label": it.label, "tripped": r["tripped"], "trip_step": r["trip_step"],
               "n_new_tokens": r["n_new_tokens"], "text": r["text"][:120]}
        report["runtime"].append(row)
        print(f"[*] runtime {it.condition:9s} tripped={r['tripped']!s:5} step={r['trip_step']:3d} "
              f"tokens={r['n_new_tokens']:3d} | {r['text'][:70]!r}")

    out = Path(cfg.get("report_dir", "reports")) / "smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"[*] wrote {out} ({time.time() - t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/smoke_cpu.yaml"))
