#!/usr/bin/env python3
"""External check: score XSTest with the inbound partition trained in S2.

XSTest (Röttger et al., 2024, CC BY 4.0) has 250 safe prompts that sound alarming, such as homonyms,
figurative language and safe contexts, and 200 unsafe contrast prompts. None of it was seen in training.

Each prompt is placed in the same agent transcript as our requests (system prompt plus tool schema),
so the vectors come from the same runtime regime. The S2 runtime bank scores it at the S2 layer with
the S2 thresholds, unchanged.

Reported:
  - false alarms on safe prompts, per policy and any policy, overall and per XSTest type
  - flag rate on unsafe prompts (XSTest harms are mostly outside our three policies, so this is
    descriptive, not a target)
  - AUROC of unsafe against safe using the highest policy score
  - the TF-IDF baseline from S4, trained on the S2 training split, on the same prompts

Prompt text is never printed.

Usage: python scripts/external_xstest.py configs/s2_cpu.yaml [--csv data/external/xstest/xstest_prompts.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from kernelguard.data.schema import Item, by_policy, load_jsonl, split_by_family
from kernelguard.eval.metrics import auroc
from kernelguard.extract.activations import extract_many, load_model, cache_path
from kernelguard.probes.bank import ProbeBank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/s2_cpu.yaml")
    ap.add_argument("--csv", default="data/external/xstest/xstest_prompts.csv")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    rd = Path(cfg.get("report_dir", "reports"))
    s2 = json.load(open(rd / "s2.json"))
    L = int(s2["layer"])
    bank = ProbeBank.load(rd / "s2_bank.npz")

    rows = list(csv.DictReader(open(a.csv, newline="")))
    xs = [Item(id=f"xstest-{r['id']}", policy="SHELL_EXEC", condition="discuss_ben", family="xstest",
               user=r["prompt"].strip(), meta={"label": r["label"], "type": r["type"]}) for r in rows]
    unsafe = np.array([x.meta["label"] == "unsafe" for x in xs])
    types = np.array([x.meta["type"] for x in xs])
    print(f"[*] XSTest: {len(xs)} prompts, {int((~unsafe).sum())} safe, {int(unsafe.sum())} unsafe, layer {L}")

    cache = cfg["cache_dir"] + "-xstest"
    need = any(not cache_path(cache, cfg["model"], x).exists() for x in xs)
    mb = load_model(cfg["model"], device=cfg["device"], dtype=cfg["dtype"]) if need else None
    acts, _ = extract_many(mb, xs, cfg["model"], cache_dir=cache, verbose=True, batch_size=cfg.get("batch_size", 1))
    S = bank.score(acts[:, L, :])  # [n, P]
    F = S >= bank.tau[None, :]
    anyF = F.any(1)
    report = {"layer": L, "n_safe": int((~unsafe).sum()), "n_unsafe": int(unsafe.sum()), "vectors": {}, "text": {}}

    print("\n[*] vector partition, S2 thresholds unchanged")
    print(f"  {'policy':13s} {'false alarms on safe':>20s} {'flagged unsafe':>15s}")
    for k, p in enumerate(bank.names):
        fa, fu = float(F[~unsafe, k].mean()), float(F[unsafe, k].mean())
        report["vectors"][p] = {"false_alarm_safe": fa, "flag_unsafe": fu}
        print(f"  {p:13s} {fa:20.1%} {fu:15.1%}")
    report["vectors"]["any"] = {"false_alarm_safe": float(anyF[~unsafe].mean()), "flag_unsafe": float(anyF[unsafe].mean()),
                                "auroc_unsafe_vs_safe_max_score": auroc(unsafe.astype(int), S.max(1))}
    v = report["vectors"]["any"]
    print(f"  {'any policy':13s} {v['false_alarm_safe']:20.1%} {v['flag_unsafe']:15.1%}")
    print(f"  AUROC unsafe vs safe, highest policy score: {v['auroc_unsafe_vs_safe_max_score']:.3f}")

    by_type = defaultdict(dict)
    for t in sorted(set(types[~unsafe].tolist())):
        m = (types == t) & ~unsafe
        by_type[t] = float(anyF[m].mean())
    report["vectors"]["false_alarm_safe_by_type"] = by_type
    print("\n  false alarms on safe prompts by XSTest type, any policy")
    for t, r in by_type.items():
        print(f"    {t:24s} {r:6.0%}")

    # TF-IDF baseline trained on the S2 training split, per policy, threshold at target_fpr.
    items = load_jsonl(cfg["data"])
    fpr = cfg["target_fpr"]
    T = np.zeros_like(F)
    for k, p in enumerate(bank.names):
        tr, _ = split_by_family(by_policy(items)[p], cfg["holdout_frac"])
        ytr = np.array([i.label for i in tr])
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, lowercase=True)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(vec.fit_transform([i.user for i in tr]), ytr)
        s_tr = clf.decision_function(vec.transform([i.user for i in tr]))
        tau = float(np.quantile(s_tr[ytr == 0], 1 - fpr))
        T[:, k] = clf.decision_function(vec.transform([x.user for x in xs])) >= tau
        report["text"][p] = {"false_alarm_safe": float(T[~unsafe, k].mean()), "flag_unsafe": float(T[unsafe, k].mean())}
    anyT = T.any(1)
    report["text"]["any"] = {"false_alarm_safe": float(anyT[~unsafe].mean()), "flag_unsafe": float(anyT[unsafe].mean())}
    print("\n[*] TF-IDF baseline (word 1-2 grams), trained on the S2 training split")
    print(f"  {'policy':13s} {'false alarms on safe':>20s} {'flagged unsafe':>15s}")
    for p, r in report["text"].items():
        print(f"  {p:13s} {r['false_alarm_safe']:20.1%} {r['flag_unsafe']:15.1%}")

    json.dump(report, open(rd / "external_xstest.json", "w"), indent=2, default=float)
    print(f"\nwrote {rd}/external_xstest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
