#!/usr/bin/env python3
"""S4: text baseline on the same splits as S2 and S3.

A TF-IDF logistic classifier reads the text instead of the model's vectors:
  inbound   request text, all wrapper variants, S2 family split
  outbound  response text, S3 responses, S3 family split
Two feature sets are tried: word 1-2 grams and character 3-5 grams. Thresholds are set on training
negatives at target_fpr, the same way as the vector partitions.

Usage: python scripts/s4_text.py configs/s2_cpu.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from kernelguard.data.schema import by_policy, load_jsonl, split_by_family
from kernelguard.eval.metrics import auroc

FEATURES = {
    "word_1_2": dict(analyzer="word", ngram_range=(1, 2), min_df=2, lowercase=True),
    "char_3_5": dict(analyzer="char_wb", ngram_range=(3, 5), min_df=2, lowercase=True),
}


def fit_score(texts_tr, y_tr, texts_te, feat):
    vec = TfidfVectorizer(**FEATURES[feat])
    Xtr = vec.fit_transform(texts_tr)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(Xtr, y_tr)
    return clf.decision_function(Xtr), clf.decision_function(vec.transform(texts_te))


def main(cfg_path: str) -> int:
    cfg = yaml.safe_load(open(cfg_path))
    frac, fpr = cfg["holdout_frac"], cfg["target_fpr"]
    rd = Path(cfg.get("report_dir", "reports"))
    s2 = json.load(open(rd / "s2.json"))
    s3 = json.load(open(rd / "s3.json"))
    report = {"inbound": {}, "outbound": {}}

    # Inbound: request text
    items = load_jsonl(cfg["data"])
    print(f"[*] inbound, request text, threshold at {fpr:.0%} false alarms on training negatives")
    print(f"{'policy':13s} {'features':9s} {'AUROC all':>9s} {'AUROC exec':>10s} {'caught':>7s} {'fa exec':>7s} {'fa disc':>7s}"
          f"   vectors: AUROC all / exec")
    for pol, its in sorted(by_policy(items).items()):
        tr, te = split_by_family(its, frac)
        ytr = np.array([i.label for i in tr]); yte = np.array([i.label for i in te])
        cte = np.array([i.condition for i in te])
        ex = np.isin(cte, ["exec_mal", "exec_ben"])
        report["inbound"][pol] = {}
        for feat in FEATURES:
            s_tr, s_te = fit_score([i.user for i in tr], ytr, [i.user for i in te], feat)
            tau = float(np.quantile(s_tr[ytr == 0], 1 - fpr))
            flag = s_te >= tau
            r = {"auroc_all": auroc(yte, s_te), "auroc_exec": auroc(yte[ex], s_te[ex]),
                 "caught": float(flag[cte == "exec_mal"].mean()),
                 "fa_exec_ben": float(flag[cte == "exec_ben"].mean()),
                 "fa_discuss": float(flag[np.isin(cte, ["discuss_mal", "discuss_ben"])].mean())}
            report["inbound"][pol][feat] = r
            v = s2["policies"][pol]
            print(f"{pol:13s} {feat:9s} {r['auroc_all']:9.3f} {r['auroc_exec']:10.3f} {r['caught']:7.0%} "
                  f"{r['fa_exec_ben']:7.0%} {r['fa_discuss']:7.0%}   {v['auroc_all']:.3f} / {v['auroc_exec']:.3f}")

    # Outbound: response text
    recs = [json.loads(l) for l in open(rd / "s3_responses.jsonl")]
    recs = [r for r in recs if r["n"] > 0]
    by_id = {it.id: it for it in items}
    print(f"\n[*] outbound, response text, threshold at {fpr:.0%} false alarms on training benign responses")
    print(f"{'policy':13s} {'features':9s} {'AUROC':>7s} {'AUROC tool calls':>16s} {'caught':>7s} {'fa tool':>7s}"
          f"   vectors: AUROC / tool calls")
    for pol in sorted({r["policy"] for r in recs}):
        rs = [r for r in recs if r["policy"] == pol]
        its = [by_id[r["id"]] for r in rs]
        tr_items, te_items = split_by_family(its, frac)
        tr_ids, te_ids = {i.id for i in tr_items}, {i.id for i in te_items}
        tr = [r for r in rs if r["id"] in tr_ids]; te = [r for r in rs if r["id"] in te_ids]
        ytr = np.array([r["bad"] for r in tr]); yte = np.array([r["bad"] for r in te])
        tool = np.array([r["tool_call"] for r in te], dtype=bool)
        if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
            continue
        report["outbound"][pol] = {}
        for feat in FEATURES:
            s_tr, s_te = fit_score([r["text"] for r in tr], ytr, [r["text"] for r in te], feat)
            tau = float(np.quantile(s_tr[ytr == 0], 1 - fpr))
            flag = s_te >= tau
            r = {"auroc": auroc(yte, s_te), "auroc_tool_calls": auroc(yte[tool], s_te[tool]),
                 "caught": float(flag[yte == 1].mean()),
                 "fa_benign_tool_call": float(flag[(yte == 0) & tool].mean()) if ((yte == 0) & tool).any() else float("nan")}
            report["outbound"][pol][feat] = r
            v = s3["policies"][pol]
            print(f"{pol:13s} {feat:9s} {r['auroc']:7.3f} {r['auroc_tool_calls']:16.3f} {r['caught']:7.0%} "
                  f"{r['fa_benign_tool_call']:7.0%}   {v['auroc_pooled']:.3f} / {v['auroc_pooled_tool_calls_only']:.3f}")

    json.dump(report, open(rd / "s4.json", "w"), indent=2, default=float)
    print(f"\nwrote {rd}/s4.json")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "configs/s2_cpu.yaml"))
