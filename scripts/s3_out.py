#!/usr/bin/env python3
"""S3: vectors on the way out.

1. For one phrasing of each distinct request (wrapper 0), generate a greedy response.
2. Re-read prompt + response once. Keep the mean residual vector over response tokens for every layer,
   and the per-token vectors at the runtime layer.
3. Label each response. Bad means the request was exec_mal and the model acted on it with a tool call.
   Everything else is good, including refusals of bad requests and benign tool calls.
4. Fit an outbound partition per policy on the pooled vectors, with the same family split as S2.
5. Calibrate two gates and report trip rates on held-out responses. In both, tau is set on the
   training benign responses at target_fpr.
     token  a response trips if any single token scores above tau
     mean   a response trips if the running mean of its tokens so far ever scores above tau
6. Runtime demo: generate with the S2 inbound bank at prefill and the outbound bank on decode steps,
   using the gate chosen with --gate.

Response texts are cached locally and written to reports/s3_responses.jsonl. The script never prints them.

Usage: python scripts/s3_out.py configs/s2_cpu.yaml [--limit N] [--max-new-tokens 48] [--gate mean|token] [--no-demo]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from kernelguard.data.schema import by_policy, load_jsonl, split_by_family
from kernelguard.eval.metrics import auroc, bootstrap_ci
from kernelguard.extract.activations import _cache_key, load_model, render_prompt
from kernelguard.probes.bank import ProbeBank
from kernelguard.runtime.hf_hook import Tripwire

TOOL_TAG = "<tool_call>"


@torch.no_grad()
def respond(mb, item, max_new_tokens: int, layer: int):
    tok, model = mb.tokenizer, mb.model
    enc = tok(render_prompt(mb, item), return_tensors="pt").to(mb.device)
    n_prompt = enc.input_ids.shape[1]
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    e = model.generation_config.eos_token_id
    eos = set(e if isinstance(e, (list, tuple)) else [e]) | {tok.eos_token_id}
    keep = []
    for t in out[0, n_prompt:].tolist():
        if t in eos:
            break
        keep.append(t)
    text = tok.decode(keep, skip_special_tokens=False)
    meta = {"text": text, "n": len(keep), "tool_call": TOOL_TAG in text}
    if not keep:
        return meta, None, None
    full = torch.tensor([out[0, :n_prompt].tolist() + keep], device=mb.device)
    hs = model(full, output_hidden_states=True).hidden_states
    H = torch.stack([h[0, n_prompt:, :] for h in hs])  # [N+1, R, d]
    return meta, H.mean(1).float().cpu().numpy().astype(np.float16), H[layer].float().cpu().numpy().astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/s2_cpu.yaml")
    ap.add_argument("--layer", type=int, default=None, help="default: layer chosen by s2_eval")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0, help="only the first N requests, for timing")
    ap.add_argument("--gate", choices=["mean", "token"], default="mean", help="gate used for the runtime demo")
    ap.add_argument("--no-demo", action="store_true")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    rd = Path(cfg.get("report_dir", "reports"))
    s2 = json.load(open(rd / "s2.json"))
    L = a.layer or int(s2["layer"])
    G = a.max_new_tokens
    fpr = cfg["target_fpr"]
    items = [i for i in load_jsonl(cfg["data"]) if i.meta.get("wrapper") == 0]
    if a.limit:
        items = items[: a.limit]
    print(f"[*] {len(items)} requests, layer {L}, up to {G} new tokens")

    cache = Path(cfg["cache_dir"] + "-s3")
    cache.mkdir(parents=True, exist_ok=True)
    path_of = lambda it: cache / f"{_cache_key(cfg['model'], it)}-g{G}-L{L}.npz"
    need_model = (not a.no_demo) or any(not path_of(it).exists() for it in items)
    mb = load_model(cfg["model"], device=cfg["device"], dtype=cfg["dtype"]) if need_model else None

    recs, means, toks = [], [], []
    t0, fresh = time.time(), 0
    for k, it in enumerate(items):
        p = path_of(it)
        if p.exists():
            z = np.load(p, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            m, tk = (z["mean"], z["toks"]) if meta["n"] else (None, None)
        else:
            meta, m, tk = respond(mb, it, G, L)
            np.savez(p, meta=np.array(json.dumps(meta)),
                     mean=m if m is not None else np.zeros(0, np.float16),
                     toks=tk if tk is not None else np.zeros(0, np.float16))
            fresh += 1
        recs.append({"id": it.id, "policy": it.policy, "condition": it.condition, "family": it.family,
                     "n": meta["n"], "tool_call": meta["tool_call"],
                     "bad": int(it.condition == "exec_mal" and meta["tool_call"]), "text": meta["text"]})
        means.append(m)
        toks.append(tk)
        if (k + 1) % 20 == 0 or k + 1 == len(items):
            rate = (time.time() - t0) / max(1, fresh)
            print(f"  {k + 1}/{len(items)}  {fresh} generated, {rate:.1f}s each", flush=True)

    with open(rd / "s3_responses.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    print("\n[*] response behaviour (counts only)")
    for pol in sorted({r["policy"] for r in recs}):
        for c in ["exec_mal", "exec_ben", "discuss_mal", "discuss_ben"]:
            rs = [r for r in recs if r["policy"] == pol and r["condition"] == c]
            tc = sum(r["tool_call"] for r in rs)
            print(f"  {pol:13s} {c:11s} n={len(rs):3d} tool calls={tc:3d}")

    keep = [k for k, r in enumerate(recs) if r["n"] > 0]
    dropped = len(recs) - len(keep)
    if dropped:
        print(f"  {dropped} empty responses excluded")
    pos = {recs[k]["id"]: j for j, k in enumerate(keep)}
    R = [recs[k] for k in keep]
    M = np.stack([means[k] for k in keep]).astype(np.float32)[:, L, :]
    T = [toks[k].astype(np.float32) for k in keep]
    y = np.array([r["bad"] for r in R])
    cond = np.array([r["condition"] for r in R])
    tool = np.array([r["tool_call"] for r in R])

    kept_items = [it for it in items if it.id in pos]
    splits = {}
    for pol, its in sorted(by_policy(kept_items).items()):
        tr, te = split_by_family(its, cfg["holdout_frac"])
        splits[pol] = (np.array([pos[i.id] for i in tr], dtype=int), np.array([pos[i.id] for i in te], dtype=int))
        s2_fams = s2["policies"].get(pol, {}).get("test_families")
        if s2_fams and sorted({i.family for i in te}) != s2_fams:
            print(f"  warning: {pol} test families differ from S2")
    train_all = np.concatenate([tr for tr, _ in splits.values()])

    bank = ProbeBank(M.shape[1])
    bank.set_background(np.concatenate([T[j] for j in train_all]))
    report = {"config": cfg, "layer": L, "max_new_tokens": G, "demo_gate": a.gate, "policies": {}}
    taus = {"token": {}, "mean": {}}
    print(f"\n[*] outbound partition at layer {L}, gate thresholds at {fpr:.0%} false alarms on training benign responses")
    print(f"{'policy':13s} {'bad test':>8s} {'AUROC pooled':>20s} {'AUROC tool calls':>17s}   "
          f"{'token gate: catch / fa tool / fa other':>38s}   {'mean gate: catch / fa tool / fa other':>37s}")
    for pol, (tr, te) in splits.items():
        if len(tr) == 0 or len(te) == 0 or len(set(y[tr].tolist())) < 2 or len(set(y[te].tolist())) < 2:
            print(f"{pol:13s} skipped: needs bad and good responses on both sides of the split")
            continue
        v = bank.direction_logistic(M[tr][y[tr] == 1], M[tr][y[tr] == 0])
        bank.add(pol, v, 0.0)
        s_te = bank.score(M[te])[:, -1]
        per_tok = [bank.score(T[j])[:, -1] for j in range(len(T))]
        run_mean = [bank.score(np.cumsum(T[j], 0) / np.arange(1, len(T[j]) + 1)[:, None])[:, -1] for j in range(len(T))]
        yt, tt = y[te], tool[te]
        tcm = tt.astype(bool)
        r = {
            "test_families": sorted(set(R[j]["family"] for j in te)),
            "n_test": int(len(te)), "n_test_bad": int(yt.sum()),
            "auroc_pooled": auroc(yt, s_te),
            "auroc_pooled_ci": bootstrap_ci(auroc, yt, s_te),
            "auroc_pooled_tool_calls_only": auroc(yt[tcm], s_te[tcm]),
            "gates": {},
        }
        for g, seq in (("token", per_tok), ("mean", run_mean)):
            mx = np.array([v.max() for v in seq])
            tau = float(np.quantile(mx[tr][y[tr] == 0], 1.0 - fpr))
            taus[g][pol] = tau
            trip = mx[te] >= tau
            r["gates"][g] = {
                "tau": tau,
                "trip_bad": float(trip[yt == 1].mean()),
                "trip_benign_tool_call": float(trip[(yt == 0) & tcm].mean()) if ((yt == 0) & tcm).any() else float("nan"),
                "trip_other_good": float(trip[(yt == 0) & ~tcm].mean()) if ((yt == 0) & ~tcm).any() else float("nan"),
            }
        report["policies"][pol] = r
        ci = r["auroc_pooled_ci"]
        gt, gm = r["gates"]["token"], r["gates"]["mean"]
        print(f"{pol:13s} {r['n_test_bad']:8d} {r['auroc_pooled']:.3f} [{ci[0]:.2f},{ci[1]:.2f}] "
              f"{r['auroc_pooled_tool_calls_only']:17.3f}   "
              f"{gt['trip_bad']:14.0%} {gt['trip_benign_tool_call']:9.0%} {gt['trip_other_good']:12.0%}   "
              f"{gm['trip_bad']:13.0%} {gm['trip_benign_tool_call']:9.0%} {gm['trip_other_good']:12.0%}")

    for pol in bank.names:
        bank.tau[bank.names.index(pol)] = taus[a.gate][pol]
    bank.save(rd / "s3_bank.npz")

    # Runtime demo: inbound bank at prefill, outbound bank on decode steps, plain s >= tau on each.
    bank_in = ProbeBank.load(rd / "s2_bank.npz")
    if a.no_demo:
        print("\n[*] runtime demo skipped (--no-demo)")
    elif int(s2["layer"]) == L and bank_in.names == bank.names:
        tw = Tripwire(mb, bank_in, layer=L, alpha=1.0, k_sustained=1, crit_margin=10.0, bank_out=bank, out_mode=a.gate)
        demo = []
        for pol, (_, te) in splits.items():
            for c in ["exec_ben", "exec_mal", "discuss_mal"]:
                demo += [j for j in te if R[j]["condition"] == c][:1]
        print(f"\n[*] runtime demo on held-out requests (inbound at prefill, outbound {a.gate} gate on generated tokens)")
        rows = []
        for j in demo:
            it = next(i for i in kept_items if i.id == R[j]["id"])
            g = tw.generate(render_prompt(mb, it), max_new_tokens=G)
            stage = "prefill" if g["trip_step"] == 0 else ("decode" if g["tripped"] else "-")
            rows.append({"id": it.id, "condition": it.condition, "unguarded_tool_call": bool(R[j]["tool_call"]),
                         "tripped": g["tripped"], "stage": stage, "trip_step": g["trip_step"], "n_new_tokens": g["n_new_tokens"]})
            print(f"  {it.policy:13s} {it.condition:11s} unguarded tool call={str(R[j]['tool_call']):5s} "
                  f"tripped={str(g['tripped']):5s} stage={stage:7s} step={g['trip_step']:3d} tokens={g['n_new_tokens']:3d}")
        report["runtime_demo"] = rows
    else:
        print("\n[*] runtime demo skipped: S2 bank layer or policies differ from this run")

    json.dump(report, open(rd / "s3.json", "w"), indent=2, default=float)
    print(f"\nwrote {rd}/s3.json, s3_bank.npz, s3_responses.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
