"""Metrics (F6): AUROC, TPR at fixed FPR, with bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float) -> float:
    """Largest TPR achievable with FPR <= target."""
    if len(set(y.tolist())) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def bootstrap_ci(fn, y: np.ndarray, s: np.ndarray, n: int = 500, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        v = fn(y[b], s[b])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def summarize(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "auroc": auroc(y, s),
        "tpr@fpr1%": tpr_at_fpr(y, s, 0.01),
        "tpr@fpr0.1%": tpr_at_fpr(y, s, 0.001),
    }
