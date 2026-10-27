"""Probe bank (P4): centered, unit-normalized directions scored by cosine, batched as one GEMM.

Fit methods:
  dom       difference of means (mu_mal - mu_ben)
  logistic  L2 logistic regression on centered unit vectors; weight vector used as direction
Optional: per-dimension z-scoring (massive-activation dims), topic orthogonalization (RQ2).
Thresholds are set on a validation split at a target FPR, never at the midpoint of extremes (F6).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


@dataclass
class ProbeBank:
    hidden_dim: int
    mu: np.ndarray = None  # [d] centering mean
    sigma: np.ndarray = None  # [d] per-dim scale (ones if z-scoring disabled)
    names: list[str] = field(default_factory=list)
    W: np.ndarray = None  # [P, d] unit directions
    tau: np.ndarray = None  # [P] cosine thresholds

    def __post_init__(self):
        if self.mu is None:
            self.mu = np.zeros(self.hidden_dim, dtype=np.float32)
        if self.sigma is None:
            self.sigma = np.ones(self.hidden_dim, dtype=np.float32)
        if self.W is None:
            self.W = np.zeros((0, self.hidden_dim), dtype=np.float32)
        if self.tau is None:
            self.tau = np.zeros((0,), dtype=np.float32)

    # ---- preprocessing -------------------------------------------------
    def set_background(self, acts: np.ndarray, zscore: bool = False) -> None:
        """acts: [n, d] from all calibration items (both classes)."""
        self.mu = acts.mean(0).astype(np.float32)
        if zscore:
            self.sigma = (acts.std(0) + 1e-6).astype(np.float32)
        else:
            self.sigma = np.ones(self.hidden_dim, dtype=np.float32)

    def features(self, acts: np.ndarray) -> np.ndarray:
        """Center, scale, unit-normalize. acts: [n, d] -> [n, d]."""
        return _unit((acts - self.mu) / self.sigma)

    # ---- directions ----------------------------------------------------
    def direction_dom(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        return _unit(self.features(pos).mean(0) - self.features(neg).mean(0))

    def direction_logistic(self, pos: np.ndarray, neg: np.ndarray, C: float = 1.0) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression

        X = np.concatenate([self.features(pos), self.features(neg)])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        clf = LogisticRegression(C=C, max_iter=2000, fit_intercept=False)
        clf.fit(X, y)
        return _unit(clf.coef_[0].astype(np.float32))

    @staticmethod
    def orthogonalize(v: np.ndarray, v_topic: np.ndarray) -> np.ndarray:
        """Gram-Schmidt: remove the topic component from v (RQ2)."""
        return _unit(v - np.dot(v, v_topic) * v_topic)

    # ---- registration ---------------------------------------------------
    def add(self, name: str, v: np.ndarray, tau: float) -> None:
        self.names.append(name)
        self.W = np.concatenate([self.W, v[None, :].astype(np.float32)])
        self.tau = np.concatenate([self.tau, np.array([tau], dtype=np.float32)])

    def set_threshold_at_fpr(self, name: str, neg_scores: np.ndarray, target_fpr: float) -> float:
        """Threshold = (1 - target_fpr) quantile of negative scores."""
        i = self.names.index(name)
        t = float(np.quantile(neg_scores, 1.0 - target_fpr))
        self.tau[i] = t
        return t

    # ---- scoring --------------------------------------------------------
    def score(self, acts: np.ndarray) -> np.ndarray:
        """acts: [n, d] -> cosine scores [n, P] in [-1, 1]."""
        return self.features(acts) @ self.W.T

    def violations(self, acts: np.ndarray) -> np.ndarray:
        return self.score(acts) >= self.tau[None, :]

    # ---- io -------------------------------------------------------------
    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu, sigma=self.sigma, W=self.W, tau=self.tau, names=np.array(self.names))

    @classmethod
    def load(cls, path: str) -> "ProbeBank":
        z = np.load(path, allow_pickle=False)
        b = cls(hidden_dim=int(z["mu"].shape[0]), mu=z["mu"], sigma=z["sigma"], W=z["W"], tau=z["tau"])
        b.names = [str(n) for n in z["names"]]
        return b
