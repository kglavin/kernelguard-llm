import numpy as np

from kernelguard.probes.bank import ProbeBank


def _synthetic(d=64, n=200, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=d)
    v /= np.linalg.norm(v)
    base = rng.normal(size=(n, d)) * 3.0 + rng.normal(size=d) * 10  # offset mean, large norms
    pos = base[: n // 2] + 8.0 * v
    neg = base[n // 2:]
    return pos.astype(np.float32), neg.astype(np.float32), v


def test_dom_recovers_direction():
    pos, neg, v = _synthetic()
    b = ProbeBank(64)
    b.set_background(np.concatenate([pos, neg]))
    d = b.direction_dom(pos, neg)
    assert abs(np.dot(d, v)) > 0.8


def test_logistic_recovers_direction():
    pos, neg, v = _synthetic()
    b = ProbeBank(64)
    b.set_background(np.concatenate([pos, neg]))
    d = b.direction_logistic(pos, neg)
    assert abs(np.dot(d, v)) > 0.8


def test_scores_bounded_and_threshold_at_fpr():
    pos, neg, _ = _synthetic()
    b = ProbeBank(64)
    b.set_background(np.concatenate([pos, neg]))
    b.add("p", b.direction_dom(pos, neg), 0.0)
    s = b.score(np.concatenate([pos, neg]))
    assert s.shape == (200, 1) and np.all(np.abs(s) <= 1.0 + 1e-5)
    t = b.set_threshold_at_fpr("p", b.score(neg)[:, 0], 0.05)
    fpr = (b.score(neg)[:, 0] >= t).mean()
    assert fpr <= 0.06


def test_orthogonalize():
    a = np.array([1.0, 1.0, 0.0]); t = np.array([0.0, 1.0, 0.0])
    o = ProbeBank.orthogonalize(a, t)
    assert abs(np.dot(o, t)) < 1e-6 and abs(np.linalg.norm(o) - 1) < 1e-6


def test_save_load(tmp_path):
    pos, neg, _ = _synthetic()
    b = ProbeBank(64)
    b.set_background(np.concatenate([pos, neg]), zscore=True)
    b.add("p", b.direction_dom(pos, neg), 0.3)
    p = tmp_path / "bank.npz"
    b.save(p)
    c = ProbeBank.load(p)
    assert c.names == ["p"] and np.allclose(c.W, b.W) and np.allclose(c.sigma, b.sigma)
