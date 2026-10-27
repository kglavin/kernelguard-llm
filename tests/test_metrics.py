import numpy as np

from kernelguard.eval.metrics import auroc, tpr_at_fpr, summarize


def test_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1]); s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert auroc(y, s) == 1.0
    assert tpr_at_fpr(y, s, 0.0) == 1.0


def test_overlap():
    y = np.array([0, 1, 0, 1]); s = np.array([0.1, 0.2, 0.3, 0.4])
    assert 0.5 < auroc(y, s) < 1.0
    r = summarize(y, s)
    assert r["n"] == 4 and r["n_pos"] == 2
