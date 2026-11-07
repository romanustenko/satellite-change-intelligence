"""SCD metrics: exact agreement with a transcription of the published reference code."""

import math

import numpy as np
import pytest
import torch
from scipy import stats

from satchange.semantic.scd_metrics import SemanticChangeMetrics, cohen_kappa


def reference(preds, labels, num_class):
    """Verbatim logic of ChangeMamba changedetection/evaluation/scd.py (SCDD_eval_all)."""
    def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels, strict=True):
        hist += fast_hist(pred.flatten(), label.flatten(), num_class)
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0

    def cal_kappa(h):
        if h.sum() == 0:
            return 0.0
        po = np.diag(h).sum() / h.sum()
        pe = np.matmul(h.sum(1), h.sum(0).T) / h.sum() ** 2
        return 0.0 if pe == 1 else float((po - pe) / (1 - pe))

    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist) + 1e-10)
    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    sc_tp = np.diag(hist[1:, 1:]).sum()
    sc_precision = sc_tp / (change_pred_sum + 1e-10)
    sc_recall = sc_tp / (change_label_sum + 1e-10)
    fscd = stats.hmean([sc_precision, sc_recall]) if sc_precision > 0 and sc_recall > 0 else 0.0
    miou = float((iu[0] + iu[1]) / 2)
    return kappa_n0, float(fscd), miou, float(kappa_n0 * math.exp(iu[1]) / math.e)


def random_maps(rng, n, size=16, classes=7):
    base = rng.integers(0, classes, (n, size, size))
    other = rng.integers(0, classes, (n, size, size))
    noisy = np.where(rng.random((n, size, size)) < 0.7, base, other)
    return noisy, base


def test_matches_reference_on_random_maps_and_batch_split():
    rng = np.random.default_rng(0)
    pred, label = random_maps(rng, 6)
    expected = reference(list(pred), list(label), 7)
    metrics = SemanticChangeMetrics(7)
    for start in (0, 2, 5):  # arbitrary batching must not change pooled counts
        stop = {0: 2, 2: 5, 5: 6}[start]
        metrics.update(torch.from_numpy(pred[start:stop]), torch.from_numpy(label[start:stop]))
    got = metrics.compute()
    for name, value in zip(("kappa_n0", "fscd", "miou", "sek"), expected, strict=True):
        assert got[name] == pytest.approx(value, rel=1e-12, abs=1e-12)


def test_perfect_prediction_and_known_values():
    label = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    perfect = SemanticChangeMetrics(7)
    perfect.update(label, label)
    out = perfect.compute()
    assert out["oa"] == 1.0 and out["miou"] == pytest.approx(1.0, abs=1e-9)
    assert out["fscd"] == pytest.approx(1.0) and out["kappa_n0"] == pytest.approx(1.0)
    assert out["sek"] == pytest.approx(math.exp(1.0) / math.e)  # kappa 1, change IoU 1 -> 1.0
    all_wrong = SemanticChangeMetrics(7)
    all_wrong.update(torch.zeros_like(label), label)
    wrong = all_wrong.compute()
    assert wrong["fscd"] == 0.0 and wrong["iou_change"] == pytest.approx(0.0, abs=1e-9)
    assert cohen_kappa(np.zeros((3, 3))) == 0.0


def test_rejects_invalid_input():
    metrics = SemanticChangeMetrics(7)
    with pytest.raises(ValueError, match="Shape mismatch"):
        metrics.update(torch.zeros(2, 2, dtype=torch.long), torch.zeros(3, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="integer"):
        metrics.update(torch.zeros(2, 2), torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="outside"):
        metrics.update(torch.full((2, 2), 7, dtype=torch.long), torch.zeros(2, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="No pixels"):
        metrics.compute()
