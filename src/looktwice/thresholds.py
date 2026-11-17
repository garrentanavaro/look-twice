from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

BADGES = ("authentic", "uncertain", "manipulated")


def best_f1_threshold(y_true, prob) -> float:
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thr = precision_recall_curve(y_true, prob)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float((thr[i - 1] + thr[i]) / 2) if i > 0 else float(thr[i])


def select_band(y_true, prob, t: float, target_acc: float = 0.95, max_width: float = 0.4,
                step: float = 0.01) -> dict:
    y_true, prob = np.asarray(y_true).astype(int), np.asarray(prob, dtype=float)
    result = None
    for w in np.arange(0.0, max_width + 1e-9, step):
        lo, hi = max(0.0, t - w), min(1.0, t + w)
        confident = (prob < lo) | (prob >= hi)
        coverage = float(confident.mean())
        acc = float(((prob[confident] >= hi).astype(int) == y_true[confident]).mean()) if confident.any() else float("nan")
        result = {"t": t, "lo": lo, "hi": hi, "width": round(float(w), 4), "coverage": coverage,
                  "confident_acc": acc, "target_acc": target_acc, "reached": bool(acc >= target_acc)}
        if confident.any() and acc >= target_acc:
            break
    return result


def choose_thresholds(y_true, prob, target_acc: float = 0.95, max_width: float = 0.4) -> dict:
    return select_band(y_true, prob, best_f1_threshold(y_true, prob), target_acc, max_width)


def badge(prob, lo: float, hi: float) -> np.ndarray:
    prob = np.asarray(prob, dtype=float)
    return np.where(prob >= hi, "manipulated", np.where(prob < lo, "authentic", "uncertain"))
