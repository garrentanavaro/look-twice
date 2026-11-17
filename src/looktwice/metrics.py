from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y_true)) == 2
    return {
        "acc": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob) if both else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)), "threshold": threshold,
    }


def video_level(preds: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    return (preds.groupby("video_id")
                 .agg(label=("label", "first"), prob=(prob_col, "mean"), n_frames=(prob_col, "size"))
                 .reset_index())


def prefixed(d: dict, prefix: str, keys=("acc", "precision", "recall", "f1", "auc")) -> dict:
    return {f"{prefix}{k}": d[k] for k in keys}
