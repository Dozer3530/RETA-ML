"""
Research-grade evaluation metrics for the RETA outlier filter.

Computes a comprehensive set of indices suitable for a paper, appropriate for an
imbalanced multiclass spatial outlier-detection problem:

- Overall: accuracy (micro-F1), balanced accuracy, macro-F1 (all & present
  classes), weighted-F1, Cohen's kappa, Matthews correlation (MCC).
- Per-class: precision / recall / F1 / support, plus one-vs-rest ROC-AUC and
  PR-AUC (average precision) when probabilities are available.
- Outlier-detection framing (Clean vs not-Clean): precision, recall, F1,
  specificity, ROC-AUC, PR-AUC.
- Probabilistic quality: multiclass Brier score, NLL, expected calibration
  error (ECE).
- Operational: data retention (fraction kept as Clean), review rate, and — when
  review flags are supplied — review precision/recall (do flagged points
  actually correspond to errors?).

All sklearn calls are guarded so degenerate inputs (e.g. a class absent from the
test field) never crash the report; they're reported as NaN instead.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

CLASS_NAMES = ("Clean", "Operational Error", "Global Outlier", "Local Outlier")


def _safe(fn, *a, **k):
    try:
        return float(fn(*a, **k))
    except Exception:
        return float("nan")


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 4) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < k and 0 <= p < k:
            cm[t, p] += 1
    return cm


def full_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    probs: Optional[np.ndarray] = None,
    review_flag: Optional[np.ndarray] = None,
    num_classes: int = 4,
    class_names: Sequence[str] = CLASS_NAMES,
) -> Dict:
    """Return a nested dict of metrics. Inputs are already restricted to rows
    with a known true label (y_true in [0, num_classes-1])."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, cohen_kappa_score,
        matthews_corrcoef, f1_score, precision_recall_fscore_support,
        roc_auc_score, average_precision_score, log_loss,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n = int(len(y_true))
    present = sorted(set(y_true.tolist()))
    cm = confusion(y_true, y_pred, num_classes)

    out: Dict = {"n": n, "classes_present": present, "confusion_matrix": cm.tolist()}

    # ---- overall ----
    out["overall"] = {
        "accuracy": _safe(accuracy_score, y_true, y_pred),
        "balanced_accuracy": _safe(balanced_accuracy_score, y_true, y_pred),
        "macro_f1_all": _safe(f1_score, y_true, y_pred, average="macro",
                              labels=list(range(num_classes)), zero_division=0),
        "macro_f1_present": _safe(f1_score, y_true, y_pred, average="macro",
                                  labels=present, zero_division=0),
        "weighted_f1": _safe(f1_score, y_true, y_pred, average="weighted", zero_division=0),
        "cohen_kappa": _safe(cohen_kappa_score, y_true, y_pred),
        "mcc": _safe(matthews_corrcoef, y_true, y_pred),
    }

    # ---- per class ----
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    per = {}
    for c in range(num_classes):
        d = {"precision": float(p[c]), "recall": float(r[c]), "f1": float(f[c]),
             "support": int(s[c])}
        if probs is not None and s[c] > 0 and s[c] < n:
            y_bin = (y_true == c).astype(int)
            d["roc_auc_ovr"] = _safe(roc_auc_score, y_bin, probs[:, c])
            d["pr_auc_ovr"] = _safe(average_precision_score, y_bin, probs[:, c])
        per[class_names[c]] = d
    out["per_class"] = per

    # ---- outlier detection (Clean vs not-Clean) ----
    y_out = (y_true != 0).astype(int)
    p_out = (y_pred != 0).astype(int)
    tp = int(((y_out == 1) & (p_out == 1)).sum()); fp = int(((y_out == 0) & (p_out == 1)).sum())
    fn = int(((y_out == 1) & (p_out == 0)).sum()); tn = int(((y_out == 0) & (p_out == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    binm = {
        "precision": prec, "recall_sensitivity": rec, "specificity": spec,
        "f1": (2 * prec * rec / (prec + rec)) if (prec and rec and not np.isnan(prec) and not np.isnan(rec)) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    if probs is not None:
        score_out = 1.0 - probs[:, 0]  # P(not Clean)
        binm["roc_auc"] = _safe(roc_auc_score, y_out, score_out)
        binm["pr_auc"] = _safe(average_precision_score, y_out, score_out)
    out["outlier_detection"] = binm

    # ---- probabilistic quality ----
    if probs is not None:
        oneh = np.zeros_like(probs); oneh[np.arange(n), y_true] = 1.0
        out["probabilistic"] = {
            "brier": float(np.mean(np.sum((probs - oneh) ** 2, axis=1))),
            "nll": _safe(log_loss, y_true, probs, labels=list(range(num_classes))),
            "ece": _ece(probs, y_true),
        }

    # ---- operational ----
    op = {
        "data_retention_pred_clean": float(np.mean(y_pred == 0)),
        "predicted_outlier_rate": float(np.mean(y_pred != 0)),
    }
    if review_flag is not None:
        review_flag = np.asarray(review_flag, dtype=bool)
        err = y_true != y_pred
        op["review_rate"] = float(np.mean(review_flag))
        op["review_precision"] = float(err[review_flag].mean()) if review_flag.any() else float("nan")
        op["review_recall_of_errors"] = float(review_flag[err].mean()) if err.any() else float("nan")
    out["operational"] = op
    return out


def _ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if m.any():
            ece += float(m.mean()) * abs(float(acc[m].mean()) - float(conf[m].mean()))
    return float(ece)
