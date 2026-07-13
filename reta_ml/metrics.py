"""
Evaluation utilities for Sprint 5.

Computes per-class precision/recall/F1, macro F1, and confusion matrices using
the project's canonical 4-class ordering:
  0=Clean, 1=Operational Error, 2=Global Outlier, 3=Local Outlier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from reta_ml.graph import LABEL_MAP, label_to_index


CLASS_NAMES: Tuple[str, ...] = (
    "Clean",
    "Operational Error",
    "Global Outlier",
    "Local Outlier",
)


def labels_to_indices(labels: Sequence[object]) -> np.ndarray:
    """Map label strings to indices; unknown -> -1.

    Uses the shared, alias-aware :func:`reta_ml.graph.label_to_index` so that
    short forms like "Local"/"Global" are mapped correctly (rather than being
    dropped to -1), consistently with the GNN's label construction.
    """
    return np.asarray([label_to_index(v) for v in labels], dtype=np.int64)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    num_classes: int = 4,
    ignore_index: int = -1,
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if t == ignore_index:
            continue
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def prf_from_cm(cm: np.ndarray, *, eps: float = 1e-12) -> Dict[str, float]:
    cm = np.asarray(cm, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be square")
    k = int(cm.shape[0])

    out: Dict[str, float] = {}
    f1s: List[float] = []
    for c in range(k):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - cm[c, c])
        fn = float(cm[c, :].sum() - cm[c, c])
        support = float(cm[c, :].sum())

        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall + eps)
            if (precision + recall) > 0
            else 0.0
        )
        out[f"class_{c}_precision"] = float(precision)
        out[f"class_{c}_recall"] = float(recall)
        out[f"class_{c}_f1"] = float(f1)
        out[f"class_{c}_support"] = float(support)
        f1s.append(float(f1))

    out["macro_f1"] = float(sum(f1s) / max(len(f1s), 1))

    # Macro F1 over only the classes that actually have support in this set.
    # Averaging over all 4 classes when 1–2 of them are absent caps the score
    # far below 1.0 and makes models look worse than they are; this present-
    # class macro is the fairer headline metric and is used for model
    # selection.
    present = [c for c in range(k) if float(cm[c, :].sum()) > 0]
    out["macro_f1_present"] = (
        float(sum(out[f"class_{c}_f1"] for c in present) / len(present))
        if present
        else 0.0
    )
    out["n_present_classes"] = float(len(present))
    return out


def metrics_table_from_cm(cm: np.ndarray) -> List[Dict[str, object]]:
    """Return a list of dict rows suitable for CSV/markdown."""
    m = prf_from_cm(cm)
    rows: List[Dict[str, object]] = []
    for c, name in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class_index": c,
                "class_name": name,
                "precision": m[f"class_{c}_precision"],
                "recall": m[f"class_{c}_recall"],
                "f1": m[f"class_{c}_f1"],
                "support": int(m[f"class_{c}_support"]),
            }
        )
    rows.append(
        {
            "class_index": "",
            "class_name": "macro",
            "precision": "",
            "recall": "",
            "f1": m["macro_f1"],
            "support": int(cm.sum()),
        }
    )
    return rows


def entropy_from_probs(probs: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)

