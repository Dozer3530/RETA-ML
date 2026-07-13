"""
Temperature scaling (single scalar T) for calibration.

Fits T on a calibration set by minimizing NLL:
  p = softmax(logits / T)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import torch
    from torch import Tensor, nn
except ImportError as e:  # pragma: no cover
    raise ImportError("temperature_scaling requires torch") from e


@dataclass
class TemperatureScalingResult:
    temperature: float
    nll_before: float
    nll_after: float


def _nll_from_logits(logits: Tensor, y: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(logits, y, reduction="mean")


def fit_temperature(
    logits: Tensor,
    y: Tensor,
    *,
    ignore_index: int = -1,
    max_iter: int = 200,
    init_temperature: float = 1.0,
) -> TemperatureScalingResult:
    """
    Fit a single temperature using LBFGS on valid labels (y != ignore_index).
    Returns the fitted temperature and NLL before/after.
    """
    logits = logits.detach()
    y = y.detach().to(torch.long)

    valid = y != int(ignore_index)
    if not torch.any(valid):
        return TemperatureScalingResult(
            temperature=1.0, nll_before=float("nan"), nll_after=float("nan")
        )

    l = logits[valid]
    yt = y[valid]

    nll_before = float(_nll_from_logits(l, yt).item())

    log_T = torch.nn.Parameter(
        torch.tensor(float(np.log(max(init_temperature, 1e-6))), dtype=torch.float32)
    )

    opt = torch.optim.LBFGS([log_T], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        T = torch.exp(log_T).clamp(1e-3, 1e3)
        loss = _nll_from_logits(l / T, yt)
        loss.backward()
        return loss

    opt.step(closure)

    T = float(torch.exp(log_T).clamp(1e-3, 1e3).item())
    nll_after = float(_nll_from_logits(l / T, yt).item())
    return TemperatureScalingResult(temperature=T, nll_before=nll_before, nll_after=nll_after)


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        t = 1.0
    return logits / t


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    *,
    n_bins: int = 15,
    ignore_index: int = -1,
) -> float:
    """
    Simple ECE using max confidence and correctness.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if p.ndim != 2:
        raise ValueError("probs must be 2D [N, C]")
    if p.shape[0] != y.shape[0]:
        raise ValueError("probs and y_true must have same N")

    valid = y != int(ignore_index)
    if not np.any(valid):
        return float("nan")

    p = p[valid]
    y = y[valid]

    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    acc = (pred == y).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        w = float(mask.mean())
        bin_acc = float(acc[mask].mean())
        bin_conf = float(conf[mask].mean())
        ece += w * abs(bin_acc - bin_conf)
    return float(ece)

