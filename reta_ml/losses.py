"""
Loss functions for Sprint 4 training.

Masked hierarchical loss (single 4-class head):

  L_total = L_op + I(op_clean) * L_global + I(global_clean) * L_local

Where:
- op_clean: ground truth is NOT Operational Error
- global_clean: ground truth is NOT Operational Error and NOT Global Outlier

Classes (canonical):
  0 = Clean
  1 = Operational Error
  2 = Global Outlier
  3 = Local Outlier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FocalConfig:
    gamma: float = 2.0
    alpha_op: Optional[Tuple[float, float]] = None  # (not-op, op)
    alpha_global: Optional[Tuple[float, float]] = None  # (not-global, global)
    alpha_local: Optional[Tuple[float, float]] = None  # (clean, local)
    ignore_index: int = -1
    eps: float = 1e-8


def focal_loss(
    logits: Tensor,
    target: Tensor,
    *,
    gamma: float = 2.0,
    alpha: Optional[Tensor] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Tensor:
    """Multi-class focal loss on logits.

    Parameters
    ----------
    logits : Tensor
        Shape [N, C]
    target : Tensor
        Shape [N], integer in [0, C-1]
    gamma : float
        Focusing parameter.
    alpha : Tensor, optional
        Per-class weighting, shape [C].
    reduction : str
        "none" | "mean" | "sum"
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if target.ndim != 1:
        raise ValueError("target must have shape [N]")
    if logits.shape[0] != target.shape[0]:
        raise ValueError("logits and target must agree on N")

    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()

    target = target.to(dtype=torch.long)
    idx = target.view(-1, 1)
    log_pt = logp.gather(1, idx).squeeze(1)
    pt = p.gather(1, idx).squeeze(1)

    if alpha is not None:
        if alpha.ndim != 1 or alpha.shape[0] != logits.shape[1]:
            raise ValueError("alpha must have shape [C]")
        alpha_t = alpha.to(device=logits.device, dtype=logits.dtype).gather(
            0, target
        )
    else:
        alpha_t = torch.ones_like(pt)

    loss = -alpha_t * torch.pow((1.0 - pt).clamp_min(0.0), gamma) * log_pt

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean() if loss.numel() else loss.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


def _logsumexp_cols(logits: Tensor, cols: Tuple[int, ...]) -> Tensor:
    return torch.logsumexp(logits[:, list(cols)], dim=1)


def masked_hierarchical_focal_loss(
    logits_4: Tensor,
    y: Tensor,
    *,
    cfg: Optional[FocalConfig] = None,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute masked hierarchical focal loss from a single 4-class head.

    Returns (loss_scalar, metrics_dict).
    """
    cfg = cfg or FocalConfig()
    ignore_index = int(cfg.ignore_index)

    if logits_4.ndim != 2 or logits_4.shape[1] != 4:
        raise ValueError("logits_4 must have shape [N, 4]")
    if y.ndim != 1 or y.shape[0] != logits_4.shape[0]:
        raise ValueError("y must have shape [N]")

    y = y.to(dtype=torch.long, device=logits_4.device)
    valid = y != ignore_index
    if not torch.any(valid):
        z = logits_4.sum() * 0.0
        return z, {
            "loss_total": 0.0,
            "loss_op": 0.0,
            "loss_global": 0.0,
            "loss_local": 0.0,
            "n_valid": 0.0,
        }

    lv = logits_4[valid]
    yv = y[valid]

    # --- L_op: Operational Error vs Not Operational ---
    # not-op = logsumexp(clean, global, local), op = logit(op)
    op_logits = torch.stack(
        [_logsumexp_cols(lv, (0, 2, 3)), lv[:, 1]], dim=1
    )  # [N,2]
    op_target = (yv == 1).to(torch.long)  # 1=op
    alpha_op = (
        torch.tensor(cfg.alpha_op, dtype=lv.dtype, device=lv.device)
        if cfg.alpha_op is not None
        else None
    )
    l_op_vec = focal_loss(
        op_logits,
        op_target,
        gamma=float(cfg.gamma),
        alpha=alpha_op,
        reduction="none",
        eps=cfg.eps,
    )

    # --- L_global: Global Outlier vs (Clean or Local), masked to non-op nodes ---
    op_clean = yv != 1
    global_logits = torch.stack(
        [_logsumexp_cols(lv, (0, 3)), lv[:, 2]], dim=1
    )  # [N,2]
    global_target = (yv == 2).to(torch.long)  # 1=global
    alpha_global = (
        torch.tensor(cfg.alpha_global, dtype=lv.dtype, device=lv.device)
        if cfg.alpha_global is not None
        else None
    )
    l_global_vec = focal_loss(
        global_logits,
        global_target,
        gamma=float(cfg.gamma),
        alpha=alpha_global,
        reduction="none",
        eps=cfg.eps,
    )
    l_global_vec = l_global_vec * op_clean.to(l_global_vec.dtype)

    # --- L_local: Local Outlier vs Clean, masked to (Clean or Local) nodes only ---
    global_clean = (yv != 1) & (yv != 2)
    local_logits = lv[:, [0, 3]]  # clean vs local, [N,2]
    local_target = (yv == 3).to(torch.long)  # 1=local
    alpha_local = (
        torch.tensor(cfg.alpha_local, dtype=lv.dtype, device=lv.device)
        if cfg.alpha_local is not None
        else None
    )
    l_local_vec = focal_loss(
        local_logits,
        local_target,
        gamma=float(cfg.gamma),
        alpha=alpha_local,
        reduction="none",
        eps=cfg.eps,
    )
    l_local_vec = l_local_vec * global_clean.to(l_local_vec.dtype)

    l_total_vec = l_op_vec + l_global_vec + l_local_vec
    loss = l_total_vec.mean()

    # Component means over *all valid nodes* (with masked nodes contributing 0)
    n_valid = float(valid.sum().item())
    metrics = {
        "loss_total": float(loss.item()),
        "loss_op": float(l_op_vec.mean().item()),
        "loss_global": float(l_global_vec.mean().item()),
        "loss_local": float(l_local_vec.mean().item()),
        "n_valid": n_valid,
    }
    return loss, metrics

