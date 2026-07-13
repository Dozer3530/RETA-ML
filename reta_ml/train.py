"""
Sprint 4 training loop: train on train split, validate on test split.

Artifacts per run:
- runs/<timestamp>_<model>/config.json
- runs/<timestamp>_<model>/metrics.csv
- runs/<timestamp>_<model>/best_model.pt   (state_dict only)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn

from reta_ml.losses import FocalConfig, masked_hierarchical_focal_loss
from reta_ml import config as cfg


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for a single training run.

    Most defaults are sourced from ``reta_ml.config`` so that experiments and
    Optuna sweeps can share a single consolidated configuration.
    """

    model_name: str = "heterogat"
    epochs: int = cfg.TRAIN_EPOCHS
    lr: float = cfg.TRAIN_LR
    weight_decay: float = cfg.TRAIN_WEIGHT_DECAY
    batch_size: int = cfg.TRAIN_BATCH_SIZE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Focal / hierarchical loss
    focal_gamma: float = cfg.FOCAL_GAMMA
    auto_alpha: bool = cfg.FOCAL_AUTO_ALPHA  # if True and alphas are None, derive from train labels
    alpha_4class: Optional[Tuple[float, float, float, float]] = None  # (Clean, Op, Global, Local)
    alpha_op: Optional[Tuple[float, float]] = None
    alpha_global: Optional[Tuple[float, float]] = None
    alpha_local: Optional[Tuple[float, float]] = None
    ignore_index: int = -1

    # Repro
    seed: int = 42

    # Runs output
    runs_dir: str = "runs"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(*, runs_dir: str, model_name: str) -> Path:
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / f"{_now_stamp()}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@torch.no_grad()
def derive_task_alphas_from_loader(
    loader,
    *,
    ignore_index: int = -1,
    num_classes: int = 4,
    eps: float = 1e-8,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Derive (alpha_op, alpha_global, alpha_local) from label frequencies.

    Produces inverse-frequency weights, normalized so each 2-class alpha pair
    has mean 1.0.
    """
    counts = torch.zeros(num_classes, dtype=torch.long)
    for batch in loader:
        y = batch["point"].y.detach().cpu().to(torch.long)
        y = y[y != ignore_index]
        if y.numel():
            counts += torch.bincount(y.clamp_min(0), minlength=num_classes)[:num_classes]

    # Inverse-frequency weights, but ONLY over classes that are actually
    # present. A class with zero support would otherwise get 1/eps (~1e8) and
    # dominate every task group that references it (e.g. an absent Global
    # Outlier class would drive the "operational error" weight to ~0, so the
    # model would stop learning operational errors). Absent classes get weight
    # 0 here and a neutral fallback of 1.0 where they appear as a task target.
    counts_f = counts.to(torch.float32)
    present = counts_f > 0
    w = torch.zeros(num_classes, dtype=torch.float32)
    if bool(present.any()):
        w[present] = 1.0 / counts_f[present]
        w[present] = w[present] / w[present].mean().clamp_min(eps)

    def _grp_mean(idxs) -> Tensor:
        sel = [i for i in idxs if bool(present[i])]
        return w[sel].mean() if sel else torch.tensor(1.0)

    def _w(i: int) -> Tensor:
        return w[i] if bool(present[i]) else torch.tensor(1.0)

    def _norm_pair(a: Tensor) -> Tuple[float, float]:
        a = a / a.mean().clamp_min(eps)
        return (float(a[0].item()), float(a[1].item()))

    alpha_op = _norm_pair(torch.stack([_grp_mean([0, 2, 3]), _w(1)]))
    alpha_global = _norm_pair(torch.stack([_grp_mean([0, 3]), _w(2)]))
    alpha_local = _norm_pair(torch.stack([_w(0), _w(3)]))
    return alpha_op, alpha_global, alpha_local


def _confusion_matrix(y_true: Tensor, y_pred: Tensor, num_classes: int) -> Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.view(-1).tolist(), y_pred.view(-1).tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def classification_metrics_from_cm(cm: Tensor, eps: float = 1e-12) -> Dict[str, float]:
    """Return per-class P/R/F1 and macro F1 for a confusion matrix."""
    num_classes = int(cm.shape[0])
    metrics: Dict[str, float] = {}

    f1s: List[float] = []
    for c in range(num_classes):
        tp = float(cm[c, c].item())
        fp = float(cm[:, c].sum().item() - cm[c, c].item())
        fn = float(cm[c, :].sum().item() - cm[c, c].item())
        support = float(cm[c, :].sum().item())

        precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall + eps)
            if (precision + recall) > 0
            else 0.0
        )

        metrics[f"class_{c}_precision"] = precision
        metrics[f"class_{c}_recall"] = recall
        metrics[f"class_{c}_f1"] = f1
        metrics[f"class_{c}_support"] = support
        f1s.append(f1)

    metrics["macro_f1"] = float(sum(f1s) / max(len(f1s), 1))

    # Present-class macro F1: average only over classes with support, so that
    # absent classes (0 support) do not artificially deflate the score. This is
    # the metric used to select the best checkpoint.
    present = [c for c in range(num_classes) if float(cm[c, :].sum()) > 0]
    metrics["macro_f1_present"] = (
        float(sum(metrics[f"class_{c}_f1"] for c in present) / len(present))
        if present
        else 0.0
    )
    metrics["n_present_classes"] = float(len(present))
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    *,
    device: str,
    focal_cfg: FocalConfig,
    num_classes: int = 4,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_nodes = 0

    ys: List[Tensor] = []
    ps: List[Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch["point"].y
        loss, _ = masked_hierarchical_focal_loss(logits, y, cfg=focal_cfg)

        valid = y != focal_cfg.ignore_index
        total_loss += float(loss.item()) * int(valid.sum().item())
        total_nodes += int(valid.sum().item())

        if torch.any(valid):
            ys.append(y[valid].detach().cpu())
            ps.append(logits.argmax(dim=-1)[valid].detach().cpu())

    out: Dict[str, float] = {}
    out["loss"] = total_loss / max(total_nodes, 1)
    out["n_valid_nodes"] = float(total_nodes)

    if ys:
        y_true = torch.cat(ys, dim=0)
        y_pred = torch.cat(ps, dim=0)
        cm = _confusion_matrix(y_true, y_pred, num_classes=num_classes)
        out.update(classification_metrics_from_cm(cm))
    else:
        # No labels available
        out.update({f"class_{c}_precision": 0.0 for c in range(num_classes)})
        out.update({f"class_{c}_recall": 0.0 for c in range(num_classes)})
        out.update({f"class_{c}_f1": 0.0 for c in range(num_classes)})
        out.update({f"class_{c}_support": 0.0 for c in range(num_classes)})
        out["macro_f1"] = 0.0

    return out


def _write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def _append_metrics_csv(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train_and_validate(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    cfg: TrainConfig,
    run_dir: Optional[Path] = None,
    extra_config: Optional[Dict[str, object]] = None,
    progress_callback=None,
) -> Path:
    set_seed(cfg.seed)
    device = cfg.device
    model = model.to(device)

    alpha_op = cfg.alpha_op
    alpha_global = cfg.alpha_global
    alpha_local = cfg.alpha_local

    if cfg.alpha_4class is not None:
        a = cfg.alpha_4class
        # Derive task-level (binary) alphas from 4-class weights.
        alpha_op = alpha_op or (float((a[0] + a[2] + a[3]) / 3.0), float(a[1]))
        alpha_global = alpha_global or (float((a[0] + a[3]) / 2.0), float(a[2]))
        alpha_local = alpha_local or (float(a[0]), float(a[3]))

    if cfg.auto_alpha and (alpha_op is None or alpha_global is None or alpha_local is None):
        auto_op, auto_global, auto_local = derive_task_alphas_from_loader(
            train_loader, ignore_index=cfg.ignore_index
        )
        alpha_op = alpha_op or auto_op
        alpha_global = alpha_global or auto_global
        alpha_local = alpha_local or auto_local

    focal_cfg = FocalConfig(
        gamma=cfg.focal_gamma,
        alpha_op=alpha_op,
        alpha_global=alpha_global,
        alpha_local=alpha_local,
        ignore_index=cfg.ignore_index,
    )

    run_dir = run_dir or make_run_dir(runs_dir=cfg.runs_dir, model_name=cfg.model_name)
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"
    best_path = run_dir / "best_model.pt"

    cfg_out = asdict(cfg)
    cfg_out["resolved_alpha_op"] = list(alpha_op) if alpha_op is not None else None
    cfg_out["resolved_alpha_global"] = (
        list(alpha_global) if alpha_global is not None else None
    )
    cfg_out["resolved_alpha_local"] = list(alpha_local) if alpha_local is not None else None
    if extra_config:
        cfg_out["extra"] = extra_config
    _write_json(config_path, cfg_out)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_macro = -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_nodes = 0

        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)

            logits = model(batch)
            y = batch["point"].y
            loss, _ = masked_hierarchical_focal_loss(logits, y, cfg=focal_cfg)
            loss.backward()
            opt.step()

            valid = y != focal_cfg.ignore_index
            n = int(valid.sum().item())
            total_loss += float(loss.item()) * n
            total_nodes += n

        train_loss = total_loss / max(total_nodes, 1)
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            focal_cfg=focal_cfg,
            num_classes=4,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            "val_minority_recall": float(val_metrics.get("class_3_recall", 0.0)),
        }
        _append_metrics_csv(metrics_path, row)

        # Optional live-progress hook (e.g. for a GUI). Never affects training.
        if progress_callback is not None:
            try:
                progress_callback(epoch, dict(row))
            except Exception:
                pass

        # Select on present-class macro F1 (falls back to 4-class macro if the
        # present-class variant is unavailable for any reason).
        macro = float(
            val_metrics.get("macro_f1_present", val_metrics.get("macro_f1", 0.0))
        )
        if macro > best_macro:
            best_macro = macro
            torch.save(model.state_dict(), best_path)

    return run_dir

