"""
Sprint 6: Optuna hyperparameter sweep for HeteroGAT.

This module defines an Optuna study that tunes **empirical** hyper-parameters
while keeping **data-driven** parameters (e.g. spatial radius R, temporal K,
time-gap thresholds) fixed from :mod:`reta_ml.config` and the variogram.

Search space
------------
- Focal loss:
  - gamma (focusing parameter)
  - 4-class alpha weights (Clean, Operational Error, Global Outlier, Local Outlier)
- HeteroGAT architecture:
  - hidden_dim
  - num_heads
- Optimiser:
  - learning rate
  - dropout

The objective maximises **macro F1** on a held-out spatial split of the
training field (K-means on coordinates).  Each trial trains a single HeteroGAT
run and saves:

- ``runs/<timestamp>_heterogat/`` with:
  - ``config.json`` (training config + model hyperparams)
  - ``metrics.csv`` (per-epoch metrics)
  - ``best_model.pt`` (checkpoint used for the objective)
  - ``optuna_trial.json`` (trial parameters and score)

Usage
-----
From the repo root:

  python -m reta_ml.optuna_sweep --train-path data/Manual_Veris_V2.1_corrected_annotation.gpkg \\
      --main-variable yield --n-trials 50

Approximate runtime: depends on dataset size and epochs. As a rough guide,
on a single CPU with ~20 training epochs, 10–20 trials may take tens of
minutes; 50–100 trials can take **hours**. Start with 5–10 trials to sanity
check the pipeline before launching a longer sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import optuna
import torch
from torch_geometric.loader import DataLoader

from reta_ml import config
from reta_ml.augment import augment_train
from reta_ml.dataset import RETAHeteroFieldDataset
from reta_ml.losses import FocalConfig
from reta_ml.model import ModelConfig, model_from_name
from reta_ml.normalize import fit_scaler, transform_df
from reta_ml.preprocess import preprocess_pipeline
from reta_ml.split import kmeans_spatial_split
from reta_ml.train import TrainConfig, derive_task_alphas_from_loader, evaluate, train_and_validate
from reta_ml.load import load_table
from reta_ml.validate import ValidationMode, validate_dataframe


def _prepare_loaders(
    train_path: Path,
    *,
    main_variable: str,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/validation ``DataLoader`` pairs from a single training field.

    Pipeline:
    - Load and validate the training table.
    - Preprocess (Path B reimplementation → ML feature table).
    - K-means spatial split → train vs test regions.
    - Augment **train** only (if ``config.AUGMENT_TRAIN`` is True).
    - Per-field normalization (fit on augmented train, transform both splits).
    - Construct one heterograph per split and wrap in PyG ``DataLoader``.
    """
    train_path = Path(train_path)
    df_raw = load_table(
        train_path,
        validate=False,
        mode=ValidationMode.TRAINING,
        main_variable=main_variable,
    )
    validate_dataframe(df_raw, mode=ValidationMode.TRAINING, main_variable=main_variable)

    proc = preprocess_pipeline(df_raw, main_variable=main_variable)

    train_df, test_df = kmeans_spatial_split(
        proc,
        n_clusters=config.SPLIT_N_CLUSTERS,
        test_clusters=config.SPLIT_TEST_CLUSTERS,
        random_state=config.SPLIT_RANDOM_STATE,
    )

    if config.AUGMENT_TRAIN:
        aug_train = augment_train(train_df, transforms=config.DEFAULT_AUGMENTATIONS)
    else:
        aug_train = train_df

    # Fit the scaler on the un-augmented train split (augmentation changes the
    # bearing distribution), then transform the augmented train and the test
    # split with it — matching the evaluation pipeline.
    scaler, norm_cols = fit_scaler(train_df)
    train_n = transform_df(aug_train, scaler, norm_cols)
    test_n = transform_df(test_df, scaler, norm_cols)

    ds_train = RETAHeteroFieldDataset(
        [train_n],
        temporal_K=config.TEMPORAL_K,
        spatial_R=config.SPATIAL_R,
        spatial_max_neighbors=config.SPATIAL_MAX_NEIGHBORS,
        transect_K=config.TRANSECT_K,
        time_gap_threshold=config.TIME_GAP_THRESHOLD,
        time_gap_n_sigma=config.TIME_GAP_N_SIGMA,
        main_variable=main_variable,
    )
    ds_val = RETAHeteroFieldDataset(
        [test_n],
        temporal_K=config.TEMPORAL_K,
        spatial_R=config.SPATIAL_R,
        spatial_max_neighbors=config.SPATIAL_MAX_NEIGHBORS,
        transect_K=config.TRANSECT_K,
        time_gap_threshold=config.TIME_GAP_THRESHOLD,
        time_gap_n_sigma=config.TIME_GAP_N_SIGMA,
        main_variable=main_variable,
    )

    train_loader = DataLoader(ds_train, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(ds_val, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def _trial_objective_factory(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    runs_dir: Path,
) -> callable:
    """
    Create an Optuna objective function that closes over the prepared loaders.
    """

    # Infer input feature dimension once from a sample batch.
    sample = next(iter(train_loader))
    in_dim = int(sample["point"].x.shape[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial: optuna.Trial) -> float:
        # ----- Hyperparameter search space ---------------------------------
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 96, 128])
        heads = trial.suggest_categorical("heads", [2, 4])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)

        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)

        # 4-class alpha weights (Clean, Op, Global, Local) – positive and
        # rescaled inside the training loop.
        alpha_clean = trial.suggest_float("alpha_clean", 0.25, 4.0, log=True)
        alpha_op = trial.suggest_float("alpha_op", 0.25, 4.0, log=True)
        alpha_global = trial.suggest_float("alpha_global", 0.25, 4.0, log=True)
        alpha_local = trial.suggest_float("alpha_local", 0.25, 4.0, log=True)
        alpha_4class = (alpha_clean, alpha_op, alpha_global, alpha_local)

        # ----- Model + training config ------------------------------------
        mcfg = ModelConfig(
            hidden_dim=hidden_dim,
            num_layers=config.GNN_NUM_LAYERS,
            heads=heads,
            dropout=dropout,
            conv_aggr="sum",
            residual=True,
        )
        model = model_from_name("heterogat", in_dim=in_dim, cfg=mcfg)

        tcfg = TrainConfig(
            model_name="heterogat",
            epochs=config.TRAIN_EPOCHS,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=config.TRAIN_BATCH_SIZE,
            device=device,
            focal_gamma=gamma,
            auto_alpha=config.FOCAL_AUTO_ALPHA,
            alpha_4class=alpha_4class,
        )

        # Each trial gets its own run directory; tag it with the trial number.
        tagged_runs_dir = runs_dir / "optuna"
        tagged_runs_dir.mkdir(parents=True, exist_ok=True)

        run_dir = train_and_validate(
            model,
            train_loader,
            val_loader,
            cfg=tcfg,
            run_dir=None,
            extra_config={"optuna_trial": int(trial.number), "optuna_params": dict(trial.params)},
        )

        # Reload best checkpoint and evaluate macro F1 on the held-out split.
        best_path = run_dir / "best_model.pt"
        model.load_state_dict(torch.load(best_path, map_location=device))

        alpha_op, alpha_global, alpha_local = derive_task_alphas_from_loader(
            train_loader, ignore_index=tcfg.ignore_index
        )
        focal_cfg = FocalConfig(
            gamma=tcfg.focal_gamma,
            alpha_op=alpha_op,
            alpha_global=alpha_global,
            alpha_local=alpha_local,
            ignore_index=tcfg.ignore_index,
        )
        metrics = evaluate(model.to(device), val_loader, device=device, focal_cfg=focal_cfg)
        macro_f1 = float(metrics.get("macro_f1", 0.0))

        # Persist trial info alongside the run for later inspection.
        trial_info: Dict[str, object] = {
            "trial_number": int(trial.number),
            "value_macro_f1": macro_f1,
            "params": dict(trial.params),
            "run_dir": str(run_dir),
        }
        (run_dir / "optuna_trial.json").write_text(json.dumps(trial_info, indent=2, sort_keys=True))

        # Attach metadata to the trial object so the study can recover the
        # best run_dir afterwards.
        trial.set_user_attr("run_dir", str(run_dir))
        trial.set_user_attr("macro_f1", macro_f1)
        return macro_f1

    return objective


def run_sweep(
    *,
    train_path: Path,
    main_variable: str,
    n_trials: int,
    runs_dir: Path,
    study_name: str | None = None,
) -> optuna.Study:
    """
    Run an Optuna study that maximises macro F1 on a held-out spatial split.
    """
    train_loader, val_loader = _prepare_loaders(train_path, main_variable=main_variable)
    objective = _trial_objective_factory(train_loader, val_loader, runs_dir=runs_dir)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)

    # Save a summary of the best trial.
    best = study.best_trial
    best_run_dir = Path(best.user_attrs.get("run_dir", ""))
    summary = {
        "study_name": study.study_name,
        "best_value_macro_f1": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": dict(best.params),
        "best_run_dir": str(best_run_dir),
    }
    best_path = runs_dir / "optuna_best_trial.json"
    best_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"[Optuna] Best macro F1={best.value:.4f} (trial {best.number})")
    if best_run_dir.exists():
        print(f"[Optuna] Best checkpoint in: {best_run_dir}")
    else:
        print("[Optuna] WARNING: best run_dir not found on disk.")
    print(f"[Optuna] Summary written to: {best_path}")
    return study


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sprint 6 Optuna sweep for HeteroGAT.")
    ap.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training GeoPackage/CSV/Parquet (defaults to config.DATA_DIR / TRAIN_TEST_FILE).",
    )
    ap.add_argument(
        "--main-variable",
        type=str,
        default=None,
        help=f"Name of the main variable column (default: config.MAIN_VARIABLE='{config.MAIN_VARIABLE}').",
    )
    ap.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run (e.g. 50–100 for a full sweep).",
    )
    ap.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Root directory for training runs (default: 'runs').",
    )
    ap.add_argument(
        "--study-name",
        type=str,
        default="reta_heterogat_optuna",
        help="Name for the Optuna study (for logging / resumption).",
    )
    return ap.parse_args()


def main() -> None:
    """CLI entry point: run an Optuna sweep with the consolidated config."""
    args = _parse_args()
    train_path = Path(args.train_path) if args.train_path else (config.DATA_DIR / config.TRAIN_TEST_FILE)
    main_variable = args.main_variable or config.MAIN_VARIABLE
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_sweep(
        train_path=train_path,
        main_variable=main_variable,
        n_trials=int(args.n_trials),
        runs_dir=runs_dir,
        study_name=args.study_name,
    )


if __name__ == "__main__":  # pragma: no cover - CLI helper
    main()

