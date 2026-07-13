"""
LOFO ablation engine — single-stage and two-stage variants with optional scaler.

Supports controlled experiments isolating: multi-band vs fixed-radius local stats,
StandardScaler, and two-stage sequential (operational → value/spatial) architecture.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from reta_ml import config
from reta_ml.load import load_table
from reta_ml.preprocess import (
    LocalStatsMode,
    preprocess_pipeline,
    recompute_value_stats,
)
from reta_ml.split import kmeans_spatial_split
from reta_ml.graph import build_labels
from reta_ml.normalize import fit_scaler, transform_df
from reta_ml.lofo_benchmark import (
    _balanced_class_weights,
    _balanced_sample_weight,
    _build_models,
    _feature_matrix,
    _metrics,
)
from reta_ml.metrics import CLASS_NAMES

ProgressCb = Optional[Callable[[str], None]]

AblationRunId = Literal[
    "v2", "v4", "v4_fixedbands", "v4_fixedbands_nostd",
    "v4_fixedbands_novalue_nomean", "v4_5", "v5_control", "v5_full",
]


def ablation_spec(run_id: AblationRunId) -> Dict:
    """Return protocol parameters for a named ablation run."""
    specs = {
        "v2": dict(
            two_stage=False,
            use_scaler=False,
            local_stats_mode="fixed",
            single_stage_features=list(config.RF_XGB_FEATURES_LEGACY),
            stage1_features=None,
            stage2_features=None,
            label="Run 1 — baseline (fixed 15 m, single-stage, no scaler)",
        ),
        "v4": dict(
            two_stage=False,
            use_scaler=False,
            local_stats_mode="multiband",
            single_stage_features=list(config.RF_XGB_FEATURES),
            stage1_features=None,
            stage2_features=None,
            label="Run 2 — multi-band features only (single-stage, no scaler)",
        ),
        "v4_fixedbands": dict(
            two_stage=False,
            use_scaler=False,
            local_stats_mode="fixedbands",
            single_stage_features=list(config.RF_XGB_FEATURES_FIXED_BANDS),
            stage1_features=None,
            stage2_features=None,
            label="Run 2 fixed-band — 15/30/45 m multi-band (single-stage, no scaler)",
        ),
        "v4_fixedbands_nostd": dict(
            two_stage=False,
            use_scaler=False,
            local_stats_mode="fixedbands",
            single_stage_features=list(config.RF_XGB_FEATURES_FIXED_BANDS_NO_STD),
            stage1_features=None,
            stage2_features=None,
            label="Run 2 fixed-band no-std — 15/30/45 m, no local_std (single-stage)",
        ),
        "v4_fixedbands_novalue_nomean": dict(
            two_stage=False,
            use_scaler=False,
            local_stats_mode="fixedbands",
            single_stage_features=list(config.RF_XGB_FEATURES_FIXED_BANDS_NO_STD_NO_VALUE),
            stage1_features=None,
            stage2_features=None,
            label="Run 2 fixed-band trimmed — no value, no local_mean, no local_std",
        ),
        "v4_5": dict(
            two_stage=False,
            use_scaler=True,
            local_stats_mode="multiband",
            single_stage_features=list(config.RF_XGB_FEATURES),
            stage1_features=None,
            stage2_features=None,
            label="Run 3 — multi-band + StandardScaler (single-stage)",
        ),
        "v5_control": dict(
            two_stage=True,
            use_scaler=False,
            local_stats_mode="fixed",
            single_stage_features=None,
            stage1_features=list(config.STAGE1_FEATURES),
            stage2_features=list(config.STAGE2_FEATURES_LEGACY),
            label="Run 4 — two-stage, fixed 15 m value features (control)",
        ),
        "v5_full": dict(
            two_stage=True,
            use_scaler=True,
            local_stats_mode="multiband",
            single_stage_features=None,
            stage1_features=list(config.STAGE1_FEATURES),
            stage2_features=list(config.STAGE2_FEATURES),
            label="Run 5 — two-stage + multi-band + scaler (full proposal)",
        ),
    }
    if run_id not in specs:
        raise ValueError(f"Unknown ablation run_id: {run_id}")
    return deepcopy(specs[run_id])


def _stage1_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Operational-error detection (class 1) as a binary task."""
    yt = (np.asarray(y_true) == 1).astype(int)
    yp = (np.asarray(y_pred) == 1).astype(int)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "operational_precision": round(float(prec), 4),
        "operational_recall": round(float(rec), 4),
        "operational_f1": round(float(f1), 4),
        "operational_support": int(yt.sum()),
    }


def _prepare_splits(
    train_specs: List[Tuple[str, str]],
    test_spec: Tuple[str, str],
    *,
    n_clusters: int,
    seed: int,
    local_stats_mode: LocalStatsMode,
    progress: ProgressCb,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def _say(m: str) -> None:
        if progress:
            progress(m)

    train_regions, val_regions = [], []
    for path, col in train_specs:
        _say(f"Preprocessing + K-means split (k={n_clusters}): {Path(path).stem}")
        raw = load_table(Path(path), validate=False).copy()
        raw["value"] = pd.to_numeric(raw[col], errors="coerce")
        proc = preprocess_pipeline(
            raw, main_variable="value", local_stats_mode=local_stats_mode,
        )
        tr_reg, val_reg = kmeans_spatial_split(
            proc, n_clusters=n_clusters,
            test_clusters=config.SPLIT_TEST_CLUSTERS, random_state=seed,
        )
        train_regions.append(
            recompute_value_stats(
                tr_reg, main_variable="value", local_stats_mode=local_stats_mode,
            )
        )
        val_regions.append(
            recompute_value_stats(
                val_reg, main_variable="value", local_stats_mode=local_stats_mode,
            )
        )

    train_all = pd.concat(train_regions, ignore_index=True)
    val_all = pd.concat(val_regions, ignore_index=True)

    test_path, test_col = test_spec
    _say(f"Preprocessing held-out TEST field: {Path(test_path).stem}")
    raw_t = load_table(Path(test_path), validate=False).copy()
    raw_t["value"] = pd.to_numeric(raw_t[test_col], errors="coerce")
    proc_t = preprocess_pipeline(
        raw_t, main_variable="value", local_stats_mode=local_stats_mode,
    )
    return train_all, val_all, proc_t


def _scale_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler, cols = fit_scaler(train_df, feature_cols=feature_cols)
    return (
        transform_df(train_df, scaler, cols),
        transform_df(val_df, scaler, cols),
        transform_df(test_df, scaler, cols),
    )


def _xy_single(
    df: pd.DataFrame, feats: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    X = _feature_matrix(df, feats)
    y = build_labels(df)
    m = y != -1
    return X[m], y[m]


def _predict_two_stage(
    clf1,
    clf2,
    le2,
    X1: np.ndarray,
    X2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (combined 4-class preds, stage1 binary op preds)."""
    op_bin = clf1.predict(X1).astype(int)
    combined = np.zeros(len(X1), dtype=int)
    not_op = op_bin == 0
    if np.any(not_op):
        s2 = clf2.predict(X2[not_op]).astype(int)
        combined[not_op] = le2.inverse_transform(s2).astype(int)
    combined[op_bin == 1] = 1
    return combined, op_bin


def run_one_fold_ablation(
    test_spec: Tuple[str, str],
    train_specs: List[Tuple[str, str]],
    *,
    run_id: AblationRunId,
    models: Tuple[str, ...] = ("rf", "xgb"),
    n_clusters: int = config.LOFO_N_CLUSTERS,
    seed: int = config.SPLIT_RANDOM_STATE,
    max_ratio: Optional[float] = None,
    progress: ProgressCb = None,
) -> Dict:
    """Run one LOFO fold under the ablation protocol for *run_id*."""
    from sklearn.preprocessing import LabelEncoder

    spec = ablation_spec(run_id)
    two_stage = spec["two_stage"]
    use_scaler = spec["use_scaler"]
    local_stats_mode: LocalStatsMode = spec["local_stats_mode"]

    def _say(m: str) -> None:
        if progress:
            progress(m)

    train_all, val_all, proc_t = _prepare_splits(
        train_specs, test_spec,
        n_clusters=n_clusters, seed=seed,
        local_stats_mode=local_stats_mode, progress=progress,
    )

    out_models: Dict[str, Dict] = {}

    if not two_stage:
        feats = spec["single_stage_features"]
        tr_df, va_df, te_df = train_all, val_all, proc_t
        if use_scaler:
            _say("Fitting StandardScaler on train regions…")
            tr_df, va_df, te_df = _scale_splits(train_all, val_all, proc_t, feats)

        X_tr, y_tr = _xy_single(tr_df, feats)
        X_val, y_val = _xy_single(va_df, feats)
        X_te, y_te = _xy_single(te_df, feats)

        le = LabelEncoder().fit(y_tr)
        y_tr_enc = le.transform(y_tr)
        sw = _balanced_sample_weight(y_tr_enc, max_ratio)
        rf_cw = ("balanced" if max_ratio is None
                 else _balanced_class_weights(y_tr_enc, max_ratio))

        for name, clf, use_sw in _build_models(models, seed, rf_class_weight=rf_cw):
            _say(f"Fitting {name} (train {len(y_tr):,} pts)…")
            if use_sw:
                clf.fit(X_tr, y_tr_enc, sample_weight=sw)
            else:
                clf.fit(X_tr, y_tr_enc)
            val_pred = le.inverse_transform(clf.predict(X_val).astype(int)).astype(int)
            te_pred = le.inverse_transform(clf.predict(X_te).astype(int)).astype(int)
            out_models[name] = {
                "validation": _metrics(y_val, val_pred),
                "test": _metrics(y_te, te_pred),
            }
    else:
        s1_feats = spec["stage1_features"]
        s2_feats = spec["stage2_features"]

        tr1, va1, te1 = train_all, val_all, proc_t
        tr2, va2, te2 = train_all, val_all, proc_t
        if use_scaler:
            _say("Fitting StandardScaler (stage 1 + stage 2) on train regions…")
            tr1, va1, te1 = _scale_splits(train_all, val_all, proc_t, s1_feats)
            tr2, va2, te2 = _scale_splits(train_all, val_all, proc_t, s2_feats)

        X1_tr, y_tr = _xy_single(tr1, s1_feats)
        X1_val, y_val = _xy_single(va1, s1_feats)
        X1_te, y_te = _xy_single(te1, s1_feats)

        X2_tr, y2_tr = _xy_single(tr2, s2_feats)
        X2_val, _ = _xy_single(va2, s2_feats)
        X2_te, _ = _xy_single(te2, s2_feats)

        y1_tr = (y_tr == 1).astype(int)
        sw1 = _balanced_sample_weight(y1_tr, max_ratio)
        rf_cw1 = ("balanced" if max_ratio is None
                  else _balanced_class_weights(y1_tr, max_ratio))

        m2 = y2_tr != 1
        X2_tr_s = X2_tr[m2]
        y2_s = y2_tr[m2]
        le2 = LabelEncoder().fit(y2_s)
        y2_enc = le2.transform(y2_s)
        sw2 = _balanced_sample_weight(y2_enc, max_ratio)
        rf_cw2 = ("balanced" if max_ratio is None
                  else _balanced_class_weights(y2_enc, max_ratio))

        for name, clf1, use_sw1 in _build_models(models, seed, rf_class_weight=rf_cw1):
            _say(f"Fitting {name} stage 1 (operational, {len(y1_tr):,} pts)…")
            if use_sw1:
                clf1.fit(X1_tr, y1_tr, sample_weight=sw1)
            else:
                clf1.fit(X1_tr, y1_tr)

            if name.startswith("XGBoost"):
                _, clf2, use_sw2 = _build_models(("xgb",), seed, rf_class_weight=rf_cw2)[0]
            else:
                _, clf2, use_sw2 = _build_models(("rf",), seed, rf_class_weight=rf_cw2)[0]

            _say(f"Fitting {name} stage 2 (value/spatial, {len(y2_s):,} pts)…")
            if use_sw2:
                clf2.fit(X2_tr_s, y2_enc, sample_weight=sw2)
            else:
                clf2.fit(X2_tr_s, y2_enc)

            val_pred, val_s1 = _predict_two_stage(
                clf1, clf2, le2, X1_val, X2_val,
            )
            te_pred, te_s1 = _predict_two_stage(
                clf1, clf2, le2, X1_te, X2_te,
            )
            out_models[name] = {
                "validation": _metrics(y_val, val_pred),
                "test": _metrics(y_te, te_pred),
                "stage1_validation": _stage1_binary_metrics(y_val, val_s1),
                "stage1_test": _stage1_binary_metrics(y_te, te_s1),
            }

    feature_cols = (
        spec["single_stage_features"]
        if not two_stage
        else spec["stage1_features"] + ["|"] + spec["stage2_features"]
    )

    return {
        "test_field": Path(test_spec[0]).stem,
        "test_value_col": test_spec[1],
        "train_fields": [Path(p).stem for p, _ in train_specs],
        "run_id": run_id,
        "ablation_label": spec["label"],
        "two_stage": two_stage,
        "use_scaler": use_scaler,
        "local_stats_mode": local_stats_mode,
        "feature_columns": feature_cols,
        "n_clusters": n_clusters,
        "seed": seed,
        "max_ratio": max_ratio,
        "n_train": int(len(y_tr)) if not two_stage else int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "train_classes_present": sorted(int(c) for c in np.unique(y_tr)),
        "test_classes_present": sorted(int(c) for c in np.unique(y_te)),
        "models": out_models,
    }


def run_lofo_ablation(
    files: List[str],
    value_cols: List[str],
    *,
    run_id: AblationRunId,
    models: Tuple[str, ...] = ("rf", "xgb"),
    n_clusters: int = config.LOFO_N_CLUSTERS,
    seed: int = config.SPLIT_RANDOM_STATE,
    max_ratio: Optional[float] = None,
    progress: ProgressCb = None,
) -> Dict:
    """Run every LOFO fold for an ablation protocol."""
    if len(files) < 2:
        raise ValueError("Need at least two fields for leave-one-field-out.")
    if len(files) != len(value_cols):
        raise ValueError("files and value_cols must have the same length.")

    spec = ablation_spec(run_id)
    folds = {}
    for i in range(len(files)):
        test_spec = (files[i], value_cols[i])
        train_specs = [(files[j], value_cols[j]) for j in range(len(files)) if j != i]
        if progress:
            progress(f"=== Fold {i + 1}/{len(files)}: hold out {Path(files[i]).stem} ===")
        folds[Path(files[i]).stem] = run_one_fold_ablation(
            test_spec, train_specs,
            run_id=run_id, models=models,
            n_clusters=n_clusters, seed=seed, max_ratio=max_ratio,
            progress=progress,
        )

    out = {
        "run_id": run_id,
        "ablation_label": spec["label"],
        "two_stage": spec["two_stage"],
        "use_scaler": spec["use_scaler"],
        "local_stats_mode": spec["local_stats_mode"],
        "files": [Path(f).stem for f in files],
        "value_cols": list(value_cols),
        "models": list(models),
        "n_clusters": n_clusters,
        "seed": seed,
        "feature_columns": folds[next(iter(folds))]["feature_columns"],
        "folds": folds,
    }
    if max_ratio is not None:
        out["max_ratio"] = max_ratio
    return out
