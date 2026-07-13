"""
Per-field normalization: fit scaler on **train only**, transform train / test /
validation.

Temporal order is preserved — rows are never shuffled.  The scaler object (a
scikit-learn ``StandardScaler``) can be serialized for inference on new fields.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from reta_ml import config


# Continuous model features to standardize. Excludes boolean/ID columns and
# absolute coordinates (graph-only). The geometry/motion columns get a normal
# global StandardScaler (their units are stable across fields). The value
# columns (value_norm, field_rel_spread) are handled specially: they share one
# robust GLOBAL scale and are NOT mean-centered (see _VALUE_SCALE_COLS).
_DEFAULT_NORM_COLS: List[str] = [
    "dist_m",
    "bearing_deg",
    "bearing_diff",
    "time_dt",
    "speed_mps",
    "accel_mps2",
    "value_norm",
    "field_rel_spread",
]

# Columns sharing a single robust GLOBAL value scale (fit on training). They are
# NOT mean-centered by the scaler: value_norm is already per-field median-centered
# in preprocessing, and field_rel_spread is a field IQR we only divide. Dividing
# both by the SAME global scale is what preserves cross-field variability — a
# noisy field keeps a wider value_norm and a larger field_rel_spread than a tight
# one, instead of every field being normalized to unit variance.
_VALUE_SCALE_COLS: List[str] = ["value_norm", "field_rel_spread"]

# Field-relative motion features: raw column → relative column (÷ region median).
MOTION_REL_RAW_COLS: Tuple[str, ...] = ("dist_m", "speed_mps", "accel_mps2")
MOTION_REL_FEATURE_COLS: Tuple[str, ...] = ("dist_rel", "speed_rel", "accel_rel")


def _robust_global_scale(values: np.ndarray) -> float:
    """Robust spread (IQR, then std fallback) of the finite values; >= eps."""
    v = values[np.isfinite(values)]
    if v.size >= 2:
        q75, q25 = np.percentile(v, [75, 25])
        s = float(q75 - q25)
        if s <= 0:
            s = float(np.std(v))
        return s if s > 0 else 1.0
    return 1.0


def fit_value_centering(train_df: pd.DataFrame, value_col: str = "value") -> Tuple[float, float]:
    """Robust per-field centering stats (median, IQR) from the TRAIN rows only.

    Used to (re)compute value_norm / field_rel_spread after a within-field split,
    so that a field's validation region does not leak into the per-field median
    used to center its train features. Returns (median, iqr); iqr falls back to
    std then 1.0.
    """
    v = pd.to_numeric(train_df[value_col], errors="coerce").to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size >= 2:
        median = float(np.median(finite))
        q75, q25 = np.percentile(finite, [75, 25])
        iqr = float(q75 - q25)
        if iqr <= 0:
            iqr = float(np.std(finite))
        if iqr <= 0:
            iqr = 1.0
    else:
        median, iqr = 0.0, 1.0
    return median, iqr


def apply_value_centering(
    df: pd.DataFrame,
    median: float,
    iqr: float,
    value_col: str = "value",
) -> pd.DataFrame:
    """Set value_norm = value - median and field_rel_spread = iqr on a copy of *df*.

    The global-scale division still happens later in :func:`transform_df`; this
    only fixes the per-field centering/spread to a chosen (train-only) origin.
    """
    out = df.copy()
    if value_col in out.columns:
        vv = pd.to_numeric(out[value_col], errors="coerce").astype(float)
        out["value_norm"] = vv - float(median)
        out["field_rel_spread"] = float(iqr)
    return out


def fit_motion_reference(df: pd.DataFrame) -> Dict[str, float]:
    """Robust per-region reference medians for raw motion columns (train rows only).

    Used to express speed/step/accel relative to a field or spatial region's
    typical operating scale — comparable across equipment types. Returns a map
    ``{raw_col: median}`` with a positive fallback when median is zero.
    """
    refs: Dict[str, float] = {}
    for raw in MOTION_REL_RAW_COLS:
        if raw not in df.columns:
            continue
        v = pd.to_numeric(df[raw], errors="coerce").to_numpy(dtype=float)
        finite = v[np.isfinite(v)]
        if finite.size >= 1:
            med = float(np.median(finite))
            refs[raw] = med if abs(med) > 1e-12 else 1.0
        else:
            refs[raw] = 1.0
    return refs


def apply_motion_rel(
    df: pd.DataFrame,
    refs: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Set ``dist_rel``, ``speed_rel``, ``accel_rel`` on a copy of *df*.

    Each relative column is ``raw / median(raw)`` where the median is taken from
    *refs* if provided, otherwise from the rows in *df* (field- or region-own
    statistic — never borrowed from another field). Raw motion columns are left
    unchanged.
    """
    out = df.copy()
    if refs is None:
        refs = fit_motion_reference(out)
    for raw, rel in zip(MOTION_REL_RAW_COLS, MOTION_REL_FEATURE_COLS):
        if raw not in out.columns:
            out[rel] = np.nan
            continue
        denom = float(refs.get(raw, 1.0))
        if denom <= 0:
            denom = 1.0
        v = pd.to_numeric(out[raw], errors="coerce").astype(float)
        out[rel] = v / denom
    return out


def _resolve_cols(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]],
) -> List[str]:
    """Return the intersection of requested columns and those present in *df*.

    The raw main variable is intentionally NOT auto-added: it is not a model
    feature (absolute units do not generalize), so it is left untouched in the
    table rather than being overwritten with standardized values.
    """
    candidates = feature_cols if feature_cols is not None else _DEFAULT_NORM_COLS
    return [c for c in candidates if c in df.columns]


def fit_scaler(
    train_df: pd.DataFrame,
    *,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[StandardScaler, List[str]]:
    """
    Fit a ``StandardScaler`` on the train split.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train split (possibly augmented).
    feature_cols : list[str], optional
        Columns to scale.  Default: ``_DEFAULT_NORM_COLS`` + main variable.

    Returns
    -------
    (scaler, cols) : tuple[StandardScaler, list[str]]
        Fitted scaler and the actual column list used.
    """
    cols = _resolve_cols(train_df, feature_cols)
    if not cols:
        raise ValueError("No normalizable columns found in train_df.")

    scaler = StandardScaler()
    values = train_df[cols].values.astype(float)
    finite_mask = np.isfinite(values)
    col_means = np.nanmean(values, axis=0)
    col_stds = np.nanstd(values, axis=0)
    col_stds[col_stds == 0] = 1.0

    # Value columns: one shared robust global scale, no mean-centering. Derived
    # from value_norm (the per-field-centered value) so the same physical scale
    # divides both value_norm and field_rel_spread, preserving cross-field
    # variability. Falls back to field_rel_spread if value_norm is absent.
    if any(c in cols for c in _VALUE_SCALE_COLS):
        src = "value_norm" if "value_norm" in cols else "field_rel_spread"
        gscale = _robust_global_scale(train_df[src].to_numpy(dtype=float))
        for c in _VALUE_SCALE_COLS:
            if c in cols:
                j = cols.index(c)
                col_means[j] = 0.0
                col_stds[j] = gscale

    scaler.mean_ = col_means
    scaler.scale_ = col_stds
    scaler.var_ = col_stds ** 2
    scaler.n_features_in_ = len(cols)
    scaler.n_samples_seen_ = np.sum(finite_mask, axis=0)
    scaler.feature_names_in_ = np.array(cols)

    return scaler, cols


def transform_df(
    df: pd.DataFrame,
    scaler: StandardScaler,
    cols: List[str],
) -> pd.DataFrame:
    """
    Apply a fitted scaler to *df*, returning a copy.

    NaN values stay NaN.  Temporal order is preserved (no row shuffling).
    """
    out = df.copy()
    values = out[cols].values.astype(float)
    scaled = (values - scaler.mean_) / scaler.scale_
    scaled = np.where(np.isfinite(values), scaled, np.nan)
    out[cols] = scaled
    return out


def normalize_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    *,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """
    Convenience: fit on train, transform all splits.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Train and test splits (train may already be augmented).
    val_df : pd.DataFrame or None
        Validation set; transformed if provided.
    feature_cols : list[str], optional
        Override columns to normalize.

    Returns
    -------
    (train_n, test_n, val_n, scaler) :
        Normalized copies of each split (val_n is ``None`` if *val_df* was ``None``),
        plus the fitted scaler for serialization / inference.
    """
    scaler, cols = fit_scaler(train_df, feature_cols=feature_cols)
    train_n = transform_df(train_df, scaler, cols)
    test_n = transform_df(test_df, scaler, cols)
    val_n = transform_df(val_df, scaler, cols) if val_df is not None else None
    return train_n, test_n, val_n, scaler
