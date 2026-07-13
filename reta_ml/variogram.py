"""
Directional variogram for spatial sensor data.

Yield (and ECa) data has **geometric anisotropy**: spatial continuity is
longer **along-track** (harvesting direction) than **cross-track**.  We
compute a directional variogram **perpendicular** to the dominant harvesting
direction and use its (shorter) range to set the spatial neighbourhood
radius *R*.

If the experimental variogram shows nested structure (two plateaux), we use
the **shorter** (inner) range — the first plateau captures within-management-
zone variability, which is the relevant scale for outlier filtering.

Public API
----------
- ``dominant_direction``  : estimate the main harvesting bearing from data.
- ``directional_variogram``: compute empirical semi-variance in a given
  angular band.
- ``fit_variogram``       : fit a spherical model and extract range.
- ``suggest_R``           : convenience wrapper returning R and optional K.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dominant_direction(
    df: pd.DataFrame,
    bearing_col: str = "bearing_deg",
) -> float:
    """
    Estimate the dominant harvesting direction (0–180°) from bearing_deg.

    Uses circular statistics on doubled angles so that 0° and 180° (opposite
    passes) map to the same direction.
    """
    bearings = df[bearing_col].dropna().values
    if len(bearings) == 0:
        return 0.0
    theta2 = np.radians(bearings * 2)
    mean_sin = np.mean(np.sin(theta2))
    mean_cos = np.mean(np.cos(theta2))
    dominant = (np.degrees(np.arctan2(mean_sin, mean_cos)) / 2) % 180
    return float(dominant)


def _angular_distance(a: float, b: float) -> float:
    """Smallest angle between two directions in [0, 180]."""
    d = abs(a - b) % 360
    if d > 180:
        d = 360 - d
    return d


# ---------------------------------------------------------------------------
# Empirical variogram
# ---------------------------------------------------------------------------

def directional_variogram(
    df: pd.DataFrame,
    direction_deg: float,
    *,
    value_col: Optional[str] = None,
    n_lags: int = 20,
    max_dist: Optional[float] = None,
    tolerance_deg: float = 22.5,
    coord_cols: Tuple[str, str] = ("_x_m", "_y_m"),
    max_pairs: int = 500_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical directional semi-variogram.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed table with coordinates and a value column.
    direction_deg : float
        Azimuth (degrees, 0 = north, clockwise) for the variogram
        direction.  To get the **perpendicular** range, pass
        ``dominant_direction + 90``.
    value_col : str, optional
        Column with the variable of interest (default: config.MAIN_VARIABLE).
    n_lags : int
        Number of lag bins.
    max_dist : float, optional
        Maximum lag distance (metres).  Default: half the extent.
    tolerance_deg : float
        Angular tolerance around the target direction (±).
    coord_cols : tuple[str, str]
        Metric coordinate columns.
    max_pairs : int
        Cap on the number of point-pairs evaluated (subsample if exceeded).

    Returns
    -------
    (lags, semivariance, counts) : tuple[ndarray, ndarray, ndarray]
        Bin centres (m), empirical semi-variance per bin, number of pairs
        per bin.
    """
    from reta_ml import config as _cfg

    if value_col is None:
        value_col = _cfg.MAIN_VARIABLE

    cx, cy = coord_cols
    mask = df[value_col].notna() & df[cx].notna() & df[cy].notna()
    sub = df.loc[mask].reset_index(drop=True)
    if len(sub) < 2:
        return np.array([]), np.array([]), np.array([])

    x = sub[cx].values.astype(float)
    y = sub[cy].values.astype(float)
    z = sub[value_col].values.astype(float)

    if max_dist is None:
        extent = max(x.max() - x.min(), y.max() - y.min())
        max_dist = extent / 2.0
    if max_dist <= 0:
        return np.array([]), np.array([]), np.array([])

    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lag_centres = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    gamma = np.zeros(n_lags, dtype=float)
    counts = np.zeros(n_lags, dtype=int)

    n = len(sub)
    if n * (n - 1) // 2 > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=int(np.sqrt(2 * max_pairs)), replace=False)
        x, y, z = x[idx], y[idx], z[idx]
        n = len(idx)

    direction_rad = np.radians(direction_deg)
    tol_rad = np.radians(tolerance_deg)

    for i in range(n):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        dists = np.sqrt(dx * dx + dy * dy)
        angles = np.arctan2(dx, dy)  # azimuth: 0=north, CW

        angle_diff = np.abs(angles - direction_rad)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
        angle_diff = np.minimum(angle_diff, np.pi - angle_diff)

        in_band = angle_diff <= tol_rad
        dz2 = (z[i + 1:] - z[i]) ** 2

        for lag_idx in range(n_lags):
            in_lag = in_band & (dists >= lag_edges[lag_idx]) & (dists < lag_edges[lag_idx + 1])
            if np.any(in_lag):
                gamma[lag_idx] += dz2[in_lag].sum()
                counts[lag_idx] += int(in_lag.sum())

    valid = counts > 0
    gamma[valid] = gamma[valid] / (2.0 * counts[valid])

    return lag_centres, gamma, counts


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------

def _spherical_model(h: np.ndarray, nugget: float, sill: float, range_m: float) -> np.ndarray:
    """Spherical variogram model."""
    r = np.clip(h / range_m, 0, 1)
    return nugget + (sill - nugget) * (1.5 * r - 0.5 * r ** 3) * (h <= range_m) + (sill) * (h > range_m)


def fit_variogram(
    lags: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray,
) -> Dict[str, float]:
    """
    Fit a spherical variogram model to the empirical semi-variance.

    Uses weighted least-squares with counts as weights.  Returns dict with
    ``nugget``, ``sill``, ``range_m``.  If the data show nested structure
    (two plateaux), the returned range corresponds to the **shorter** inner
    range.

    Parameters
    ----------
    lags, gamma, counts : ndarray
        Output of :func:`directional_variogram`.

    Returns
    -------
    dict
        ``{"nugget": float, "sill": float, "range_m": float}``
    """
    valid = (counts > 0) & np.isfinite(gamma)
    if valid.sum() < 3:
        return {"nugget": 0.0, "sill": 0.0, "range_m": 0.0}

    h = lags[valid]
    g = gamma[valid]
    w = counts[valid].astype(float)

    sill_est = float(np.median(g[-max(1, len(g) // 4):]))
    nugget_est = float(g[0]) if len(g) > 0 else 0.0

    first_above = np.where(g >= 0.95 * sill_est)[0]
    if len(first_above) > 0:
        range_est = float(h[first_above[0]])
    else:
        range_est = float(h[-1])

    # Detect nested structure: look for a local plateau followed by another rise
    if len(g) >= 6:
        mid = len(g) // 2
        first_half_max = np.max(g[:mid])
        second_half_max = np.max(g[mid:])
        if second_half_max > 1.3 * first_half_max:
            inner_plateau = np.where(g[:mid] >= 0.85 * first_half_max)[0]
            if len(inner_plateau) > 0:
                range_est = float(h[inner_plateau[0]])
                sill_est = float(first_half_max)

    try:
        from scipy.optimize import curve_fit

        def _model(h_arr, nug, sl, rng):
            rng = max(rng, 1e-3)
            return _spherical_model(h_arr, nug, sl, rng)

        popt, _ = curve_fit(
            _model,
            h,
            g,
            p0=[nugget_est, sill_est, range_est],
            sigma=1.0 / np.sqrt(w),
            bounds=([0, 0, 1e-3], [sill_est * 2, sill_est * 3, h[-1] * 2]),
            maxfev=5000,
        )
        nugget_est, sill_est, range_est = float(popt[0]), float(popt[1]), float(popt[2])
    except Exception:
        pass

    return {
        "nugget": nugget_est,
        "sill": sill_est,
        "range_m": range_est,
    }


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def suggest_R(
    df: pd.DataFrame,
    *,
    value_col: Optional[str] = None,
    n_lags: int = 20,
    r_factor: Tuple[float, float] = (0.5, 1.0),
    spacing_m: Optional[float] = None,
) -> Dict[str, object]:
    """
    End-to-end helper: detect dominant direction → compute perpendicular
    variogram → fit model → return suggested *R* and optional *K*.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with ``_x_m``, ``_y_m``, ``bearing_deg``,
        and the main variable.
    value_col : str, optional
        Main variable column.
    n_lags : int
        Number of lag bins for the variogram.
    r_factor : tuple[float, float]
        Multiply the fitted range by these factors to give a
        (R_low, R_high) suggestion.
    spacing_m : float, optional
        Typical point spacing (metres); if provided, K ≈ range / spacing.

    Returns
    -------
    dict
        Keys: ``dominant_direction_deg``, ``perpendicular_deg``,
        ``variogram_params`` (nugget, sill, range_m),
        ``suggested_R_low``, ``suggested_R_high``, ``suggested_K`` (or None).
    """
    dom = dominant_direction(df)
    perp = (dom + 90) % 180

    lags, gamma, counts = directional_variogram(
        df, perp, value_col=value_col, n_lags=n_lags,
    )

    params = fit_variogram(lags, gamma, counts)
    rng = params["range_m"]

    r_low = r_factor[0] * rng
    r_high = r_factor[1] * rng

    k_suggest = None
    if spacing_m is not None and spacing_m > 0 and rng > 0:
        k_suggest = max(1, int(round(rng / spacing_m)))

    return {
        "dominant_direction_deg": dom,
        "perpendicular_deg": perp,
        "variogram_params": params,
        "suggested_R_low": r_low,
        "suggested_R_high": r_high,
        "suggested_K": k_suggest,
    }
