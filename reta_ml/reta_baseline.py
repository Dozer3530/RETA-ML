"""
Sprint 5 baseline: RETA-style sequential threshold filtering (Path B, headless).

This reimplements the *phase ordering* of RETA:
  Phase 1 Operational → Phase 2 Global → Phase 3 Local

It is intentionally conservative and uses sensible defaults, while keeping
the same 4 final classes as the GNN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from reta_ml import config
from reta_ml.preprocess import compute_geometry_and_transects


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    for col in config.TIME_COLUMN_CANDIDATES:
        if col not in df.columns:
            continue
        ser = df[col]
        numeric = pd.to_numeric(ser, errors="coerce")
        if numeric.notna().all():
            return numeric.values.astype(float)
        dt = pd.to_datetime(ser, errors="coerce")
        if dt.notna().any():
            return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()
    return np.arange(len(df), dtype=float)


def _medcouple(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return 0.0
    x_sorted = np.sort(x)
    median_x = np.median(x_sorted)
    z = x_sorted - median_x
    lower = z[z <= 0]
    upper = z[z >= 0]
    if len(lower) == 0 or len(upper) == 0:
        return 0.0
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=2000, replace=False)
        return _medcouple(x[idx])
    lower_g, upper_g = np.meshgrid(lower, upper)
    diff = upper_g - lower_g
    sum_vals = upper_g + lower_g
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = sum_vals / diff
    kernel[diff == 0] = np.sign(sum_vals[diff == 0])
    return float(np.median(kernel))


def _aoi_bounds(x: np.ndarray, *, k: float = 1.5) -> Tuple[float, float]:
    s = pd.Series(x).dropna()
    if len(s) < 5:
        return (float("-inf"), float("inf"))
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    mc = _medcouple(s.values)
    if mc >= 0:
        lo = q1 - k * np.exp(-4 * mc) * iqr
        hi = q3 + k * np.exp(3 * mc) * iqr
    else:
        lo = q1 - k * np.exp(-3 * mc) * iqr
        hi = q3 + k * np.exp(4 * mc) * iqr
    return (float(lo), float(hi))


def _infer_swath_width_m(df: pd.DataFrame) -> float:
    for col in ("swath_width", "Width"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v) and 2.0 < float(v.median()) < 40.0:
                return float(v.median())

    # Try convex hull area / distance if shapely is available.
    if "_x_m" in df.columns and "_y_m" in df.columns and "dist_m" in df.columns:
        try:
            from shapely.geometry import MultiPoint  # type: ignore

            pts = list(zip(df["_x_m"].astype(float).values, df["_y_m"].astype(float).values))
            hull = MultiPoint(pts).convex_hull
            area_m2 = float(hull.area)
            total_dist = float(pd.to_numeric(df["dist_m"], errors="coerce").fillna(0).sum())
            if total_dist > 0:
                sw = area_m2 / total_dist
                if 2.0 < sw < 40.0:
                    return float(sw)
        except Exception:
            pass

    return 10.0


def _detect_overlaps_grid_v7(
    df: pd.DataFrame,
    *,
    swath_width_m: float,
    cell_size_m: float = 1.0,
    min_speckle_size: int = 5,
) -> pd.Series:
    """
    Approximate RETA overlap detection: mark points whose grid cell band was
    previously occupied by a different transect.
    """
    x_m = df["_x_m"].astype(float).values
    y_m = df["_y_m"].astype(float).values
    x_m = x_m - float(np.nanmin(x_m))
    y_m = y_m - float(np.nanmin(y_m))

    gx = (x_m / cell_size_m).astype(int)
    gy = (y_m / cell_size_m).astype(int)
    bearings = np.radians(df["bearing_deg"].astype(float).values)
    tids = df.get("transect_id", pd.Series([-1] * len(df))).astype(float).values

    swath_cells = int(max(1, round(swath_width_m / cell_size_m)))

    grid_occupied: Dict[Tuple[int, int], int] = {}
    overlaps = np.zeros(len(df), dtype=bool)

    for i in range(len(df)):
        bx, by = int(gx[i]), int(gy[i])
        theta = float(bearings[i]) + (np.pi / 2.0)
        dx = float(np.cos(theta) * (1.0 / cell_size_m))
        dy = float(np.sin(theta) * (1.0 / cell_size_m))
        points_to_check = [(bx, by)]
        for frac in (0.25, 0.5):
            points_to_check.append(
                (int(bx + (dx * swath_cells * frac)), int(by + (dy * swath_cells * frac)))
            )
            points_to_check.append(
                (int(bx - (dx * swath_cells * frac)), int(by - (dy * swath_cells * frac)))
            )

        tid = int(tids[i]) if np.isfinite(tids[i]) else -1
        if tid < 0:
            tid = -(i + 1_000_000)

        is_ov = False
        for pt in points_to_check:
            if pt in grid_occupied:
                if grid_occupied[pt] != tid:
                    is_ov = True
            else:
                grid_occupied[pt] = tid
        overlaps[i] = is_ov

    # Speckle removal: keep only runs of >= min_speckle_size
    if overlaps.any() and min_speckle_size > 1:
        s = pd.Series(overlaps)
        change = s.ne(s.shift()).cumsum()
        counts = s.groupby(change).transform("count")
        overlaps = (s & (counts >= min_speckle_size)).values.astype(bool)

    return pd.Series(overlaps, index=df.index, name="is_overlap")


@dataclass(frozen=True)
class RetaBaselineParams:
    variable_col: str = config.MAIN_VARIABLE

    # Operational
    filter_maneuvers: bool = True
    filter_overlap: bool = True
    filter_speed: bool = True
    filter_accel: bool = True
    filter_dist_jump: bool = True
    auto_speed: bool = True
    min_speed_mps: float = 0.2
    max_speed_mps: float = 20.0
    accel_max_mps2: float = 1.5
    distance_jump_factor: float = 5.0
    manual_start_points: int = 0
    manual_end_points: int = 0

    # Global
    use_aoi: bool = True
    global_outlier_margin: float = 0.3
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    # Local
    use_local_std: bool = True
    local_std_sigma: float = 3.0
    grid_cell_multiplier: float = 3.0
    use_cross_track: bool = True


def reta_threshold_baseline_predict(
    df_raw: pd.DataFrame,
    *,
    params: Optional[RetaBaselineParams] = None,
) -> pd.DataFrame:
    """
    Return a copy of df_raw with `filtering_category_reta_pred` and intermediate
    columns used by the baseline.
    """
    params = params or RetaBaselineParams()
    var_col = params.variable_col

    # Geometry + transects (Path B, no QGIS)
    df = compute_geometry_and_transects(df_raw)
    if df.empty:
        out = df_raw.copy()
        out["filtering_category_reta_pred"] = np.nan
        return out

    # Speed/accel derived from time
    t = _time_seconds(df)
    time_diff = np.empty(len(df), dtype=float)
    time_diff[0] = 0.0
    time_diff[1:] = np.abs(np.diff(t))
    safe_time = np.where(time_diff > 0, time_diff, np.nan)
    speed = (df["dist_m"].astype(float).values / safe_time)
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    accel = np.abs(np.diff(speed, prepend=speed[0])) / safe_time
    accel = np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0)
    df["time_diff"] = time_diff
    df["speed_mps"] = speed
    df["accel_mps2"] = accel

    n = len(df)
    op_flag = np.zeros(n, dtype=bool)
    gl_flag = np.zeros(n, dtype=bool)
    lo_flag = np.zeros(n, dtype=bool)

    # --- Phase 1: Operational ---
    if params.filter_maneuvers:
        op_flag |= df["is_turn"].fillna(False).astype(bool).values
        op_flag |= df["is_short"].fillna(False).astype(bool).values

    if params.filter_overlap:
        sw = _infer_swath_width_m(df)
        df["is_overlap"] = _detect_overlaps_grid_v7(df, swath_width_m=sw)
        op_flag |= df["is_overlap"].fillna(False).astype(bool).values

    if params.filter_speed:
        min_s, max_s = params.min_speed_mps, params.max_speed_mps
        if params.auto_speed:
            speeds = pd.Series(df["speed_mps"]).astype(float)
            valid = speeds[(speeds > 0.1) & (speeds < 40.0)]
            if len(valid):
                min_s = float(max(0.1, valid.quantile(0.05) * 0.8))
                max_s = float(min(30.0, valid.quantile(0.99) * 1.2))
        op_flag |= df["speed_mps"].astype(float).values < float(min_s)
        op_flag |= df["speed_mps"].astype(float).values > float(max_s)

    if params.filter_accel:
        op_flag |= df["accel_mps2"].astype(float).values > float(params.accel_max_mps2)

    if params.filter_dist_jump:
        med = float(pd.to_numeric(df["dist_m"], errors="coerce").median())
        if np.isfinite(med) and med > 0:
            op_flag |= df["dist_m"].astype(float).values > float(params.distance_jump_factor * med)

    # Manual delays (optional)
    if (params.manual_start_points > 0 or params.manual_end_points > 0) and (var_col in df.columns):
        tids = df.get("transect_id", pd.Series([-1] * n)).astype(int).values
        for tid in np.unique(tids):
            if tid <= 0:
                continue
            idx = np.where(tids == tid)[0]
            if len(idx) == 0:
                continue
            if params.manual_start_points > 0:
                k = min(int(params.manual_start_points), len(idx))
                op_flag[idx[:k]] = True
            if params.manual_end_points > 0:
                k = min(int(params.manual_end_points), len(idx))
                op_flag[idx[-k:]] = True

    # --- Phase 2: Global outliers (computed on Operational-clean points only) ---
    if var_col in df.columns:
        op_clean = ~op_flag
        vals = pd.to_numeric(df[var_col], errors="coerce").values.astype(float)
        v_clean = vals[op_clean & np.isfinite(vals)]

        min_v = params.min_value
        max_v = params.max_value
        if params.use_aoi and len(v_clean):
            lo, hi = _aoi_bounds(v_clean)
            margin = float(np.clip(params.global_outlier_margin, 0.0, 1.0))
            min_v = lo - lo * margin
            max_v = hi + hi * margin

        if min_v is not None:
            gl_flag |= op_clean & (vals < float(min_v))
        if max_v is not None:
            gl_flag |= op_clean & (vals > float(max_v))

    # --- Phase 3: Local outliers (computed on Op+Global-clean points only) ---
    clean_for_local = ~(op_flag | gl_flag)

    if params.use_local_std and (var_col in df.columns) and np.any(clean_for_local):
        sw = _infer_swath_width_m(df)
        cell_size = float(max(0.1, sw * float(params.grid_cell_multiplier)))
        lx = (df["_x_m"].astype(float).values / cell_size).astype(int)
        ly = (df["_y_m"].astype(float).values / cell_size).astype(int)
        vals = pd.to_numeric(df[var_col], errors="coerce").values.astype(float)

        tmp = pd.DataFrame({"lx": lx, "ly": ly, "v": vals, "clean": clean_for_local})
        stats = (
            tmp.loc[tmp["clean"] & np.isfinite(tmp["v"])]
            .groupby(["lx", "ly"])["v"]
            .agg(["mean", "std"])
        )
        tmp = tmp.join(stats, on=["lx", "ly"])
        std = tmp["std"].fillna(0.0).values.astype(float)
        mean = tmp["mean"].values.astype(float)
        abnormal = np.isfinite(vals) & (std > 0) & (np.abs(vals - mean) > float(params.local_std_sigma) * std)
        lo_flag |= clean_for_local & abnormal

    if params.use_cross_track and (var_col in df.columns) and ("transect_id" in df.columns):
        vals = pd.to_numeric(df[var_col], errors="coerce").values.astype(float)
        tmp = df.copy()
        tmp["_v"] = vals
        tmp["_clean"] = clean_for_local & np.isfinite(vals) & (tmp["transect_id"].astype(float) > 0)
        med = tmp.loc[tmp["_clean"]].groupby("transect_id")["_v"].median()
        if len(med) >= 5:
            roll = med.rolling(window=5, center=True).median()
            ratio = med / roll.replace(0, np.nan)
            bad = ratio[(ratio > 1.5) | (ratio < 0.66)].index.astype(int).tolist()
            if bad:
                lo_flag |= clean_for_local & df["transect_id"].astype(int).isin(bad).values

    # Final category assignment (sequential priority)
    pred = np.full(n, "Clean", dtype=object)
    pred[lo_flag] = "Local Outlier"
    pred[gl_flag] = "Global Outlier"
    pred[op_flag] = "Operational Error"
    df["filtering_category_reta_pred"] = pred
    return df

