"""
Path B: Pure Python/GeoPandas reimplementation of geometry and transects.

Produces _x_m, _y_m, dist_m, bearing_deg, transect_id, is_turn, is_short, and
ML feature table with speed/acceleration, swath-scaled multi-band local
mean/std/z-score (2w/3w/4w), ratio to transect mean, and a field-level z-score.
Uses KD-tree/BallTree for radius queries. The legacy speed_diff_ratio is not
produced.
"""

import logging
import re
from typing import Literal, Optional, Tuple

LocalStatsMode = Literal["multiband", "fixed", "fixedbands"]

import numpy as np
import pandas as pd

from reta_ml import config
from reta_ml.normalize import apply_motion_rel

logger = logging.getLogger(__name__)

# Time column must resolve to near per-point timestamps for sort-by-time ordering.
# Calendar dates (few unique values) scramble spatial sequence — see Yield field.
_MIN_UNIQUE_TIME_VALUES = 100
_MIN_UNIQUE_TIME_FRACTION = 0.10

_SWATH_COL_PATTERN = re.compile(r"swath|swth|width", re.IGNORECASE)
_SWATH_COL_EXCLUDE = re.compile(r"category|score|outlier|bbox", re.IGNORECASE)
_FEET_IN_NAME = re.compile(r"(?:^|_)(?:f|ft|feet)(?:$|_)", re.IGNORECASE)
_METERS_IN_NAME = re.compile(r"(?:^|_)(?:m|meter|metre)(?:$|_)", re.IGNORECASE)
_FEET_TO_M = 0.3048


def _project_to_utm(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project lat/lon to meters (UTM). Uses pyproj; no QGIS.

    If _x_m and _y_m already exist and are valid, use them. Otherwise
    compute from lat, lon (assumed WGS84).
    """
    if "_x_m" in df.columns and "_y_m" in df.columns:
        x = np.asarray(df["_x_m"], dtype=float)
        y = np.asarray(df["_y_m"], dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all():
            return x, y
    lon = np.asarray(df["lon"], dtype=float)
    lat = np.asarray(df["lat"], dtype=float)
    try:
        from pyproj import Transformer
    except ImportError as e:
        raise ImportError(
            "preprocess requires pyproj for UTM projection. Install with: pip install pyproj"
        ) from e
    zone = int((float(np.nanmean(lon)) + 180) / 6) + 1
    zone = max(1, min(60, zone))
    lat_avg = float(np.nanmean(lat))
    utm_epsg = 32600 + zone if lat_avg >= 0 else 32700 + zone
    trans = Transformer.from_crs(
        "EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True
    )
    x, y = trans.transform(lon, lat)
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _compute_dist_bearing(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Euclidean dist_m and bearing_deg (0–360) from consecutive points."""
    prev_x = np.roll(x, 1)
    prev_y = np.roll(y, 1)
    prev_x[0] = x[0]
    prev_y[0] = y[0]
    dx = x - prev_x
    dy = y - prev_y
    dist_m = np.sqrt(dx * dx + dy * dy)
    dist_m[0] = 0.0
    # Bearing: (90 - arctan2(dy, dx)) in degrees, then mod 360
    bearing_deg = (90 - np.degrees(np.arctan2(dy, dx))) % 360
    bearing_deg = np.where(np.isnan(bearing_deg), 0.0, bearing_deg)
    return dist_m, bearing_deg


def _identify_transects_v7(
    df: pd.DataFrame,
    turn_threshold_deg: float = 45.0,
    min_len_m: float = 30.0,
    dist_break_m: float = 20.0,
    bearing_smooth_window: int = 3,
) -> None:
    """
    Segment into transects (V7 logic, matching QGIS DataCleaner).

    - Smooth bearing with rolling window, compute bearing_diff.
    - Break segment when bearing_diff > turn_threshold or dist_m > dist_break_m.
    - Valid transect = sum(dist_m) >= min_len_m → transect_id positive.
    - Short segment (5m < sum < min_len_m) → is_short=True, transect_id=-2.
    - Turn/very short → is_turn=True, transect_id=-1.
    Modifies df in place: adds transect_id, is_turn, is_short.
    """
    rads = np.radians(df["bearing_deg"].values)
    sin_avg = (
        pd.Series(np.sin(rads))
        .rolling(window=bearing_smooth_window, center=True)
        .mean()
        .values
    )
    cos_avg = (
        pd.Series(np.cos(rads))
        .rolling(window=bearing_smooth_window, center=True)
        .mean()
        .values
    )
    bearing_smooth = np.degrees(np.arctan2(sin_avg, cos_avg))
    bearing_smooth = np.where(np.isnan(bearing_smooth), 0.0, bearing_smooth)
    bearing_smooth = (bearing_smooth + 360) % 360

    bearing_diff = np.abs(np.diff(bearing_smooth, prepend=bearing_smooth[0]))
    bearing_diff = np.where(
        bearing_diff > 180, 360 - bearing_diff, bearing_diff
    )

    is_break = (
        (bearing_diff > turn_threshold_deg)
        | (df["dist_m"].values > dist_break_m)
    )
    segment_id = np.cumsum(is_break.astype(int))

    stats = (
        pd.DataFrame({"segment_id": segment_id, "dist_m": df["dist_m"].values})
        .groupby("segment_id")
        .agg(count=("dist_m", "count"), dist_sum=("dist_m", "sum"))
    )

    valid_ids = set(
        stats.index[stats["dist_sum"] >= min_len_m].tolist()
    )
    short_ids = set(
        stats.index[
            (stats["dist_sum"] < min_len_m) & (stats["dist_sum"] > 5)
        ].tolist()
    )

    transect_map = {}
    current_id = 1
    for seg_id in stats.index:
        if seg_id in valid_ids:
            transect_map[seg_id] = current_id
            current_id += 1
        elif seg_id in short_ids:
            transect_map[seg_id] = -2
        else:
            transect_map[seg_id] = -1

    raw_tid = np.array([transect_map.get(s, -1) for s in segment_id])
    df["transect_id"] = np.where(raw_tid > 0, raw_tid, -1)
    df["is_turn"] = raw_tid == -1
    df["is_short"] = raw_tid == -2


def compute_geometry_and_transects(
    df: pd.DataFrame,
    *,
    turn_threshold_deg: Optional[float] = None,
    min_len_m: Optional[float] = None,
    dist_break_m: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute metric coordinates, distance, bearing, and transect segmentation.

    Pure Python/GeoPandas; no QGIS. Applies missing-value policy: rows with
    null lat/lon are dropped before computation (geometric exclude).

    Parameters
    ----------
    df : pd.DataFrame
        Must have lat, lon. Optionally time (for ordering).
    turn_threshold_deg, min_len_m, dist_break_m : float, optional
        Override config defaults for transect detection.

    Returns
    ----------
    pd.DataFrame
        Copy of df with rows with missing geometric columns dropped, and
        _x_m, _y_m, dist_m, bearing_deg, transect_id, is_turn, is_short added.
    """
    out = df.copy()
    # Missing geometric → exclude point
    geom_ok = out["lat"].notna() & out["lon"].notna()
    out = out.loc[geom_ok].copy()
    if out.empty:
        # No georeferenced points: return an empty frame that still carries the
        # geometry schema so downstream feature/graph code does not KeyError.
        for c in ("_x_m", "_y_m", "dist_m", "bearing_deg"):
            out[c] = pd.Series(dtype=float)
        out["transect_id"] = pd.Series(dtype="int64")
        out["is_turn"] = pd.Series(dtype=bool)
        out["is_short"] = pd.Series(dtype=bool)
        return out

    # Sort by time when it has per-point (or near per-point) resolution.
    time_col = _first_time_column(out)
    if time_col is not None:
        n_rows = len(out)
        n_unique = int(out[time_col].nunique(dropna=False))
        if _time_column_usable_for_ordering(n_unique, n_rows):
            out = out.sort_values(time_col).reset_index(drop=True)
        else:
            logger.warning(
                "Low-resolution time column %r (%d unique / %d rows); "
                "keeping source row order for dist_m/bearing/transect geometry "
                "(motion timing will assume %.1f Hz).",
                time_col,
                n_unique,
                n_rows,
                1.0 / config.ASSUMED_SAMPLE_INTERVAL_S,
            )

    x, y = _project_to_utm(out)
    out["_x_m"] = x
    out["_y_m"] = y

    dist_m, bearing_deg = _compute_dist_bearing(x, y)
    out["dist_m"] = dist_m
    out["bearing_deg"] = bearing_deg

    _identify_transects_v7(
        out,
        turn_threshold_deg=turn_threshold_deg or config.TURN_THRESHOLD_DEG,
        min_len_m=min_len_m or config.MIN_TRANSECT_LEN_M,
        dist_break_m=dist_break_m or config.DIST_BREAK_M,
        bearing_smooth_window=config.BEARING_SMOOTH_WINDOW,
    )
    return out


def _first_time_column(df: pd.DataFrame) -> Optional[str]:
    """First configured time column present in *df*, or None."""
    for col in config.TIME_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _time_column_usable_for_ordering(n_unique: int, n_rows: int) -> bool:
    """True when a time column has enough distinct values to order points."""
    if n_rows <= 0:
        return False
    if n_unique < _MIN_UNIQUE_TIME_VALUES:
        return False
    if n_unique / n_rows < _MIN_UNIQUE_TIME_FRACTION:
        return False
    return True


def _assume_1hz_time_seconds(n_rows: int, index: pd.Index) -> pd.Series:
    """Synthetic monotonic time (seconds) with uniform ``ASSUMED_SAMPLE_INTERVAL_S``."""
    dt = config.ASSUMED_SAMPLE_INTERVAL_S
    return pd.Series(np.arange(n_rows, dtype=float) * dt, index=index, dtype=float)


def _time_to_seconds_series(df: pd.DataFrame) -> pd.Series:
    """Return a Series of time in seconds (epoch or numeric as-is).

    Skips time columns with too few unique values (e.g. calendar dates) — the
    same guard used for pass ordering. When no usable per-point timestamp exists,
    assumes uniform sampling at ``config.ASSUMED_SAMPLE_INTERVAL_S`` (default 1 Hz)
    in native row order so ``time_dt``, ``speed_mps``, and ``accel_mps2`` stay
    well-defined.
    """
    n_rows = len(df)
    hz = 1.0 / config.ASSUMED_SAMPLE_INTERVAL_S
    time_col = _first_time_column(df)
    if time_col is not None:
        ser = df[time_col]
        n_unique = int(ser.nunique(dropna=False))
        if _time_column_usable_for_ordering(n_unique, n_rows):
            numeric = pd.to_numeric(ser, errors="coerce")
            if numeric.notna().all():
                return numeric
            dt = pd.to_datetime(ser, errors="coerce")
            if dt.notna().any():
                return (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        logger.warning(
            "Low-resolution time column %r (%d unique / %d rows); "
            "assuming %.1f Hz sampling in row order for time_dt/speed/accel.",
            time_col,
            n_unique,
            n_rows,
            hz,
        )
    else:
        logger.warning(
            "No timestamp column found; assuming %.1f Hz sampling in row order "
            "for time_dt/speed/accel.",
            hz,
        )
    return _assume_1hz_time_seconds(n_rows, df.index)


def _swath_width_column_candidates(columns) -> list[str]:
    """Column names that may hold equipment swath/header width."""
    preferred = ("swath_width", "Width", "Swth_Wdth_")
    ordered: list[str] = [c for c in preferred if c in columns]
    for col in columns:
        if col in ordered:
            continue
        if _SWATH_COL_EXCLUDE.search(col):
            continue
        if _SWATH_COL_PATTERN.search(col):
            ordered.append(col)
    return ordered


def _swath_series_to_meters(values: pd.Series, column_name: str) -> pd.Series:
    """Normalize swath-width values to metres (GreenStar exports often use feet)."""
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return v
    med = float(v.median())
    name = column_name or ""
    if _METERS_IN_NAME.search(name):
        return v
    if _FEET_IN_NAME.search(name):
        return v * _FEET_TO_M
    # Values above ~15 are implausibly wide in metres for most combines (typical
    # 5–12 m) but normal in feet (e.g. GreenStar Swth_Wdth_ median 29.5 ft).
    if med > 15.0:
        return v * _FEET_TO_M
    return v


def infer_swath_width(df: pd.DataFrame) -> float:
    """Infer field swath width (metres) from data columns or path geometry.

    Priority: (1) median of a swath/width metadata column (fuzzy name match),
    converted to metres when values or column name indicate feet; (2) convex-hull
    area / total path length on ``_x_m``/``_y_m``, clamped to [2, 40] m;
    (3) ``config.DEFAULT_SWATH_WIDTH_M``. Returns a single scalar — swath width is
    a field-level property, not per-point.
    """
    for col in _swath_width_column_candidates(df.columns):
        v_m = _swath_series_to_meters(df[col], col)
        if len(v_m):
            med = float(v_m.median())
            if 2.0 < med < 40.0:
                return med

    if (
        "_x_m" in df.columns
        and "_y_m" in df.columns
        and "dist_m" in df.columns
        and len(df) >= 3
    ):
        x = pd.to_numeric(df["_x_m"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["_y_m"], errors="coerce").to_numpy(dtype=float)
        total_dist = float(
            pd.to_numeric(df["dist_m"], errors="coerce").fillna(0).sum()
        )
        if total_dist > 0:
            try:
                from scipy.spatial import ConvexHull

                pts = np.column_stack([x, y])
                pts = pts[np.isfinite(pts).all(axis=1)]
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    area_m2 = float(hull.volume)
                    sw = area_m2 / total_dist
                    return float(np.clip(sw, 2.0, 40.0))
            except Exception:
                pass

    return config.DEFAULT_SWATH_WIDTH_M


def _local_stats_balltree(
    xy: np.ndarray,
    values: np.ndarray,
    radius_m: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local mean, std, and z-score using BallTree radius query.
    Returns (local_mean, local_std, z_score); NaN where neighborhood too small.
    """
    from sklearn.neighbors import BallTree

    tree = BallTree(xy, metric="euclidean")
    n = len(values)
    local_mean = np.full(n, np.nan, dtype=float)
    local_std = np.full(n, np.nan, dtype=float)
    z_score = np.full(n, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return local_mean, local_std, z_score

    ind = tree.query_radius(xy, r=radius_m)
    for i in range(n):
        idx = ind[i]
        if len(idx) < min_neighbors:
            continue
        v = values[idx]
        v = v[np.isfinite(v)]
        if len(v) < min_neighbors:
            continue
        m, s = float(np.mean(v)), float(np.std(v))
        local_mean[i] = m
        local_std[i] = s if s > 0 else np.nan
        if s > 0 and np.isfinite(values[i]):
            z_score[i] = (float(values[i]) - m) / s
    return local_mean, local_std, z_score


def _transect_means(df: pd.DataFrame, var_col: str) -> pd.Series:
    """Per-transect mean of var_col (only valid transect_id > 0)."""
    valid = df["transect_id"] > 0
    if not valid.any():
        return pd.Series(dtype=float)
    return df.loc[valid].groupby("transect_id")[var_col].transform("mean")


def build_ml_feature_table(
    df: pd.DataFrame,
    main_variable: Optional[str] = None,
    *,
    local_radius_m: Optional[float] = None,
    local_min_neighbors: Optional[int] = None,
    local_stats_mode: LocalStatsMode = "multiband",
) -> pd.DataFrame:
    """
    Build ML feature table from geometry+transect DataFrame.

    Adds: bearing_diff, time_dt, speed_mps, accel_mps2, swath-scaled multi-band
    local_mean/std/z_score (2w/3w/4w), ratio_to_transect_mean, global_z.
    Does NOT add speed_diff_ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Must already have _x_m, _y_m, dist_m, bearing_deg, transect_id,
        is_turn, is_short (e.g. from compute_geometry_and_transects).
    main_variable : str, optional
        Main variable column name (default from config).
    local_radius_m : float, optional
        Used only when ``local_stats_mode='fixed'`` (default ``LOCAL_RADIUS_M``).
    local_min_neighbors : int, optional
        Override config minimum neighbors for BallTree local stats.
    local_stats_mode : {'multiband', 'fixed'}, optional
        ``multiband`` (default): 2×/3×/4× swath-scaled radii. ``fixed``: single
        radius at ``local_radius_m`` or ``LOCAL_RADIUS_M`` (legacy ablation).

    Returns
    ----------
    pd.DataFrame
        Table with ML feature columns; rows unchanged except new columns.
    """
    if main_variable is None:
        main_variable = config.MAIN_VARIABLE
    radius = local_radius_m or config.LOCAL_RADIUS_M
    min_n = local_min_neighbors or config.LOCAL_MIN_NEIGHBORS

    out = df.copy()

    # Empty input (e.g. all rows lacked valid lat/lon): return the full feature
    # schema with no rows instead of indexing missing columns / a 0-point BallTree.
    if out.empty:
        empty_cols = [
            "bearing_diff", "time_dt", "speed_mps", "accel_mps2",
            "speed_rel", "dist_rel", "accel_rel",
            "swath_width_m", "ratio_to_transect_mean",
            "global_z", "value_norm", "field_rel_spread",
        ]
        if local_stats_mode == "fixed":
            empty_cols.extend(["local_mean", "local_std", "z_score"])
        elif local_stats_mode == "fixedbands":
            for suffix in config.LOCAL_FIXED_BAND_SUFFIXES:
                empty_cols.extend([
                    f"local_mean_{suffix}", f"local_std_{suffix}", f"z_score_{suffix}",
                ])
        else:
            empty_cols.extend([
                "local_mean_2w", "local_std_2w", "z_score_2w",
                "local_mean_3w", "local_std_3w", "z_score_3w",
                "local_mean_4w", "local_std_4w", "z_score_4w",
            ])
        for c in empty_cols:
            out[c] = pd.Series(dtype=float)
        return out

    # Bearing diff (smooth then diff, same as transect logic for consistency)
    rads = np.radians(out["bearing_deg"].values)
    sin_avg = (
        pd.Series(np.sin(rads))
        .rolling(window=config.BEARING_SMOOTH_WINDOW, center=True)
        .mean()
        .values
    )
    cos_avg = (
        pd.Series(np.cos(rads))
        .rolling(window=config.BEARING_SMOOTH_WINDOW, center=True)
        .mean()
        .values
    )
    bearing_smooth = np.degrees(np.arctan2(sin_avg, cos_avg))
    bearing_smooth = np.where(np.isnan(bearing_smooth), 0.0, bearing_smooth)
    bearing_smooth = (bearing_smooth + 360) % 360
    bdiff = np.abs(np.diff(bearing_smooth, prepend=bearing_smooth[0]))
    out["bearing_diff"] = np.where(bdiff > 180, 360 - bdiff, bdiff)

    # Time delta (seconds)
    time_sec = _time_to_seconds_series(out)
    out["time_dt"] = time_sec.diff().fillna(0).abs()

    # Speed (m/s) and acceleration (m/s^2): operational/motion signals used as
    # model features. A sensor-reported ground speed is preferred when present;
    # otherwise speed is derived from distance and time interval. These help
    # detect operational anomalies (maneuvers, abrupt speed changes, stops).
    if "speed_mps" in out.columns and pd.to_numeric(out["speed_mps"], errors="coerce").notna().any():
        speed = pd.to_numeric(out["speed_mps"], errors="coerce")
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = out["dist_m"].astype(float) / out["time_dt"].replace(0, np.nan)
    speed = speed.replace([np.inf, -np.inf], np.nan)
    out["speed_mps"] = speed
    # Acceleration (m/s^2): the RATE of change of speed, i.e. Δspeed / Δt. It is
    # signed (a sudden stop is a large negative value, a sudden start a large
    # positive one) so the model can distinguish braking from accelerating.
    # Dividing by time_dt is required for correct units under irregular GPS
    # sampling — a raw speed difference (m/s) conflates the sampling rate with
    # the motion. dt=0 (duplicate timestamps) -> NaN, filled with 0 downstream.
    with np.errstate(divide="ignore", invalid="ignore"):
        out["accel_mps2"] = speed.diff() / out["time_dt"].replace(0, np.nan)
    out["accel_mps2"] = out["accel_mps2"].replace([np.inf, -np.inf], np.nan)

    # Field-relative motion magnitudes (median of this table's rows — recomputed
    # per spatial region in recompute_value_stats for leakage-safe LOFO).
    out = apply_motion_rel(out)

    # Value-based and spatial-statistic features (BallTree neighbourhood stats,
    # transect ratio, global z, value_norm/field_rel_spread). Factored into a
    # helper so the SAME logic can be re-applied to a spatial sub-region after a
    # train/val split (see recompute_value_stats), keeping those per-field
    # statistics leakage-free.
    out = _compute_value_stats(
        out, main_variable, min_n,
        local_stats_mode=local_stats_mode,
        radius_m=radius,
    )

    return out


def _compute_value_stats(
    out: pd.DataFrame,
    main_variable: str,
    min_n: int,
    *,
    local_stats_mode: LocalStatsMode = "multiband",
    radius_m: Optional[float] = None,
) -> pd.DataFrame:
    """Compute the value/spatial-statistic feature columns IN PLACE on *out*.

    Columns: swath_width_m; local stats (multi-band or fixed-radius);
    ratio_to_transect_mean, global_z, value_norm, field_rel_spread.
    Every statistic is derived ONLY from the rows present in *out* — so when
    *out* is a single spatial region (e.g. a K-means train or validation region,
    or a held-out field), the features carry no information from rows outside
    that region. Requires _x_m/_y_m and transect_id to already be present.
    """
    swath_w = infer_swath_width(out)
    out["swath_width_m"] = swath_w

    xy = out[["_x_m", "_y_m"]].values.astype(float)
    if main_variable in out.columns:
        vals = pd.to_numeric(out[main_variable], errors="coerce").values
        if local_stats_mode == "fixed":
            r = radius_m if radius_m is not None else config.LOCAL_RADIUS_M
            lm, ls, zs = _local_stats_balltree(xy, vals, r, min_n)
            out["local_mean"] = lm
            out["local_std"] = ls
            out["z_score"] = zs
        elif local_stats_mode == "fixedbands":
            for r_m, suffix in zip(config.LOCAL_FIXED_BANDS_M,
                                   config.LOCAL_FIXED_BAND_SUFFIXES):
                lm, ls, zs = _local_stats_balltree(xy, vals, r_m, min_n)
                out[f"local_mean_{suffix}"] = lm
                out[f"local_std_{suffix}"] = ls
                out[f"z_score_{suffix}"] = zs
        else:
            for mult, suffix in ((2, "2w"), (3, "3w"), (4, "4w")):
                lm, ls, zs = _local_stats_balltree(
                    xy, vals, swath_w * mult, min_n
                )
                out[f"local_mean_{suffix}"] = lm
                out[f"local_std_{suffix}"] = ls
                out[f"z_score_{suffix}"] = zs
        # Ratio to transect mean
        tmean = _transect_means(out, main_variable)
        out["ratio_to_transect_mean"] = (
            out[main_variable].astype(float) / tmean.replace(0, np.nan)
        )
        # Field-level (global) z-score: value standardized within this field.
        # Unit-free, so it is comparable across different sensors/variables
        # (a yield field, an EC field and an elevation field all map to the
        # same scale). This captures *global* outliers without leaking the
        # variable's absolute units into the model.
        finite = vals[np.isfinite(vals)]
        if finite.size >= 2:
            g_mean = float(np.mean(finite))
            g_std = float(np.std(finite))
            if g_std <= 0:
                g_std = 1.0
            out["global_z"] = (
                pd.to_numeric(out[main_variable], errors="coerce").astype(float)
                - g_mean
            ) / g_std
        else:
            out["global_z"] = np.nan

        # Value-based model features (spatially-aware design). The value is
        # centered by THIS region's median (robust, level-invariant); the global
        # scale division happens later in normalization (fit on the training set
        # and saved). field_rel_spread carries the region's own IQR so the model
        # knows how variable this field is vs a typical one — it is likewise
        # divided by the saved global scale at normalization time.
        valf = pd.to_numeric(out[main_variable], errors="coerce").astype(float)
        if finite.size >= 2:
            field_median = float(np.median(finite))
            q75, q25 = np.percentile(finite, [75, 25])
            field_iqr = float(q75 - q25)
            if field_iqr <= 0:
                field_iqr = float(np.std(finite)) or 1.0
            out["value_norm"] = valf - field_median       # scaled later (global)
            out["field_rel_spread"] = float(field_iqr)    # scaled later (global)
        else:
            out["value_norm"] = np.nan
            out["field_rel_spread"] = np.nan
    else:
        out["ratio_to_transect_mean"] = np.nan
        out["global_z"] = np.nan
        out["value_norm"] = np.nan
        out["field_rel_spread"] = np.nan
        if local_stats_mode == "fixed":
            out["local_mean"] = np.nan
            out["local_std"] = np.nan
            out["z_score"] = np.nan
        elif local_stats_mode == "fixedbands":
            for suffix in config.LOCAL_FIXED_BAND_SUFFIXES:
                out[f"local_mean_{suffix}"] = np.nan
                out[f"local_std_{suffix}"] = np.nan
                out[f"z_score_{suffix}"] = np.nan
        else:
            for suffix in ("2w", "3w", "4w"):
                out[f"local_mean_{suffix}"] = np.nan
                out[f"local_std_{suffix}"] = np.nan
                out[f"z_score_{suffix}"] = np.nan

    return out


def recompute_value_stats(
    df: pd.DataFrame,
    main_variable: str = "value",
    *,
    local_radius_m: Optional[float] = None,
    local_min_neighbors: Optional[int] = None,
    local_stats_mode: LocalStatsMode = "multiband",
) -> pd.DataFrame:
    """Re-derive value/spatial and motion-relative features from *df*'s OWN rows only.

    Use after a spatial train/val split so that each region's swath-scaled
    local_mean/std, z_score bands, ratio_to_transect_mean, global_z,
    value_norm/field_rel_spread, and motion relatives (speed_rel, dist_rel,
    accel_rel) are computed exclusively from that region's points — no leakage
    across the split. Swath width is inferred from the region's own geometry
    (not from other regions or fields). Raw geometry/motion columns (dist_m,
    bearing*, time_dt, speed_mps, accel_mps2, transect flags) are left
    untouched. Returns a copy.
    """
    radius = local_radius_m or config.LOCAL_RADIUS_M
    min_n = local_min_neighbors or config.LOCAL_MIN_NEIGHBORS
    out = df.copy()
    if out.empty:
        return out
    out = _compute_value_stats(
        out, main_variable, min_n,
        local_stats_mode=local_stats_mode,
        radius_m=radius,
    )
    return apply_motion_rel(out)


def preprocess_pipeline(
    df: pd.DataFrame,
    main_variable: Optional[str] = None,
    *,
    apply_missing_value_policy: bool = True,
    local_stats_mode: LocalStatsMode = "multiband",
    local_radius_m: Optional[float] = None,
) -> pd.DataFrame:
    """
    Full preprocessing: geometry + transects + ML feature table.

    - Drops rows with missing lat/lon (geometric exclude).
    - Missing time → time_dt may be 0/NaN; segment still flagged by distance.
    - Missing main variable → row kept; local stats and ratio may be NaN (mask in loss later).
    Adds the full feature schema, including speed_mps and accel_mps2.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input with lat, lon, time, and (for training) main variable and filtering_category.
    main_variable : str, optional
        Main variable column (default config.MAIN_VARIABLE).
    apply_missing_value_policy : bool
        If True, drop rows with missing geometric columns before computation.

    Returns
    ----------
    pd.DataFrame
        ML feature table with the full set of feature columns.
    """
    if main_variable is None:
        main_variable = config.MAIN_VARIABLE
    step = compute_geometry_and_transects(df)
    return build_ml_feature_table(
        step,
        main_variable=main_variable,
        local_stats_mode=local_stats_mode,
        local_radius_m=local_radius_m,
    )
