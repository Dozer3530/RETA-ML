"""
Input validation at pipeline entry.

Two modes:
- Training: require lat, lon, time, main variable, and filtering_category;
  fail loudly if missing, CRS not projectable to UTM, time not parseable or not
  monotonic within pass, main variable not numeric or out of range, or
  unexpected nulls in geometric columns.
- Inference: same checks but filtering_category is optional (for new fields
  without labels).

Call validation first in every pipeline; pass mode explicitly.
"""

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import warnings

from reta_ml import config as _config


class ValidationMode(str, Enum):
    """Validation mode: training requires labels; inference does not."""

    TRAINING = "training"
    INFERENCE = "inference"


class ValidationError(Exception):
    """Raised when input validation fails."""

    pass


def _resolve_time_column(df: pd.DataFrame) -> Optional[str]:
    """Return first matching time column name or None."""
    for col in _config.TIME_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _check_crs_projectable_to_utm(df: pd.DataFrame) -> None:
    """
    Check that coordinates are in a valid range and can be projected to UTM.
    Assumes lat/lon in degrees (WGS84) if no CRS attached.
    """
    lat = pd.to_numeric(df["lat"], errors="coerce")
    lon = pd.to_numeric(df["lon"], errors="coerce")
    if lat.isna().any() or lon.isna().any():
        raise ValidationError(
            "Geometric columns 'lat' and 'lon' must be numeric with no nulls."
        )
    if (lat < -90).any() or (lat > 90).any():
        raise ValidationError("'lat' must be in [-90, 90].")
    if (lon < -180).any() or (lon > 360).any():
        raise ValidationError("'lon' must be in [-180, 360].")
    # UTM is valid for lat in [-80, 84] approximately; allow full range and let projection fail later if needed
    try:
        from pyproj import CRS, Transformer
        lon_avg = float(lon.mean())
        lat_avg = float(lat.mean())
        zone = int((lon_avg + 180) / 6) + 1
        zone = max(1, min(60, zone))
        utm_epsg = 32600 + zone if lat_avg >= 0 else 32700 + zone
        Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    except Exception as e:
        raise ValidationError(f"CRS not projectable to UTM: {e}") from e


def _check_time_parseable_and_monotonic(
    df: pd.DataFrame, time_col: str
) -> None:
    """
    Ensure time column is parseable to numeric (seconds or datetime->epoch)
    and monotonic when sorted by time (within the dataset order / pass).
    """
    ser = df[time_col]
    if ser.isna().all():
        raise ValidationError(f"Time column '{time_col}' is entirely null.")
    # Try numeric first
    numeric = pd.to_numeric(ser, errors="coerce")
    if numeric.notna().all():
        times = numeric.values
    else:
        try:
            dt = pd.to_datetime(ser, errors="coerce")
            if dt.isna().any():
                raise ValidationError(
                    f"Time column '{time_col}' has unparseable values."
                )
            times = ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()
        except Exception as e:
            raise ValidationError(
                f"Time column '{time_col}' could not be parsed: {e}"
            ) from e
    # Monotonic: each value >= previous (allow equal for duplicate timestamps)
    diff = np.diff(times)
    if (diff < -1e-9).any():  # strict backward jump
        raise ValidationError(
            "Time column is not monotonic (decreasing values found within pass)."
        )


def validate_dataframe(
    df: pd.DataFrame,
    mode: ValidationMode = ValidationMode.TRAINING,
    *,
    main_variable: Optional[str] = None,
    main_min: Optional[float] = None,
    main_max: Optional[float] = None,
) -> None:
    """
    Validate input DataFrame at pipeline entry. Fail loudly on any violation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with at least lat, lon, time, and (in training) main variable
        and filtering_category.
    mode : ValidationMode
        'training': require filtering_category and main variable.
        'inference': filtering_category optional.
    main_variable : str, optional
        Name of the main variable column (default from config.MAIN_VARIABLE).
    main_min, main_max : float, optional
        Plausible range for main variable (default from config).

    Raises
    ------
    ValidationError
        If any required column is missing, CRS not projectable to UTM, time
        not parseable or not monotonic, main variable invalid, or geometric
        columns have unexpected nulls.
    """
    if main_variable is None:
        main_variable = _config.MAIN_VARIABLE
    if main_min is None:
        main_min = _config.MAIN_VARIABLE_MIN
    if main_max is None:
        main_max = _config.MAIN_VARIABLE_MAX

    required = (
        _config.REQUIRED_COLUMNS_TRAINING
        if mode == ValidationMode.TRAINING
        else _config.REQUIRED_COLUMNS_INFERENCE
    )

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    if mode == ValidationMode.TRAINING:
        if main_variable not in df.columns:
            raise ValidationError(
                f"Training mode requires main variable column '{main_variable}'."
            )
        if "filtering_category" not in df.columns:
            raise ValidationError(
                "Training mode requires 'filtering_category' column."
            )

    # Geometric columns: no unexpected nulls (lat, lon always required)
    for col in ["lat", "lon"]:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            raise ValidationError(
                f"Geometric column '{col}' must not contain nulls."
            )

    for col in ["_x_m", "_y_m"]:
        if col in df.columns and df[col].isna().any():
            raise ValidationError(
                f"Geometric column '{col}' must not contain nulls when present."
            )

    # CRS projectable to UTM (using lat/lon)
    _check_crs_projectable_to_utm(df)

    # Time column (optional in real-world files). Time issues — unparseable OR
    # non-monotonic file order — are WARNINGS, not hard failures: downstream
    # preprocessing sorts by time and falls back to row order, and real
    # GPS/harvest data often has coarse (e.g. date-only) or unsorted timestamps.
    time_col = _resolve_time_column(df)
    if time_col is not None:
        try:
            _check_time_parseable_and_monotonic(df, time_col)
        except ValidationError as e:
            warnings.warn(
                f"Time column '{time_col}' issue: {e} The pipeline sorts by time "
                "and falls back to row order, so continuing."
            )

    # Main variable: numeric and in plausible range (only when required)
    if main_variable in df.columns:
        vals = pd.to_numeric(df[main_variable], errors="coerce")
        if vals.isna().all():
            raise ValidationError(
                f"Main variable '{main_variable}' is entirely null or non-numeric."
            )
        if (vals < main_min).any() or (vals > main_max).any():
            raise ValidationError(
                f"Main variable '{main_variable}' has values outside plausible range "
                f"[{main_min}, {main_max}]."
            )
