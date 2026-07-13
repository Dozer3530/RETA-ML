"""
Data loader: read train/test and validation from data/; run validation
before any preprocessing.

Missing value policy:
- Missing time → distance fallback for gap detection; flag segment.
- Missing main variable → node stays in graph; mask in loss later.
- Missing geometric (lat/lon) → exclude point.
"""

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from reta_ml import config
from reta_ml.validate import validate_dataframe, ValidationMode


def _read_geofile(path: Path) -> pd.DataFrame:
    """
    Read a GeoPackage or GeoJSON into a DataFrame with lat, lon extracted.
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise ImportError(
            "Loading GPKG requires geopandas. Install with: pip install geopandas"
        ) from e
    gdf = gpd.read_file(path)
    gdf = gdf.copy()

    def _valid_lon_lat(lon: pd.Series, lat: pd.Series) -> bool:
        lon_n = pd.to_numeric(lon, errors="coerce")
        lat_n = pd.to_numeric(lat, errors="coerce")
        if lon_n.isna().any() or lat_n.isna().any():
            return False
        if (lat_n < -90).any() or (lat_n > 90).any():
            return False
        if (lon_n < -180).any() or (lon_n > 360).any():
            return False
        return True

    # Prefer existing explicit lon/lat columns if they look like degrees.
    if "lon" in gdf.columns and "lat" in gdf.columns and _valid_lon_lat(gdf["lon"], gdf["lat"]):
        pass
    else:
        # Try common alternative column names.
        candidates = [
            ("Longitude", "Latitude"),
            ("LONGITUDE", "LATITUDE"),
            ("Long", "Lat"),
            ("LONG", "LAT"),
        ]
        found = False
        for lon_col, lat_col in candidates:
            if lon_col in gdf.columns and lat_col in gdf.columns and _valid_lon_lat(gdf[lon_col], gdf[lat_col]):
                gdf["lon"] = pd.to_numeric(gdf[lon_col], errors="coerce")
                gdf["lat"] = pd.to_numeric(gdf[lat_col], errors="coerce")
                found = True
                break

        if not found and hasattr(gdf, "geometry") and gdf.geometry is not None:
            geom = gdf.geometry
            try:
                # If geometry is projected, reproject to EPSG:4326 first.
                if getattr(gdf, "crs", None) is not None and not gdf.crs.is_geographic:
                    geom = geom.to_crs("EPSG:4326")
            except Exception:
                pass
            gdf["lon"] = geom.x
            gdf["lat"] = geom.y

    return pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))


def load_table(
    path: Path,
    *,
    validate: bool = True,
    mode: ValidationMode = ValidationMode.TRAINING,
    main_variable: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a single table from path (GPKG/CSV/Parquet). Optionally validate.

    Parameters
    ----------
    path : Path
        Path to GeoPackage, CSV, or Parquet.
    validate : bool
        If True, run validate_dataframe before returning (fail loudly on error).
    mode : ValidationMode
        Training or inference (affects required columns).
    main_variable : str, optional
        Main variable column name for validation.

    Returns
    ----------
    pd.DataFrame
        Raw table; if validate=True it passes validation (no preprocessing).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".gpkg":
        df = _read_geofile(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    if validate:
        validate_dataframe(df, mode=mode, main_variable=main_variable or config.MAIN_VARIABLE)
    return df


def load_and_validate(
    train_path: Optional[Path] = None,
    validation_path: Optional[Path] = None,
    *,
    mode: ValidationMode = ValidationMode.TRAINING,
    main_variable: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train/test and validation from data/; run validation before any preprocessing.

    Parameters
    ----------
    train_path, validation_path : Path, optional
        Override config paths. If None, use config.DATA_DIR / config.TRAIN_TEST_FILE
        and config.DATA_DIR / config.VALIDATION_FILE. If a path does not exist,
        that DataFrame will be empty (so caller can skip validation file if missing).
    mode : ValidationMode
        Training or inference.
    main_variable : str, optional
        Main variable column name.

    Returns
    ----------
    (train_df, validation_df) : Tuple[pd.DataFrame, pd.DataFrame]
        Raw DataFrames after validation. Empty DataFrame if file not found.
    """
    main_variable = main_variable or config.MAIN_VARIABLE
    data_dir = config.DATA_DIR
    train_path = train_path or data_dir / config.TRAIN_TEST_FILE
    validation_path = validation_path or data_dir / config.VALIDATION_FILE

    train_df = pd.DataFrame()
    if train_path.exists():
        train_df = load_table(
            train_path, validate=True, mode=mode, main_variable=main_variable
        )

    val_df = pd.DataFrame()
    if validation_path.exists():
        val_df = load_table(
            validation_path, validate=True, mode=mode, main_variable=main_variable
        )

    return train_df, val_df


def print_filtering_category_distribution(
    df: pd.DataFrame,
    *,
    category_col: str = "filtering_category",
    warn_local_threshold: float = 0.02,
) -> None:
    """
    Print filtering_category counts and proportions. If Local Outlier < 2%,
    note that focal loss and class weights must be aggressive.

    Parameters
    ----------
    df : pd.DataFrame
        Table with category column (e.g. after load; may be empty).
    category_col : str
        Column name for class labels.
    warn_local_threshold : float
        If proportion of "Local Outlier" is below this, print warning.
    """
    if category_col not in df.columns or df.empty:
        print(f"[Class distribution] No column '{category_col}' or empty table.")
        return
    counts = df[category_col].value_counts()
    total = len(df)
    print("[Class distribution] Counts and proportions:")
    for cat, n in counts.items():
        pct = 100.0 * n / total
        print(f"  {cat}: {n} ({pct:.2f}%)")
    if "Local Outlier" in counts.index:
        pct_local = 100.0 * counts["Local Outlier"] / total
        if pct_local < 100.0 * warn_local_threshold:
            print(
                f"  NOTE: Local Outlier < {100 * warn_local_threshold:.0f}% — "
                "focal loss γ and class weights must be aggressive."
            )
