"""
Graph construction for heterogeneous GNN: temporal, spatial, and transect edges.

Builds a single graph per field from a preprocessed DataFrame.  Augmentation
must be applied **before** graph construction so that the edge structure
matches the (possibly transformed) geometry.

Three edge types
----------------
- **Temporal**: each point linked to the previous/next K points in time order,
  with edges omitted when the time gap exceeds a configurable threshold.
- **Spatial**: each point linked to all neighbours within radius R (metres)
  in (_x_m, _y_m) via BallTree.
- **Transect**: within each ``transect_id > 0``, a chain of edges connecting
  each point to its predecessor and successor in row order (no full clique).

All spatial/radius queries use ``sklearn.neighbors.BallTree`` for scalability.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from reta_ml import config as _cfg

# Label string → integer mapping (canonical order for the project)
LABEL_MAP: Dict[str, int] = {
    "Clean": 0,
    "Operational Error": 1,
    "Global Outlier": 2,
    "Local Outlier": 3,
}

# Alias forms found in real annotation files, matched case- and
# whitespace-insensitively by :func:`label_to_index`.  Critically, some files
# label local outliers simply as "Local" (and global as "Global") rather than
# the canonical "Local Outlier"/"Global Outlier"; without these aliases those
# rows were silently dropped (mapped to -1), erasing entire classes.
LABEL_ALIASES: Dict[str, int] = {
    "clean": 0,
    "operational error": 1,
    "operational": 1,
    "op error": 1,
    "op": 1,
    "global outlier": 2,
    "global": 2,
    "local outlier": 3,
    "local": 3,
}


def label_to_index(value: object) -> int:
    """Map a raw ``filtering_category`` value to a class index (or -1).

    Matching is robust to case and surrounding whitespace and accepts the
    short label forms ("Local", "Global") in addition to the canonical names
    ("Local Outlier", "Global Outlier").  Returns -1 for unknown/missing.
    """
    if value is None:
        return -1
    if value in LABEL_MAP:  # fast path: exact canonical match
        return LABEL_MAP[value]
    try:
        key = str(value).strip().casefold()
    except Exception:
        return -1
    if key in ("", "nan", "none", "na"):
        return -1
    return LABEL_ALIASES.get(key, -1)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    """Return a float array of time in seconds, using the first available
    time column from the config candidate list."""
    for col in _cfg.TIME_COLUMN_CANDIDATES:
        if col not in df.columns:
            continue
        ser = df[col]
        numeric = pd.to_numeric(ser, errors="coerce")
        if numeric.notna().all():
            return numeric.values.astype(float)
        dt = pd.to_datetime(ser, errors="coerce")
        if dt.notna().any():
            return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy()
    # Fallback: use row order as a monotonic proxy time.
    return np.arange(len(df), dtype=float)


def compute_time_gap_threshold(
    df: pd.DataFrame,
    *,
    n_sigma: float = 3.0,
    fallback: float = float("inf"),
) -> float:
    """Compute a per-field time-gap threshold as mean(dt) + n_sigma * std(dt).

    Falls back to *fallback* if time information is unavailable.
    """
    t = _time_seconds(df)
    if np.isnan(t).all() or len(t) < 2:
        return fallback
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt >= 0)]
    if len(dt) == 0:
        return fallback
    return float(np.mean(dt) + n_sigma * np.std(dt))


# ---------------------------------------------------------------------------
# Edge builders
# ---------------------------------------------------------------------------

def build_temporal_edges(
    df: pd.DataFrame,
    *,
    K: int = 3,
    time_gap_threshold: Optional[float] = None,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Build temporal edges: each node to its previous/next K in time order.

    An edge (i, j) is **omitted** if the absolute time difference between
    consecutive points on the path from i to j crosses the gap threshold
    (meaning there is an interruption, e.g. headland turn or data gap).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed field table sorted in temporal order.
    K : int
        Number of forward and backward temporal neighbours.
    time_gap_threshold : float, optional
        Maximum consecutive-point time delta (seconds).  If *None*, computed
        automatically as ``mean(dt) + n_sigma * std(dt)`` for this field.
    n_sigma : float
        Multiplier for automatic threshold (ignored when *time_gap_threshold*
        is given).

    Returns
    -------
    np.ndarray
        Shape ``(2, num_edges)`` COO edge index (int64).
    """
    n = len(df)
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    t = _time_seconds(df)

    if time_gap_threshold is None:
        time_gap_threshold = compute_time_gap_threshold(df, n_sigma=n_sigma)

    # Pre-compute consecutive deltas; mark large gaps
    consecutive_dt = np.empty(n, dtype=float)
    consecutive_dt[0] = 0.0
    consecutive_dt[1:] = np.abs(np.diff(t))
    gap_mask = consecutive_dt > time_gap_threshold  # True where gap is too big

    src_list = []
    dst_list = []

    for i in range(n):
        # Forward neighbours (i+1 … i+K)
        for offset in range(1, K + 1):
            j = i + offset
            if j >= n:
                break
            # Check that no consecutive gap on the path i→j exceeds threshold
            path_ok = True
            for step in range(i + 1, j + 1):
                if gap_mask[step]:
                    path_ok = False
                    break
            if path_ok:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

    if not src_list:
        return np.empty((2, 0), dtype=np.int64)

    return np.array([src_list, dst_list], dtype=np.int64)


def build_spatial_edges(
    df: pd.DataFrame,
    *,
    radius_m: float = 15.0,
    coord_cols: Tuple[str, str] = ("_x_m", "_y_m"),
    max_neighbors: Optional[int] = None,
) -> np.ndarray:
    """Build spatial edges via BallTree radius query.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed field table with metric coordinates.
    radius_m : float
        Neighbourhood radius in metres.
    coord_cols : tuple[str, str]
        Metric coordinate columns.

    Returns
    -------
    np.ndarray
        Shape ``(2, num_edges)`` COO edge index (int64), no self-loops.
    """
    cx, cy = coord_cols
    xy = df[[cx, cy]].values.astype(float)
    n = len(xy)
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    tree = BallTree(xy, metric="euclidean")

    src_list: list[int] = []
    dst_list: list[int] = []

    if max_neighbors is not None and int(max_neighbors) > 0:
        k = min(int(max_neighbors) + 1, n)
        dists, inds = tree.query(xy, k=k)
        for i in range(n):
            for dist, j in zip(dists[i].tolist(), inds[i].tolist()):
                if j == i:
                    continue
                if float(dist) <= float(radius_m):
                    src_list.append(i)
                    dst_list.append(int(j))
                    src_list.append(int(j))
                    dst_list.append(i)
    else:
        indices = tree.query_radius(xy, r=radius_m)
        for i, neighbors in enumerate(indices):
            for j in neighbors:
                if j != i:
                    src_list.append(i)
                    dst_list.append(int(j))

    if not src_list:
        return np.empty((2, 0), dtype=np.int64)

    return np.array([src_list, dst_list], dtype=np.int64)


def build_transect_edges(
    df: pd.DataFrame,
    *,
    K: int = 1,
) -> np.ndarray:
    """Build transect chain edges within each valid transect.

    For each ``transect_id > 0``, points are chained in row order: each node
    is connected to the previous K and next K within that transect.  This
    produces a chain (K=1) or short-range k-NN along the pass (K>1), **not**
    a full clique.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed field table with ``transect_id``.
    K : int
        Number of forward / backward neighbours within the transect chain.

    Returns
    -------
    np.ndarray
        Shape ``(2, num_edges)`` COO edge index (int64).
    """
    if "transect_id" not in df.columns:
        return np.empty((2, 0), dtype=np.int64)

    tids = df["transect_id"].values
    src_list = []
    dst_list = []

    for tid in np.unique(tids):
        if tid <= 0:
            continue
        idx = np.where(tids == tid)[0]
        m = len(idx)
        for local_i in range(m):
            for offset in range(1, K + 1):
                local_j = local_i + offset
                if local_j >= m:
                    break
                i_global = int(idx[local_i])
                j_global = int(idx[local_j])
                src_list.append(i_global)
                dst_list.append(j_global)
                src_list.append(j_global)
                dst_list.append(i_global)

    if not src_list:
        return np.empty((2, 0), dtype=np.int64)

    return np.array([src_list, dst_list], dtype=np.int64)


# ---------------------------------------------------------------------------
# Edge attributes
# ---------------------------------------------------------------------------

def build_edge_attr(
    df: pd.DataFrame,
    edge_index: np.ndarray,
    spatial_R: float = 15.0,
    *,
    coord_cols: Tuple[str, str] = ("_x_m", "_y_m"),
) -> np.ndarray:
    """Per-edge attributes ``[scaled_distance, scaled_|Δbearing|]``.

    Distance between endpoints is in metres, scaled by ``spatial_R``; the
    absolute bearing difference (degrees, wrapped to [0, 180]) is scaled by 180.
    Both land roughly in [0, 1] so they are comparable across edge types.
    Returns shape ``(E, 2)`` float32 (``(0, 2)`` when there are no edges).
    """
    if edge_index is None or edge_index.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    cx, cy = coord_cols
    n = len(df)
    x = (df[cx].to_numpy(dtype=float) if cx in df.columns else np.zeros(n))
    y = (df[cy].to_numpy(dtype=float) if cy in df.columns else np.zeros(n))
    b = (df["bearing_deg"].to_numpy(dtype=float) if "bearing_deg" in df.columns
         else np.zeros(n))

    s = edge_index[0].astype(int)
    d = edge_index[1].astype(int)
    dist = np.hypot(x[s] - x[d], y[s] - y[d])
    bdiff = np.abs(b[s] - b[d])
    bdiff = np.where(bdiff > 180.0, 360.0 - bdiff, bdiff)

    scale = float(spatial_R) if spatial_R and spatial_R > 0 else 1.0
    attr = np.stack([dist / scale, bdiff / 180.0], axis=1)
    return np.nan_to_num(attr, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature matrix and label vector
# ---------------------------------------------------------------------------

def feature_columns(main_variable: Optional[str] = None) -> list:
    """Return the canonical, ordered list of node-feature columns.

    This is ``config.ML_FEATURE_COLUMNS`` plus the main variable appended at
    the end.  It does **not** depend on which columns a given DataFrame
    happens to contain, so the feature dimension and column order are
    identical for every field (train vs validation vs inference).
    """
    if main_variable is None:
        main_variable = _cfg.MAIN_VARIABLE
    cols = list(_cfg.ML_FEATURE_COLUMNS)
    # Optionally append the raw main variable. Off by default because absolute
    # units do not generalize across sensors (see config.ML_FEATURE_COLUMNS).
    if (
        getattr(_cfg, "INCLUDE_MAIN_VARIABLE_FEATURE", False)
        and main_variable
        and main_variable not in cols
    ):
        cols.append(main_variable)
    return cols


def build_node_features(
    df: pd.DataFrame,
    main_variable: Optional[str] = None,
) -> np.ndarray:
    """Extract the ML feature matrix from the preprocessed DataFrame.

    Feature columns follow :func:`feature_columns` (``config.ML_FEATURE_COLUMNS``
    plus the main variable).  The matrix always has exactly those columns in a
    fixed order; columns missing from *df* are filled with 0.0.  This guarantees
    that train and validation feature matrices are aligned (same width, same
    column meaning) regardless of per-file schema differences, which keeps the
    model's input dimension stable across fields.

    NOTE: absolute coordinates (``_x_m``/``_y_m``) and the categorical
    ``transect_id`` are intentionally **not** model features — they do not
    generalize across fields and are only used for graph construction.  Missing
    values are filled with 0.0 (the post-normalization mean).

    Returns
    -------
    np.ndarray
        Shape ``(num_nodes, num_features)``, float32.
    """
    cols = feature_columns(main_variable)
    n = len(df)
    x = np.zeros((n, len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        if c in df.columns:
            col = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
            x[:, j] = np.nan_to_num(col, nan=0.0).astype(np.float32)
    return x


def build_labels(
    df: pd.DataFrame,
    label_col: str = "filtering_category",
) -> np.ndarray:
    """Map filtering_category strings to integer class indices.

    Mapping: 0=Clean, 1=Operational Error, 2=Global Outlier, 3=Local Outlier.
    Unknown or missing labels are mapped to -1.

    Returns
    -------
    np.ndarray
        Shape ``(num_nodes,)``, int64.
    """
    if label_col not in df.columns:
        return np.full(len(df), -1, dtype=np.int64)
    return np.array(
        [label_to_index(v) for v in df[label_col]], dtype=np.int64
    )


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_field_graph(
    df: pd.DataFrame,
    *,
    temporal_K: int = 3,
    spatial_R: float = 15.0,
    spatial_max_neighbors: Optional[int] = None,
    transect_K: int = 1,
    time_gap_threshold: Optional[float] = None,
    time_gap_n_sigma: float = 3.0,
    main_variable: Optional[str] = None,
) -> Dict[str, object]:
    """Build all three edge sets and the node feature matrix for one field.

    This is the primary entry point for graph construction.  It expects a
    preprocessed DataFrame (geometry, transects, ML features already computed;
    augmentation already applied for the train split).

    Parameters
    ----------
    df : pd.DataFrame
        Single-field preprocessed table.
    temporal_K : int
        Temporal neighbourhood size.
    spatial_R : float
        Spatial radius in metres.
    transect_K : int
        Transect chain neighbourhood size.
    time_gap_threshold : float, optional
        Explicit time gap threshold; auto-computed if *None*.
    time_gap_n_sigma : float
        Sigma multiplier for automatic threshold.
    main_variable : str, optional
        Name of the main variable column.

    Returns
    -------
    dict
        ``x``: node features (float32 ndarray, [N, F]),
        ``y``: integer labels (int64 ndarray, [N]),
        ``edge_index_temporal``: (2, E_t) int64,
        ``edge_index_spatial``: (2, E_s) int64,
        ``edge_index_transect``: (2, E_tr) int64,
        ``num_nodes``: int,
        ``time_gap_threshold``: float (the value used).
    """
    df = df.reset_index(drop=True)

    if time_gap_threshold is None:
        time_gap_threshold = compute_time_gap_threshold(
            df, n_sigma=time_gap_n_sigma
        )

    edge_temporal = build_temporal_edges(
        df, K=temporal_K, time_gap_threshold=time_gap_threshold
    )
    edge_spatial = build_spatial_edges(
        df, radius_m=spatial_R, max_neighbors=spatial_max_neighbors
    )
    edge_transect = build_transect_edges(df, K=transect_K)
    x = build_node_features(df, main_variable=main_variable)
    y = build_labels(df)

    # Edge attributes: [scaled distance, scaled |Δbearing|] per edge, so GAT
    # attention can be distance/orientation aware instead of treating all
    # neighbours equally. Distance is scaled by the spatial radius and bearing
    # difference by 180°, keeping both roughly in [0, 1].
    ea_temporal = build_edge_attr(df, edge_temporal, spatial_R)
    ea_spatial = build_edge_attr(df, edge_spatial, spatial_R)
    ea_transect = build_edge_attr(df, edge_transect, spatial_R)

    return {
        "x": x,
        "y": y,
        "edge_index_temporal": edge_temporal,
        "edge_index_spatial": edge_spatial,
        "edge_index_transect": edge_transect,
        "edge_attr_temporal": ea_temporal,
        "edge_attr_spatial": ea_spatial,
        "edge_attr_transect": ea_transect,
        "num_nodes": len(df),
        "time_gap_threshold": time_gap_threshold,
    }
