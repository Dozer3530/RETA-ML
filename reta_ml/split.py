"""
K-means spatial train/test split.

Splits the *training* file into train and test regions based on spatial
clusters of (lat, lon) or (_x_m, _y_m).  The validation file is never split.

Split is **step 1** in the Sprint 2 pipeline — no augmentation happens before
this.  The returned DataFrames carry a ``_split`` column ("train" / "test").
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def kmeans_spatial_split(
    df: pd.DataFrame,
    *,
    n_clusters: int = 2,
    test_clusters: Optional[List[int]] = None,
    coord_cols: Tuple[str, str] = ("_x_m", "_y_m"),
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a preprocessed DataFrame into train / test by spatial K-means.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed table with metric coordinates (``_x_m``, ``_y_m`` by
        default, or ``lat``, ``lon``).
    n_clusters : int
        Number of K-means clusters.  K=2 for compact fields, K=3–4 for
        irregular shapes.
    test_clusters : list[int], optional
        Cluster label(s) assigned to the **test** set.  If *None*, the
        smallest cluster becomes the test set.
    coord_cols : tuple[str, str]
        Column pair to cluster on.  Defaults to metric coords; fall back
        to ``("lat", "lon")`` when metric coords are absent.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    (train_df, test_df) : tuple[pd.DataFrame, pd.DataFrame]
        Each DataFrame is a copy of the relevant rows with an added
        ``_split`` column set to ``"train"`` or ``"test"``.
        Original index is preserved.
    """
    c0, c1 = coord_cols
    if c0 not in df.columns or c1 not in df.columns:
        if "lat" in df.columns and "lon" in df.columns:
            c0, c1 = "lat", "lon"
        else:
            raise ValueError(
                f"Coordinate columns {coord_cols} not found and no lat/lon fallback."
            )

    coords = df[[c0, c1]].values.astype(float)
    if np.isnan(coords).any():
        raise ValueError("Coordinate columns contain NaN; cannot cluster.")

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(coords)

    if test_clusters is None:
        cluster_sizes = np.bincount(labels, minlength=n_clusters)
        test_clusters = [int(np.argmin(cluster_sizes))]

    test_mask = np.isin(labels, test_clusters)

    train_df = df.loc[~test_mask].copy()
    test_df = df.loc[test_mask].copy()

    train_df["_split"] = "train"
    test_df["_split"] = "test"

    train_df["_cluster"] = labels[~test_mask]
    test_df["_cluster"] = labels[test_mask]

    return train_df, test_df


def split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, object]:
    """Return a quick summary dict of the split for logging / assertions."""
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_frac": len(train_df) / max(len(train_df) + len(test_df), 1),
        "test_frac": len(test_df) / max(len(train_df) + len(test_df), 1),
        "train_clusters": sorted(train_df["_cluster"].unique().tolist())
        if "_cluster" in train_df.columns
        else [],
        "test_clusters": sorted(test_df["_cluster"].unique().tolist())
        if "_cluster" in test_df.columns
        else [],
    }
