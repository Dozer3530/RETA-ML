"""
Geometric augmentation for the **train split only**.

Transforms are applied to metric coordinates (_x_m, _y_m) and bearing_deg.
Labels (filtering_category) are unchanged — the augmented copies retain the
same class as the original point.

Supported transforms
--------------------
- **horizontal_flip** : reflect across the Y-axis  → x' = -x, y' = y
- **vertical_flip**   : reflect across the X-axis  → x' = x,  y' = -y
- **rotate_180**      : half-turn                  → x' = -x, y' = -y
- **rotate_90**       : quarter-turn CW            → x' = y,  y' = -x
- **mirror**          : alias for horizontal_flip (reflect across Y-axis)

Bearing rules (all modular arithmetic, result in [0, 360))
----------------------------------------------------------
(a) horizontal_flip : bearing = (360 - bearing) % 360
(b) vertical_flip   : bearing = (180 - bearing) % 360
(c) rotate_180      : bearing = (bearing + 180) % 360
(d) rotate_90       : bearing = (bearing - 90)  % 360
(e) mirror           : same as horizontal_flip

Turn chirality
--------------
Horizontal mirror (and ``mirror``) inverts turn direction — if the data
contains a signed turn-direction feature (``bearing_diff`` with sign), it
must be negated.  The current Sprint-1 ``bearing_diff`` is unsigned (absolute
angular change), so chirality inversion applies only if a signed turn feature
is added later.  For safety we negate ``bearing_diff`` on horizontal-flip /
mirror transforms, flagging it as ``_augmented=True`` so downstream code can
handle this.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual transform functions
# ---------------------------------------------------------------------------

def _horizontal_flip(df: pd.DataFrame) -> pd.DataFrame:
    """Reflect across Y-axis: x → -x; bearing → (360 - bearing) % 360."""
    out = df.copy()
    out["_x_m"] = -out["_x_m"]
    out["bearing_deg"] = (360.0 - out["bearing_deg"]) % 360.0
    if "bearing_diff" in out.columns:
        out["bearing_diff"] = -out["bearing_diff"]
    out["_aug_type"] = "horizontal_flip"
    return out


def _vertical_flip(df: pd.DataFrame) -> pd.DataFrame:
    """Reflect across X-axis: y → -y; bearing → (180 - bearing) % 360."""
    out = df.copy()
    out["_y_m"] = -out["_y_m"]
    out["bearing_deg"] = (180.0 - out["bearing_deg"]) % 360.0
    out["_aug_type"] = "vertical_flip"
    return out


def _rotate_180(df: pd.DataFrame) -> pd.DataFrame:
    """Half-turn: x → -x, y → -y; bearing → (bearing + 180) % 360."""
    out = df.copy()
    out["_x_m"] = -out["_x_m"]
    out["_y_m"] = -out["_y_m"]
    out["bearing_deg"] = (out["bearing_deg"] + 180.0) % 360.0
    out["_aug_type"] = "rotate_180"
    return out


def _rotate_90(df: pd.DataFrame) -> pd.DataFrame:
    """Quarter-turn CW: (x,y) → (y,-x); bearing → (bearing - 90) % 360."""
    out = df.copy()
    x_orig = out["_x_m"].values.copy()
    out["_x_m"] = out["_y_m"]
    out["_y_m"] = -x_orig
    out["bearing_deg"] = (out["bearing_deg"] - 90.0) % 360.0
    out["_aug_type"] = "rotate_90"
    return out


def _mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Alias for horizontal_flip (reflect across Y-axis)."""
    out = _horizontal_flip(df)
    out["_aug_type"] = "mirror"
    return out


TRANSFORMS = {
    "horizontal_flip": _horizontal_flip,
    "vertical_flip": _vertical_flip,
    "rotate_180": _rotate_180,
    "rotate_90": _rotate_90,
    "mirror": _mirror,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def augment_train(
    train_df: pd.DataFrame,
    transforms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply geometric augmentations to the **train split only**.

    Returns a DataFrame containing the *original* train rows (``_aug_type =
    "original"``) concatenated with one copy per requested transform.  Labels
    (``filtering_category`` etc.) are preserved unchanged.

    Parameters
    ----------
    train_df : pd.DataFrame
        Train split from :func:`reta_ml.split.kmeans_spatial_split`.
        Must have ``_x_m``, ``_y_m``, ``bearing_deg``.
    transforms : list[str], optional
        Subset of ``TRANSFORMS`` keys.  Default: all five transforms.

    Returns
    -------
    pd.DataFrame
        Concatenated original + augmented rows.  New columns:
        ``_aug_type`` (str), ``_augmented`` (bool).
    """
    required = {"_x_m", "_y_m", "bearing_deg"}
    missing = required - set(train_df.columns)
    if missing:
        raise ValueError(f"train_df missing required columns: {missing}")

    if transforms is None:
        transforms = list(TRANSFORMS.keys())

    unknown = set(transforms) - set(TRANSFORMS.keys())
    if unknown:
        raise ValueError(f"Unknown transforms: {unknown}")

    original = train_df.copy()
    original["_aug_type"] = "original"
    original["_augmented"] = False

    parts = [original]
    for name in transforms:
        aug = TRANSFORMS[name](train_df)
        aug["_augmented"] = True
        parts.append(aug)

    return pd.concat(parts, ignore_index=True)


def apply_single_transform(
    df: pd.DataFrame,
    transform: str,
) -> pd.DataFrame:
    """
    Apply a single named transform and return the transformed copy.

    Useful for unit-testing individual bearing rules.
    """
    if transform not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{transform}'. Choose from: {list(TRANSFORMS.keys())}"
        )
    return TRANSFORMS[transform](df)
