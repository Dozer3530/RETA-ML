"""
Central configuration: paths, main variable, feature list, and sanity thresholds.

Used by validation, preprocessing, and data load. See DESIGN_SPEC_ALIGNMENT.md
and IMPLEMENTATION_SCHEDULE.md for the intended pipeline.
"""

from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Paths and data layout
# ---------------------------------------------------------------------------

# Paths are expressed relative to the repo root so that scripts can be run
# from anywhere (e.g. tests, notebooks, CLI).
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

# Default file names in data/ (override via env or caller)
TRAIN_TEST_FILE = "train_test.gpkg"  # or first GPKG if not present
VALIDATION_FILE = "validation.gpkg"

# ---------------------------------------------------------------------------
# Main variable and schema
# ---------------------------------------------------------------------------

# Name of the agronomic variable to model (e.g. "yield", "ECa").
MAIN_VARIABLE = "yield"

# Note: time column name is flexible; validation checks that at least one of
# TIME_COLUMN_CANDIDATES is present. Do not require literal "time" here.
REQUIRED_COLUMNS_TRAINING = ["lat", "lon", "filtering_category"]
REQUIRED_COLUMNS_INFERENCE = ["lat", "lon"]
TIME_COLUMN_CANDIDATES = [
    "time",
    "seconds",
    "duration",
    "gps_time",
    "timestamp",
    "datetime",
    "DateTime",
    "LocalDate",
    "Local_Date",
    "Time",
    "gps_sec",
    "GPS_Time",
    "gpstime",
    "utc_time",
    "epoch",
    "posix_time",
    "time_s",
    "time_sec",
]

# When no per-point timestamp is available (missing column or only calendar
# dates), motion features (time_dt, derived speed/accel) assume uniform sampling
# at this interval in native row / equipment logging order.
ASSUMED_SAMPLE_INTERVAL_S: float = 1.0

GEOMETRIC_COLUMNS = ["lat", "lon", "_x_m", "_y_m"]

# ---------------------------------------------------------------------------
# ML feature table (Path B reimplementation)
# ---------------------------------------------------------------------------

# Columns produced by the preprocessing pipeline and used as node features.
#
# DESIGN (spatially-aware model): rather than hand-crafting the anomaly signal
# with z-scores, we feed the chosen variable directly (normalized) and let the
# GNN learn the spatial comparison itself via message passing. The value enters
# as two columns:
#   - value_norm       : the variable centered per field (median removed) and
#                        divided by a SINGLE global scale fit on the training
#                        set. Centering is per-field, so the model is robust to
#                        a field sitting at a different absolute level; the scale
#                        is global, so a noisy field keeps a wider spread than a
#                        tight one (variability survives as information).
#   - field_rel_spread : the field's own IQR / the global scale, i.e. how
#                        variable this field is vs a typical field. The explicit
#                        "knowing that some data is more variable" signal, so the
#                        model can flag global outliers in a variability-aware way.
# Local outliers are detected by the graph (a point compared to its spatial
# neighbours through the edges + distance/bearing edge attributes), not by a
# precomputed local z-score.
#
# NOTE: motion features speed_mps/accel_mps2 capture operational (maneuver)
# artefacts. Absolute coordinates (_x_m,_y_m) and transect_id are NOT features
# (they don't generalize across fields) but ARE used for graph construction and
# the edge attributes. The legacy z-score features (global_z, z_score,
# ratio_to_transect_mean) are no longer model inputs; they are still computed
# and kept in the table for inspection / the RF baseline.
ML_FEATURE_COLUMNS: List[str] = [
    "dist_m",
    "bearing_deg",
    "bearing_diff",
    "time_dt",
    "speed_mps",
    "accel_mps2",
    "is_turn",
    "is_short",
    "value_norm",
    "field_rel_spread",
]

# The raw main variable is not appended directly: it enters (normalized) as
# value_norm above. Absolute units do not transfer across fields, but value_norm
# (per-field centered, globally scaled) does.
INCLUDE_MAIN_VARIABLE_FEATURE = False

# ---------------------------------------------------------------------------
# Named feature sets — single source of truth (used by the tabular benchmarks)
# ---------------------------------------------------------------------------
#
# Two distinct, deliberately different feature sets. Tree ensembles (RF/XGBoost)
# have no internal mechanism to capture spatial structure, so they are given the
# hand-crafted spatial statistics; the GNN learns spatial structure from the
# graph and is given the value-based features instead.
#
# IMPORTANT (leakage): the spatial statistics in RF_XGB_FEATURES (global_z,
# z_score_*w, ratio_to_transect_mean, local_mean_*w) are per-field quantities
# computed in preprocessing. When used in a split/benchmark they must be computed
# from each region's OWN points only — never fit on test data. Swath width is
# likewise inferred per region/field independently. See reports/lofo/lofo_audit_v2.md.

# Stage 1 (two-stage redesign): motion / kinematic features only — no
# is_turn/is_short and no value-based spatial statistics. Magnitude features
# use field-relative columns so cross-sensor LOFO is not dominated by equipment
# operating speed (EC survey vs combine harvest).
STAGE1_FEATURES: List[str] = [
    "bearing_deg",
    "bearing_diff",
    "time_dt",
    "speed_rel",
    "dist_rel",
    "accel_rel",
]

# Stage 2 with fixed 15/30/45 m local stats (cross-field comparable bands).
STAGE2_FEATURES_FIXED_BANDS: List[str] = [
    "global_z",
    "z_score_15m",
    "z_score_30m",
    "z_score_45m",
    "local_mean_15m",
    "local_mean_30m",
    "local_mean_45m",
    "local_std_15m",
    "local_std_30m",
    "local_std_45m",
    "ratio_to_transect_mean",
    "value",
]

# Stage 2 (two-stage redesign): value-based spatial statistics at swath-scaled
# multi-band radii (2×, 3×, 4× inferred swath width).
STAGE2_FEATURES: List[str] = [
    "global_z",
    "z_score_2w",
    "z_score_3w",
    "z_score_4w",
    "local_mean_2w",
    "local_mean_3w",
    "local_mean_4w",
    "local_std_2w",
    "local_std_3w",
    "local_std_4w",
    "ratio_to_transect_mean",
    "value",
]

# Legacy single-radius features (15 m BallTree) — ablation Run 1 baseline.
# Motion uses field-relative columns (v5 normalization); fixed-radius local stats.
RF_XGB_FEATURES_LEGACY: List[str] = [
    "bearing_deg",
    "bearing_diff",
    "time_dt",
    "is_turn",
    "is_short",
    "speed_rel",
    "dist_rel",
    "accel_rel",
    "global_z",
    "z_score",
    "ratio_to_transect_mean",
    "local_mean",
    "local_std",
    "value",
]

# Stage 2 with fixed 15 m local stats (two-stage control, Run 4).
STAGE2_FEATURES_LEGACY: List[str] = [
    "global_z",
    "z_score",
    "local_mean",
    "local_std",
    "ratio_to_transect_mean",
    "value",
]

# Single-stage fixed 15/30/45 m multi-band (cross-field comparable).
RF_XGB_FEATURES_FIXED_BANDS: List[str] = [
    "bearing_deg",
    "bearing_diff",
    "time_dt",
    "is_turn",
    "is_short",
    "speed_rel",
    "dist_rel",
    "accel_rel",
    "global_z",
    "z_score_15m",
    "z_score_30m",
    "z_score_45m",
    "ratio_to_transect_mean",
    "local_mean_15m",
    "local_mean_30m",
    "local_mean_45m",
    "local_std_15m",
    "local_std_30m",
    "local_std_45m",
    "value",
]

# Fixed-band variant without local_std columns (matches v5_normalized feature shape).
RF_XGB_FEATURES_FIXED_BANDS_NO_STD: List[str] = [
    c for c in RF_XGB_FEATURES_FIXED_BANDS
    if not c.startswith("local_std_")
]

# Paper candidate: no raw value, no local_mean (zero permutation importance).
RF_XGB_FEATURES_FIXED_BANDS_NO_STD_NO_VALUE: List[str] = [
    c for c in RF_XGB_FEATURES_FIXED_BANDS_NO_STD
    if c != "value" and not c.startswith("local_mean_")
]

# Random Forest / XGBoost: keep the hand-crafted spatial statistics. These give
# the tree models their only window into spatial structure, so they stay.
RF_XGB_FEATURES: List[str] = [
    "bearing_deg",
    "bearing_diff",
    "time_dt",
    "is_turn",
    "is_short",
    "speed_rel",
    "dist_rel",
    "accel_rel",
    "global_z",
    "z_score_2w",
    "z_score_3w",
    "z_score_4w",
    "ratio_to_transect_mean",
    "local_mean_2w",
    "local_mean_3w",
    "local_mean_4w",
    "local_std_2w",
    "local_std_3w",
    "local_std_4w",
    "value",
]

# Graph model (defined here for future use; NOT used in the RF/XGB LOFO run).
# Excludes the hand-crafted spatial statistics — the graph handles spatial
# structure internally — and uses value_norm / field_rel_spread instead.
GNN_FEATURES: List[str] = list(ML_FEATURE_COLUMNS)

# ---------------------------------------------------------------------------
# Validation sanity thresholds (data-driven / guard-rails)
# ---------------------------------------------------------------------------

# Plausible range for the main variable. Kept wide so the pipeline works for
# arbitrary geospatial variables (yield, ECa, elevation, temperature, ...),
# some of which are legitimately negative. Tighten per-dataset if desired.
MAIN_VARIABLE_MIN = -1e9
MAIN_VARIABLE_MAX = 1e9
TIME_MONOTONIC_TOLERANCE = 0.0  # strict monotonic within pass (or use small epsilon)

# ---------------------------------------------------------------------------
# Preprocessing and local statistics (mostly data-driven)
# ---------------------------------------------------------------------------

# Transect detection
TURN_THRESHOLD_DEG = 45.0
MIN_TRANSECT_LEN_M = 30.0
DIST_BREAK_M = 20.0  # segment break if gap > this
BEARING_SMOOTH_WINDOW = 3

# Swath width inference (field-level; used to scale local-stat radii).
DEFAULT_SWATH_WIDTH_M: float = 10.0

# Local statistics (KD-tree / BallTree). Radii are swath_width × {2, 3, 4};
# LOCAL_RADIUS_M is retained as a legacy fallback reference only.
LOCAL_RADIUS_M = 15.0
LOCAL_MIN_NEIGHBORS = 3  # minimum points in radius for valid local stats

# Fixed absolute radii (metres) for cross-field-comparable multi-band local stats.
LOCAL_FIXED_BANDS_M: Tuple[float, float, float] = (15.0, 30.0, 45.0)
LOCAL_FIXED_BAND_SUFFIXES: Tuple[str, str, str] = ("15m", "30m", "45m")

# ---------------------------------------------------------------------------
# K-means spatial split (Sprint 2)
# ---------------------------------------------------------------------------

# K-means on (lat, lon) or (_x_m, _y_m) to define train vs test regions.
SPLIT_N_CLUSTERS = 2  # K=2 for compact fields; K=3–4 for irregular
SPLIT_RANDOM_STATE = 42

# K-means clusters per TRAINING field in the cross-sensor LOFO benchmark. Each
# training field is partitioned into this many spatially-contiguous clusters
# (on coordinates only); the smallest cluster is held out as the spatial
# validation region and the rest are used for fitting. k=4 → 3 fit clusters + 1
# validation cluster per training field. Kept separate from SPLIT_N_CLUSTERS so
# the LOFO setting does not perturb the GNN/app default.
LOFO_N_CLUSTERS = 4

# Optional explicit mapping of clusters to train/test. When None, the smallest
# cluster becomes the test region (default Sprint 2 behaviour).
# Example:
#   SPLIT_TEST_CLUSTERS = [1]          # cluster 1 → test
#   SPLIT_TEST_CLUSTERS = [0, 2]       # clusters 0 and 2 → test
SPLIT_TEST_CLUSTERS: list[int] | None = None

# ---------------------------------------------------------------------------
# Augmentation (Sprint 2)
# ---------------------------------------------------------------------------

# Normalization regime. With the value-based feature set, level-robustness is
# achieved by per-field MEDIAN centering inside preprocessing (value_norm), while
# the SCALE is a single global value fit on the training set and saved — so a
# field's variability is preserved relative to a typical field (field_rel_spread)
# instead of being normalized away. The geometry/motion features are scaled by a
# saved global StandardScaler (their units are stable across fields). Hence we do
# NOT refit a per-field scaler at inference: the saved training scaler is applied,
# and the only per-field step (median centering) already happened in preprocess.
PER_FIELD_NORMALIZATION = False

# Toggle whether geometric augmentation is applied to the train split.
AUGMENT_TRAIN = True

# Transforms passed to reta_ml.augment.augment_train when augmentation is
# enabled.
DEFAULT_AUGMENTATIONS = [
    "horizontal_flip",
    "vertical_flip",
    "rotate_180",
    "rotate_90",
    "mirror",
]

# ---------------------------------------------------------------------------
# Graph construction (Sprint 3) – data-driven defaults
# ---------------------------------------------------------------------------

# Temporal neighbourhood: forward/backward neighbours in time order. Can be
# set from variogram-derived range / spacing if desired.
TEMPORAL_K = 3

# Spatial radius (metres) for spatial edges. In the full workflow this should
# be set from the directional variogram via reta_ml.variogram.suggest_R.
SPATIAL_R = 15.0

# Cap neighbours per node to avoid huge graphs (None = no cap).
SPATIAL_MAX_NEIGHBORS: int | None = 32

# Transect chain neighbours within each transect (1 = simple chain).
TRANSECT_K = 1

# Time-gap handling: edges are omitted when dt exceeds this threshold.
TIME_GAP_N_SIGMA = 3.0  # auto threshold = mean(dt) + N_SIGMA * std(dt)
TIME_GAP_THRESHOLD: float | None = None  # explicit override; None = auto

# ---------------------------------------------------------------------------
# Variogram (Sprint 2) – used to set SPATIAL_R and optionally TEMPORAL_K
# ---------------------------------------------------------------------------

VARIOGRAM_N_LAGS = 20
VARIOGRAM_TOLERANCE_DEG = 22.5
VARIOGRAM_R_FACTOR = (0.5, 1.0)  # multiply fitted range by these for suggested R

# ---------------------------------------------------------------------------
# GNN model + training hyper-parameters (empirical / tuned)
# ---------------------------------------------------------------------------

# HeteroGAT architecture (reta_ml.model.ModelConfig).
GNN_HIDDEN_DIM = 96
GNN_NUM_LAYERS = 2
GNN_HEADS = 4
GNN_DROPOUT = 0.2

# Training loop defaults (reta_ml.train.TrainConfig). These are starting
# points and are typically refined via Optuna sweeps.
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 1
TRAIN_LR = 3e-4
TRAIN_WEIGHT_DECAY = 1e-4

# Focal / hierarchical loss defaults. gamma and class-wise alpha are
# *empirical* and are the main targets of the Optuna sweep. When
# FOCAL_ALPHA_4CLASS is None, per-task alphas are derived automatically from
# training label frequencies.
FOCAL_GAMMA = 2.0
FOCAL_AUTO_ALPHA = True
FOCAL_ALPHA_4CLASS = None  # Optional 4-tuple (Clean, Op, Global, Local)

# ---------------------------------------------------------------------------
# Evaluation / uncertainty thresholds (Sprint 5–6)
# ---------------------------------------------------------------------------

# Entropy percentile for high-uncertainty review flags.
EVAL_ENTROPY_PERCENTILE = 90.0

# Sanity check on outlier rate: warn if predicted outlier fraction exceeds
# this fraction of points in a field.
OUTLIER_WARN_FRACTION = 0.40

# ---------------------------------------------------------------------------
# Missing value policy (documented, enforced in code)
# ---------------------------------------------------------------------------

# Missing time → distance fallback for gap detection; flag segment.
# Missing main variable → node stays in graph; mask in loss later.
# Missing geometric (lat/lon/_x_m/_y_m) → exclude point.
