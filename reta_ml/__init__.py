"""
RETA ML — GNN-based outlier filtering for spatial sensor data.

Preprocessing (geometry, transects, ML feature table) is implemented in pure
Python/GeoPandas and does not require QGIS. See README for pipeline usage.
"""

__version__ = "0.3.0"

from reta_ml import config
from reta_ml.validate import validate_dataframe, ValidationMode, ValidationError
from reta_ml.preprocess import (
    compute_geometry_and_transects,
    build_ml_feature_table,
    preprocess_pipeline,
)
from reta_ml.load import (
    load_table,
    load_and_validate,
    print_filtering_category_distribution,
)
from reta_ml.split import kmeans_spatial_split, split_summary
from reta_ml.augment import augment_train, apply_single_transform, TRANSFORMS
from reta_ml.normalize import fit_scaler, transform_df, normalize_splits
from reta_ml.variogram import (
    dominant_direction,
    directional_variogram,
    fit_variogram,
    suggest_R,
)
from reta_ml.graph import (
    build_field_graph,
    build_temporal_edges,
    build_spatial_edges,
    build_transect_edges,
    build_node_features,
    build_labels,
    compute_time_gap_threshold,
    LABEL_MAP,
)
from reta_ml.dataset import (
    RETAFieldDataset,
    RETAHeteroFieldDataset,
    field_df_to_data,
    field_df_to_heterodata,
    data_to_heterodata,
)
from reta_ml.model import HeteroGAT, ThreeGNNEnsemble
from reta_ml.losses import FocalConfig, masked_hierarchical_focal_loss
from reta_ml.train import TrainConfig, train_and_validate, evaluate
from reta_ml.inference import (
    PreprocessState,
    load_preprocess_state,
    save_preprocess_state,
    predict_field_df,
    build_and_save_preprocess_state_from_train,
)

__all__ = [
    "__version__",
    "config",
    # validation
    "validate_dataframe",
    "ValidationMode",
    "ValidationError",
    # preprocessing (Sprint 1)
    "compute_geometry_and_transects",
    "build_ml_feature_table",
    "preprocess_pipeline",
    # data load
    "load_table",
    "load_and_validate",
    "print_filtering_category_distribution",
    # split (Sprint 2)
    "kmeans_spatial_split",
    "split_summary",
    # augmentation (Sprint 2)
    "augment_train",
    "apply_single_transform",
    "TRANSFORMS",
    # normalization (Sprint 2)
    "fit_scaler",
    "transform_df",
    "normalize_splits",
    # variogram (Sprint 2)
    "dominant_direction",
    "directional_variogram",
    "fit_variogram",
    "suggest_R",
    # graph construction (Sprint 3)
    "build_field_graph",
    "build_temporal_edges",
    "build_spatial_edges",
    "build_transect_edges",
    "build_node_features",
    "build_labels",
    "compute_time_gap_threshold",
    "LABEL_MAP",
    # dataset (Sprint 3/4)
    "field_df_to_data",
    "field_df_to_heterodata",
    "data_to_heterodata",
    "RETAFieldDataset",
    "RETAHeteroFieldDataset",
    # models (Sprint 4)
    "HeteroGAT",
    "ThreeGNNEnsemble",
    # losses (Sprint 4)
    "FocalConfig",
    "masked_hierarchical_focal_loss",
    # training (Sprint 4)
    "TrainConfig",
    "train_and_validate",
    "evaluate",
    # inference (Sprint 5)
    "PreprocessState",
    "load_preprocess_state",
    "save_preprocess_state",
    "predict_field_df",
    "build_and_save_preprocess_state_from_train",
]
