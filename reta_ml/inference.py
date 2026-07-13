"""
Sprint 5 inference pipeline.

New field input (no filtering_category) → validate → preprocess → normalize
(using train-fitted scaler) → build heterograph → load model (+temperature) →
predict → append:
  - filtering_category_pred
  - filtering_probability
  - filtering_entropy
  - filtering_review_flag

Outputs CSV/Parquet and prints warnings when outlier rate is high.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch

from reta_ml import config
from reta_ml.dataset import field_df_to_heterodata
from reta_ml.metrics import CLASS_NAMES, entropy_from_probs
from reta_ml.model import model_from_name
from reta_ml.normalize import fit_scaler, transform_df
from reta_ml.preprocess import preprocess_pipeline
from reta_ml.temperature_scaling import apply_temperature
from reta_ml.validate import ValidationMode, validate_dataframe


@dataclass(frozen=True)
class PreprocessState:
    main_variable: str
    temporal_K: int
    spatial_R: float
    spatial_max_neighbors: Optional[int]
    transect_K: int
    time_gap_threshold: Optional[float]
    time_gap_n_sigma: float
    norm_cols: List[str]
    scaler_mean: List[float]
    scaler_scale: List[float]
    temperature: float = 1.0
    # Ordered model-feature columns (provenance; build_node_features derives the
    # same list deterministically from main_variable, so this is informational).
    feature_cols: Optional[List[str]] = None
    # When True, normalize a new field by its OWN statistics at inference time
    # (the stored scaler_mean/scaler_scale are then only informational). This
    # matches training and is required for generalization across heterogeneous
    # fields/sensors.
    per_field_normalization: bool = False


def save_preprocess_state(path: Path, state: PreprocessState) -> None:
    path.write_text(json.dumps(state.__dict__, indent=2, sort_keys=True))


def load_preprocess_state(path: Path) -> PreprocessState:
    obj = json.loads(Path(path).read_text())
    return PreprocessState(
        main_variable=str(obj["main_variable"]),
        temporal_K=int(obj["temporal_K"]),
        spatial_R=float(obj["spatial_R"]),
        spatial_max_neighbors=obj.get("spatial_max_neighbors", None),
        transect_K=int(obj["transect_K"]),
        time_gap_threshold=obj.get("time_gap_threshold", None),
        time_gap_n_sigma=float(obj.get("time_gap_n_sigma", config.TIME_GAP_N_SIGMA)),
        norm_cols=[str(c) for c in obj["norm_cols"]],
        scaler_mean=[float(x) for x in obj["scaler_mean"]],
        scaler_scale=[float(x) for x in obj["scaler_scale"]],
        temperature=float(obj.get("temperature", 1.0)),
        feature_cols=(
            [str(c) for c in obj["feature_cols"]]
            if obj.get("feature_cols") is not None
            else None
        ),
        per_field_normalization=bool(obj.get("per_field_normalization", False)),
    )


def _scaler_from_state(state: PreprocessState):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    sc.mean_ = np.asarray(state.scaler_mean, dtype=float)
    sc.scale_ = np.asarray(state.scaler_scale, dtype=float)
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = len(state.norm_cols)
    sc.feature_names_in_ = np.asarray(state.norm_cols)
    return sc


def _spatial_tiles(proc_n: pd.DataFrame, max_nodes: int):
    """Partition rows into spatial tiles of <= max_nodes (KMeans on metric
    coords). Each tile's indices are returned sorted ascending so within-tile
    row order stays time-ordered (preprocessing sorts by time)."""
    n = len(proc_n)
    if n <= max_nodes or max_nodes <= 0:
        return [np.arange(n)]
    k = int(np.ceil(n / max_nodes))
    from sklearn.cluster import KMeans
    xy = proc_n[["_x_m", "_y_m"]].to_numpy(dtype=float)
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(xy)
    return [np.sort(np.where(labels == c)[0]) for c in range(k) if np.any(labels == c)]


@torch.no_grad()
def predict_field_df(
    df_raw: pd.DataFrame,
    *,
    run_dir: Path,
    model_name: str = "heterogat",
    device: Optional[str] = None,
    preprocess_state: Optional[PreprocessState] = None,
    entropy_percentile: float = config.EVAL_ENTROPY_PERCENTILE,
    outlier_warn_frac: float = config.OUTLIER_WARN_FRACTION,
    output_path: Optional[Path] = None,
    max_nodes_per_tile: int = 40000,
) -> pd.DataFrame:
    """
    Run end-to-end inference on a single field DataFrame.

    If `preprocess_state` is None, expects `run_dir/preprocess_state.json`.
    """
    run_dir = Path(run_dir)
    if preprocess_state is None:
        preprocess_state = load_preprocess_state(run_dir / "preprocess_state.json")

    validate_dataframe(df_raw, mode=ValidationMode.INFERENCE, main_variable=preprocess_state.main_variable)
    if preprocess_state.main_variable not in df_raw.columns:
        raise ValueError(
            f"Missing main variable column '{preprocess_state.main_variable}' in inference input."
        )
    proc = preprocess_pipeline(df_raw, main_variable=preprocess_state.main_variable)
    if len(proc) == 0:
        raise ValueError(
            "No georeferenced observations found in the input (all rows had "
            "missing or invalid lat/lon)."
        )

    # Normalize this field. With per-field normalization, fit the scaler on this
    # field's own statistics (the stored train scaler would reintroduce
    # cross-field shift); otherwise fall back to the stored train scaler.
    if getattr(preprocess_state, "per_field_normalization", False):
        scaler, _ = fit_scaler(proc, feature_cols=list(preprocess_state.norm_cols))
    else:
        scaler = _scaler_from_state(preprocess_state)
    proc_n = transform_df(proc, scaler, preprocess_state.norm_cols)

    # Build the model once, then run it per spatial tile so memory stays bounded
    # on large fields (GNN message passing materializes all edges at once).
    from reta_ml.graph import feature_columns
    gk = dict(
        temporal_K=preprocess_state.temporal_K,
        spatial_R=preprocess_state.spatial_R,
        spatial_max_neighbors=preprocess_state.spatial_max_neighbors,
        transect_K=preprocess_state.transect_K,
        time_gap_threshold=preprocess_state.time_gap_threshold,
        time_gap_n_sigma=preprocess_state.time_gap_n_sigma,
        main_variable=preprocess_state.main_variable,
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = len(feature_columns(preprocess_state.main_variable))
    model = model_from_name(model_name, in_dim=in_dim)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    model = model.to(device).eval()

    n_pts = len(proc_n)
    tiles = _spatial_tiles(proc_n, max_nodes_per_tile)
    full_logits = torch.zeros((n_pts, int(model.num_classes) if hasattr(model, "num_classes") else 4),
                              dtype=torch.float32)
    for idx in tiles:
        sub = proc_n.iloc[idx].reset_index(drop=True)
        hd = field_df_to_heterodata(sub, **gk).to(device)
        lg = model(hd).detach().to("cpu", dtype=torch.float32)
        full_logits[torch.as_tensor(idx, dtype=torch.long)] = lg
        del hd, lg

    logits = apply_temperature(full_logits, preprocess_state.temperature)
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    pred = probs.argmax(axis=1).astype(int)

    # Post-processing
    pred_names = [CLASS_NAMES[i] if 0 <= i < 4 else "Unknown" for i in pred.tolist()]
    pred_prob = probs.max(axis=1).astype(float)
    ent = entropy_from_probs(probs).astype(float)
    thr = float(np.nanpercentile(ent, entropy_percentile)) if len(ent) else float("nan")
    review = ent >= thr if np.isfinite(thr) else np.zeros_like(ent, dtype=bool)

    out = proc_n.copy()
    out["filtering_category_pred"] = pred_names
    out["filtering_probability"] = pred_prob
    out["filtering_entropy"] = ent
    out["filtering_review_flag"] = review.astype(bool)

    outlier_frac = float(np.mean(pred != 0)) if len(pred) else 0.0
    if outlier_frac > float(outlier_warn_frac):
        print(
            f"[WARN] Predicted outlier fraction is high: {outlier_frac*100:.1f}% "
            f"(threshold={outlier_warn_frac*100:.1f}%)."
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() in (".parquet", ".pq"):
            out.to_parquet(output_path, index=False)
        else:
            out.to_csv(output_path, index=False)
    return out


def build_and_save_preprocess_state_from_train(
    train_df_raw: pd.DataFrame,
    *,
    run_dir: Path,
    main_variable: str = config.MAIN_VARIABLE,
    temporal_K: int = config.TEMPORAL_K,
    spatial_R: float = config.SPATIAL_R,
    spatial_max_neighbors: Optional[int] = None,
    transect_K: int = config.TRANSECT_K,
    time_gap_threshold: Optional[float] = config.TIME_GAP_THRESHOLD,
    time_gap_n_sigma: float = config.TIME_GAP_N_SIGMA,
    temperature: float = 1.0,
) -> PreprocessState:
    """
    Convenience for Sprint 5: fit preprocessing + scaler on train raw data and
    save `preprocess_state.json` into an existing run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    validate_dataframe(train_df_raw, mode=ValidationMode.TRAINING, main_variable=main_variable)
    proc = preprocess_pipeline(train_df_raw, main_variable=main_variable)
    scaler, cols = fit_scaler(proc)

    state = PreprocessState(
        main_variable=main_variable,
        temporal_K=int(temporal_K),
        spatial_R=float(spatial_R),
        spatial_max_neighbors=spatial_max_neighbors,
        transect_K=int(transect_K),
        time_gap_threshold=time_gap_threshold,
        time_gap_n_sigma=float(time_gap_n_sigma),
        norm_cols=list(cols),
        scaler_mean=[float(x) for x in scaler.mean_.tolist()],
        scaler_scale=[float(x) for x in scaler.scale_.tolist()],
        temperature=float(temperature),
    )
    save_preprocess_state(run_dir / "preprocess_state.json", state)
    return state

