"""
RETA_ML — local GUI app (Streamlit).

Run it on your own machine (uses your GPU automatically):
    ./run.sh              # one-click launcher (sets up env + starts the app)
or:
    streamlit run app.py

Tabs:
  • Annotate  — pick a field + a trained model, predict, and see the map.
  • Train     — train a model on one or more fields with live progress.
  • Benchmark — leave-one-field-out comparison of all models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
# Use all CPU cores (and the GPU below if present).
try:
    torch.set_num_threads(os.cpu_count() or 1)
except Exception:
    pass

from reta_ml import config
from reta_ml.load import load_table
from reta_ml.preprocess import preprocess_pipeline
from reta_ml.split import kmeans_spatial_split
from reta_ml.augment import augment_train
from reta_ml.normalize import fit_scaler, transform_df
from reta_ml.dataset import RETAHeteroFieldDataset, field_df_to_heterodata
from reta_ml.graph import build_labels, build_node_features, feature_columns, label_to_index
from reta_ml.model import ModelConfig, model_from_name
from reta_ml.metrics import confusion_matrix, prf_from_cm, labels_to_indices
from reta_ml.inference import PreprocessState, save_preprocess_state, predict_field_df

CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"  # overridden by the sidebar selector
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
CLASS = {0: ("Clean", "#9aa7b1"), 1: ("Operational Error", "#ff8c00"),
         2: ("Global Outlier", "#7b2fbf"), 3: ("Local Outlier", "#d62728")}
GK = dict(temporal_K=config.TEMPORAL_K, spatial_R=config.SPATIAL_R,
          spatial_max_neighbors=config.SPATIAL_MAX_NEIGHBORS, transect_K=config.TRANSECT_K,
          time_gap_threshold=config.TIME_GAP_THRESHOLD, time_gap_n_sigma=config.TIME_GAP_N_SIGMA)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def list_fields() -> list:
    return sorted([str(p) for p in DATA_DIR.glob("*.gpkg")] +
                  [str(p) for p in DATA_DIR.glob("*.csv")] +
                  [str(p) for p in DATA_DIR.glob("*.parquet")])


@st.cache_data(show_spinner=False)
def numeric_columns(path: str) -> list:
    df = load_table(Path(path), validate=False)
    cols = []
    for c in df.columns:
        if c in ("lat", "lon", "geometry"):
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            cols.append(c)
    return cols


# Preferred sensor value columns, in priority order (used to pre-select a
# sensible default per file in the Train tab).
_VALUE_PRIORITY = ["Yld_Vol_Dr", "Yld_Mass_W", "Yld_Mass_D", "EC_Shallow",
                   "EC SH", "ECa", "yield", "value"]


def guess_value_index(cols: list) -> int:
    for name in _VALUE_PRIORITY:
        if name in cols:
            return cols.index(name)
    return 0


def list_runs() -> list:
    """Runs usable for inference with the CURRENT code: they must have a
    checkpoint AND a preprocess_state.json whose feature schema matches the
    current feature set (this excludes stale pre-fix runs)."""
    import json
    if not RUNS_DIR.exists():
        return []
    want = len(feature_columns("value"))
    out = []
    for p in sorted(RUNS_DIR.glob("*"), reverse=True):
        if not ((p / "best_model.pt").exists() and (p / "preprocess_state.json").exists()):
            continue
        try:
            fc = json.loads((p / "preprocess_state.json").read_text()).get("feature_cols")
        except Exception:
            fc = None
        if fc is not None and len(fc) == want:
            out.append(str(p))
    return out


def make_map(df: pd.DataFrame, title: str):
    import plotly.graph_objects as go
    tn = {**{k: v[0] for k, v in CLASS.items()}, -1: "(unlabeled)"}
    hover = np.array([f"pred: {tn.get(int(p),'?')}<br>true: {tn.get(int(t),'?')}<br>prob: {pr:.2f}<br>entropy: {e:.2f}"
                      for p, t, pr, e in zip(df["pred"], df["true"], df["probability"], df["entropy"])])
    has_true = (df["true"] != -1).any()
    traces, g_pred, g_true, g_cw, g_rev = [], [], [], [], []

    def add(mask, name, color):
        d = df[mask]
        traces.append(go.Scattermap(lat=d["lat"], lon=d["lon"], mode="markers",
                      marker=dict(size=4, color=color), name=f"{name} ({int(mask.sum())})",
                      text=hover[mask.values], hoverinfo="text", visible=False))
        return len(traces) - 1

    for c, (lab, col) in CLASS.items():
        m = df["pred"] == c
        if m.any():
            g_pred.append(add(m, f"pred {lab}", col))
    if has_true:
        for c, (lab, col) in CLASS.items():
            m = df["true"] == c
            if m.any():
                g_true.append(add(m, f"true {lab}", col))
        v = df["true"] != -1
        g_cw.append(add(v & (df["true"] == df["pred"]), "correct", "#2ca02c"))
        g_cw.append(add(v & (df["true"] != df["pred"]), "misclassified", "#d62728"))
    fl = df["review_flag"].astype(bool)
    g_rev.append(add(~fl, "confident", "#cfd8dc"))
    g_rev.append(add(fl, "flagged for review", "#1f77b4"))

    fig = go.Figure(data=traces)
    for i in g_pred:
        fig.data[i].visible = True

    def vis(idxs):
        v = [False] * len(traces)
        for i in idxs:
            v[i] = True
        return v
    btns = [dict(label="Predicted", method="update", args=[{"visible": vis(g_pred)}])]
    if g_true:
        btns += [dict(label="True labels", method="update", args=[{"visible": vis(g_true)}]),
                 dict(label="Correct / Wrong", method="update", args=[{"visible": vis(g_cw)}])]
    btns.append(dict(label="Review flags", method="update", args=[{"visible": vis(g_rev)}]))
    fig.update_layout(title=title, height=650, margin=dict(l=0, r=0, t=70, b=0),
                      map=dict(style="open-street-map",
                               center=dict(lat=float(df["lat"].mean()), lon=float(df["lon"].mean())), zoom=14),
                      updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center",
                                        y=1.08, yanchor="bottom", buttons=btns, showactive=True)])
    return fig


def export_bundle(field_path, value_col, run_dir):
    """Build a full evaluation bundle (.zip) via scripts/export_run.py."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.export_run import export_run
    except ImportError as e:
        # The detailed evaluation-bundle exporter (figures/report generation) is not
        # bundled in the public repository. Train/inspect models with the code in reta_ml/.
        raise RuntimeError(
            "The full evaluation-bundle export is not available in this public build."
        ) from e
    return export_run(Path(field_path), value_col, Path(run_dir), REPO_ROOT / "reports" / "exports")


def train_and_save(specs, model_key, epochs, augment, progress_cb, device=None):
    device = device or DEVICE
    from reta_ml.train import TrainConfig, train_and_validate
    from torch_geometric.loader import DataLoader
    aug_raw, val_raw, tr_raw = [], [], []
    transforms = ["horizontal_flip", "rotate_180"] if augment else []
    for path, col in specs:
        raw = load_table(Path(path), validate=False).copy()
        raw["value"] = pd.to_numeric(raw[col], errors="coerce")
        proc = preprocess_pipeline(raw, main_variable="value")
        tr, val = kmeans_spatial_split(proc, n_clusters=config.SPLIT_N_CLUSTERS,
                                       test_clusters=config.SPLIT_TEST_CLUSTERS,
                                       random_state=config.SPLIT_RANDOM_STATE)
        aug = augment_train(tr, transforms=transforms) if transforms else tr
        # Center value_norm/field_rel_spread on the TRAIN region's own stats so
        # the validation region does not leak into the train features' centering.
        from reta_ml.normalize import fit_value_centering, apply_value_centering
        _med, _iqr = fit_value_centering(tr)
        tr = apply_value_centering(tr, _med, _iqr)
        aug = apply_value_centering(aug, _med, _iqr)
        val = apply_value_centering(val, _med, _iqr)
        tr_raw.append(tr); aug_raw.append(aug); val_raw.append(val)
    # Fit ONE global scaler on the concatenated training regions (per-field
    # median-centering of the value already happened in preprocess; this fixes
    # the global value scale + the geometry scaling), then apply it everywhere.
    scaler, cols = fit_scaler(pd.concat(tr_raw, ignore_index=True))
    aug_dfs = [transform_df(a, scaler, cols) for a in aug_raw]
    val_dfs = [transform_df(v, scaler, cols) for v in val_raw]
    last_scaler, last_cols = scaler, cols
    tl = DataLoader(RETAHeteroFieldDataset(aug_dfs, main_variable="value", **GK), batch_size=1)
    vl = DataLoader(RETAHeteroFieldDataset(val_dfs, main_variable="value", **GK), batch_size=1)
    in_dim = int(next(iter(tl))["point"].x.shape[1])
    mcfg = ModelConfig(hidden_dim=config.GNN_HIDDEN_DIM, num_layers=config.GNN_NUM_LAYERS,
                       heads=config.GNN_HEADS, dropout=config.GNN_DROPOUT, conv_aggr="sum", residual=True)
    model = model_from_name(model_key, in_dim=in_dim, cfg=mcfg)
    tcfg = TrainConfig(model_name=model_key, epochs=epochs, lr=config.TRAIN_LR,
                       weight_decay=config.TRAIN_WEIGHT_DECAY, device=device, runs_dir=str(RUNS_DIR))
    run_dir = train_and_validate(model, tl, vl, cfg=tcfg, progress_callback=progress_cb)
    save_preprocess_state(run_dir / "preprocess_state.json", PreprocessState(
        main_variable="value", temporal_K=config.TEMPORAL_K, spatial_R=float(config.SPATIAL_R),
        spatial_max_neighbors=config.SPATIAL_MAX_NEIGHBORS, transect_K=config.TRANSECT_K,
        time_gap_threshold=config.TIME_GAP_THRESHOLD, time_gap_n_sigma=float(config.TIME_GAP_N_SIGMA),
        norm_cols=list(last_cols), scaler_mean=[float(x) for x in last_scaler.mean_],
        scaler_scale=[float(x) for x in last_scaler.scale_], temperature=1.0,
        feature_cols=list(feature_columns("value")), per_field_normalization=False))
    return run_dir


# Within-field feature set for the tabular Random Forest (richer than the GNN's
# cross-field set: includes local mean/std and the raw value).
RF_FEATURES = ["dist_m", "bearing_deg", "bearing_diff", "time_dt", "is_turn", "is_short",
               "speed_mps", "accel_mps2", "global_z", "z_score", "ratio_to_transect_mean",
               "local_mean", "local_std", "value"]


def train_random_forest(field_path, value_col, test_frac: float = 0.25, seed: int = 42):
    """Train a Random Forest on ONE field with a within-field stratified split.

    This is the tabular model used for the paper's Table 1 / confusion matrix:
    a within-field stratified train/test split (so every class appears in both)
    and a class-balanced RandomForest(n_estimators=400). Returns
    (confusion_matrix, prf_dict, accuracy, predictions_df). Fixed seed =>
    identical numbers on re-run.

    Unlike the GNN, this single-field tabular model uses the RICHER feature set
    that also includes the local mean/std and the raw value: cross-field
    generalization is not a concern within one field, and those magnitude
    features materially help operational- and local-anomaly detection.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    raw = load_table(Path(field_path), validate=False).copy()
    raw["value"] = pd.to_numeric(raw[value_col], errors="coerce")
    proc = preprocess_pipeline(raw, main_variable="value")
    feats = [c for c in RF_FEATURES if c in proc.columns]
    X = proc[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    y = build_labels(proc)
    keep = y != -1
    X, y = X[keep], y[keep]
    sub = proc.loc[keep].reset_index(drop=True)
    if len(y) < 20 or len(np.unique(y)) < 2:
        raise ValueError("Need at least two annotated classes and ~20+ labeled points to train.")

    counts = np.bincount(y, minlength=4)
    strat = y if counts[counts > 0].min() >= 2 else None  # stratify only if feasible
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strat)

    clf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                 class_weight="balanced", random_state=seed)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te]).astype(int)

    cm = confusion_matrix(y[te], pred, num_classes=4)
    prf = prf_from_cm(cm)
    acc = float((pred == y[te]).mean()) if len(te) else 0.0
    pred_df = pd.DataFrame({
        "lat": sub.loc[te, "lat"].to_numpy(),
        "lon": sub.loc[te, "lon"].to_numpy(),
        "true": y[te], "pred": pred,
    })
    return cm, prf, acc, pred_df


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="RETA_ML — geospatial outlier filter", layout="wide")
st.title("RETA_ML — geospatial outlier filter")
st.sidebar.header("Compute")
_gpu_name, _gpu_gb = None, 0.0
if CUDA_AVAILABLE:
    try:
        _gpu_name = torch.cuda.get_device_name(0)
        _gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        pass
# Tiny GPUs (e.g. a 2 GB Quadro P620) reliably OOM / arch-mismatch on these
# graphs, so "Auto" quietly prefers the CPU when VRAM is under ~4 GB.
_small_gpu = bool(CUDA_AVAILABLE and 0 < _gpu_gb < 4.0)
_choice = st.sidebar.radio(
    "Device", ["Auto", "GPU", "CPU"], index=0,
    help="Auto uses the GPU only if it's big enough (>=4 GB); otherwise CPU. "
         "Pick CPU to force it, or GPU to override (may error on old/small cards).",
)
if _choice == "CPU" or not CUDA_AVAILABLE:
    DEVICE = "cpu"
elif _choice == "Auto" and _small_gpu:
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
st.sidebar.metric("Active device", "GPU (CUDA)" if DEVICE == "cuda" else "CPU")
if _gpu_name:
    cap = f"GPU: {_gpu_name} ({_gpu_gb:.0f} GB)"
    if DEVICE != "cuda":
        cap += " — auto-skipped (<4 GB)" if (_small_gpu and _choice == "Auto") else " — not in use"
    st.sidebar.caption(cap)
else:
    st.sidebar.caption("No CUDA GPU detected")
st.sidebar.caption(f"CPU threads: {torch.get_num_threads()}")
st.sidebar.caption(f"Data dir: {DATA_DIR}")

tab_annotate, tab_predict, tab_train, tab_rf, tab_lofo, tab_bench = st.tabs(
    ["🗺️ Annotate (labeled)", "🔮 Predict new data", "🎯 Train",
     "🌲 Random Forest", "🔀 Cross-sensor LOFO", "📊 Benchmark"])

# ---- Annotate ----
with tab_annotate:
    st.subheader("Annotate a field with a trained model")
    fields = list_fields()
    runs = list_runs()
    if not fields:
        st.info("Put .gpkg/.csv/.parquet files in the data/ folder.")
    elif not runs:
        st.warning("No trained models yet — train one in the Train tab first.")
    else:
        c1, c2 = st.columns(2)
        field = c1.selectbox("Field to annotate", fields, key="anno_field")
        run = c2.selectbox("Trained model (run)", runs, key="anno_run")
        col = c1.selectbox("Value column", numeric_columns(field), key="anno_col")
        if st.button("Run annotation", type="primary"):
            with st.spinner("Predicting…"):
                raw = load_table(Path(field), validate=False).copy()
                raw["value"] = pd.to_numeric(raw[col], errors="coerce")
                # Load the architecture that matches the run (name encodes it).
                mn = "ensemble" if "ensemble" in Path(run).name.lower() else "heterogat"
                try:
                    out = predict_field_df(raw, run_dir=Path(run), device=DEVICE, model_name=mn)
                except RuntimeError as e:
                    if DEVICE == "cuda" and ("cuda" in str(e).lower() or "cublas" in str(e).lower()):
                        st.warning("GPU error — using CPU instead.")
                        out = predict_field_df(raw, run_dir=Path(run), device="cpu", model_name=mn)
                    else:
                        st.error(f"Annotation failed: {e}"); st.stop()
                except Exception as e:
                    st.error(f"Could not process this field: {e}"); st.stop()
                pidx = labels_to_indices(out["filtering_category_pred"].values)
                tidx = (labels_to_indices(out["filtering_category"].values)
                        if "filtering_category" in out.columns else np.full(len(out), -1))
                dfm = pd.DataFrame({"lat": out["lat"], "lon": out["lon"], "true": tidx, "pred": pidx,
                                    "probability": out["filtering_probability"],
                                    "entropy": out["filtering_entropy"],
                                    "review_flag": out["filtering_review_flag"]})
            m1, m2, m3 = st.columns(3)
            if (tidx != -1).any():
                v = tidx != -1
                cm = confusion_matrix(tidx[v], pidx[v]); mm = prf_from_cm(cm)
                m1.metric("Macro F1 (present)", f"{mm['macro_f1_present']:.3f}")
                m2.metric("Point accuracy", f"{100*(tidx[v]==pidx[v]).mean():.1f}%")
            m3.metric("Flagged for review", f"{int(dfm['review_flag'].sum())} ({100*dfm['review_flag'].mean():.1f}%)")
            st.plotly_chart(make_map(dfm, f"{Path(field).stem}"), use_container_width=True)
            st.download_button("Download predictions CSV", out.to_csv(index=False).encode(),
                               file_name=f"predictions_{Path(field).stem}.csv")

        st.divider()
        st.caption("Export a full evaluation bundle — every metric (accuracy, balanced acc, "
                   "macro-F1, kappa, MCC, per-class P/R/F1 + ROC/PR-AUC, Clean-vs-outlier "
                   "detection, Brier/NLL/ECE), all figures, an interactive map, and report.pdf — as a .zip.")
        if st.button("📦 Export full report (.zip)", key="anno_export"):
            with st.spinner("Building report (model + figures + metrics + PDF)…"):
                z = export_bundle(field, col, run)
            st.download_button("⬇️ Download report.zip", Path(z).read_bytes(),
                               file_name=Path(z).name, key="anno_dl")

# ---- Predict new data (unlabeled) ----
with tab_predict:
    st.subheader("Predict / clean a NEW field — no labels needed")
    st.caption("Upload any field (or pick one from data/), choose a trained model, and get "
               "predictions + a map + a cleaned CSV. For data that was never annotated.")
    runs = list_runs()
    if not runs:
        st.warning("No trained models yet — train one in the Train tab first.")
    else:
        src = st.radio("Data source", ["Upload a file", "From data/ folder"],
                       horizontal=True, key="pred_src")
        path = None
        if src == "Upload a file":
            up = st.file_uploader("Field file (.gpkg / .csv / .parquet)",
                                  type=["gpkg", "csv", "parquet"], key="pred_up")
            if up is not None:
                import tempfile
                key = f"{up.name}-{up.size}"
                if st.session_state.get("pred_up_key") != key:
                    suf = Path(up.name).suffix or ".csv"
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suf)
                    tmp.write(up.getbuffer())
                    tmp.close()
                    st.session_state["pred_up_key"] = key
                    st.session_state["pred_up_path"] = tmp.name
                path = st.session_state.get("pred_up_path")
        else:
            flds = list_fields()
            path = st.selectbox("Field", flds, key="pred_pick") if flds else None
            if not flds:
                st.info("No files in data/ — switch to Upload a file.")

        if path:
            cols = numeric_columns(str(path))
            c1, c2 = st.columns(2)
            col = c1.selectbox("Value column", cols, index=guess_value_index(cols), key="pred_col")
            run = c2.selectbox("Model (run)", runs, key="pred_run")
            if st.button("Predict", type="primary", key="pred_btn"):
                with st.spinner("Predicting…"):
                    raw = load_table(Path(path), validate=False).copy()
                    raw["value"] = pd.to_numeric(raw[col], errors="coerce")
                    mn = "ensemble" if "ensemble" in Path(run).name.lower() else "heterogat"
                    try:
                        out = predict_field_df(raw, run_dir=Path(run), device=DEVICE, model_name=mn)
                    except RuntimeError as e:
                        if DEVICE == "cuda" and ("cuda" in str(e).lower() or "cublas" in str(e).lower()):
                            st.warning("GPU error — using CPU instead.")
                            out = predict_field_df(raw, run_dir=Path(run), device="cpu", model_name=mn)
                        else:
                            st.error(f"Prediction failed: {e}"); st.stop()
                    except Exception as e:
                        st.error(f"Could not process this field: {e}"); st.stop()
                    pidx = labels_to_indices(out["filtering_category_pred"].values)
                    dfm = pd.DataFrame({"lat": out["lat"], "lon": out["lon"],
                                        "true": np.full(len(out), -1), "pred": pidx,
                                        "probability": out["filtering_probability"],
                                        "entropy": out["filtering_entropy"],
                                        "review_flag": out["filtering_review_flag"]})
                names = {0: "Clean", 1: "Operational Error", 2: "Global Outlier", 3: "Local Outlier"}
                counts = {names[k]: int((pidx == k).sum()) for k in (0, 1, 2, 3)}
                n = max(len(out), 1)
                m1, m2, m3 = st.columns(3)
                m1.metric("Points", f"{len(out):,}")
                m2.metric("Predicted Clean", f"{counts['Clean']:,} ({100*counts['Clean']/n:.0f}%)")
                m3.metric("Flagged for review", f"{int(dfm['review_flag'].sum()):,} "
                                                f"({100*dfm['review_flag'].mean():.0f}%)")
                st.write(counts)
                outl = 100.0 * float((pidx != 0).mean())
                if outl > 40:
                    st.info(f"Note: {outl:.0f}% of points were flagged as non-Clean. If that seems high, "
                            "the model may be out-of-distribution for this field — train (or pick) a model "
                            "on data of the same type.")
                st.plotly_chart(make_map(dfm, Path(path).stem), use_container_width=True)
                d1, d2 = st.columns(2)
                d1.download_button("⬇️ All predictions (CSV)", out.to_csv(index=False).encode(),
                                   file_name=f"predictions_{Path(path).stem}.csv")
                d2.download_button("⬇️ CLEAN points only (CSV)", out[pidx == 0].to_csv(index=False).encode(),
                                   file_name=f"clean_{Path(path).stem}.csv")

            st.divider()
            if st.button("📦 Export full report (.zip)", key="pred_export"):
                with st.spinner("Building report (figures + metrics + PDF)…"):
                    z = export_bundle(path, col, run)
                st.download_button("⬇️ Download report.zip", Path(z).read_bytes(),
                                   file_name=Path(z).name, key="pred_dl")

# ---- Train ----
with tab_train:
    st.subheader("Train a model")
    fields = list_fields()
    if not fields:
        st.info("Put data files in the data/ folder.")
    else:
        sel = st.multiselect("Training field(s)", fields, default=fields, key="tr_fields",
                             help="Defaults to ALL fields (trains one model across them).")
        specs = []
        for f in sel:
            cols = numeric_columns(f)
            c = st.selectbox(f"Value column for {Path(f).name}", cols,
                             index=guess_value_index(cols), key=f"col_{f}")
            specs.append((f, c))
        c1, c2, c3 = st.columns(3)
        model_key = c1.selectbox("Model", ["heterogat", "ensemble"], format_func=lambda k: {"heterogat": "HeteroGAT", "ensemble": "ThreeGNNEnsemble"}[k])
        epochs = c2.slider("Epochs", 5, 100, 30)
        augment = c3.checkbox("Augment (slower; better on small fields)", value=False)
        if st.button("Train", type="primary", disabled=not specs):
            prog = st.progress(0.0)
            chart = st.line_chart(pd.DataFrame({"train_loss": [], "val_macro_f1_present": []}))
            hist = {"train_loss": [], "val_macro_f1_present": []}

            def cb(epoch, row):
                prog.progress(epoch / epochs)
                hist["train_loss"].append(row.get("train_loss", np.nan))
                hist["val_macro_f1_present"].append(row.get("val_macro_f1_present", np.nan))
                chart.add_rows(pd.DataFrame({"train_loss": [row.get("train_loss", np.nan)],
                                             "val_macro_f1_present": [row.get("val_macro_f1_present", np.nan)]}))
            with st.spinner(f"Training {model_key} on {DEVICE.upper()}…"):
                try:
                    run_dir = train_and_save(specs, model_key, epochs, augment, cb, device=DEVICE)
                except RuntimeError as e:
                    if DEVICE == "cuda" and ("cuda" in str(e).lower() or "cublas" in str(e).lower()):
                        st.warning("GPU error (likely an incompatible CUDA/torch build) — retrying on CPU. "
                                   "See the README to install a torch build matching your GPU.")
                        run_dir = train_and_save(specs, model_key, epochs, augment, cb, device="cpu")
                    else:
                        st.error(f"Training failed: {e}"); st.stop()
                except Exception as e:
                    st.error(f"Training failed: {e}"); st.stop()
            st.success(f"Done. Saved model: {run_dir}")
            st.caption("Switch to the Annotate tab to apply it to a field.")

# ---- Random Forest (reproduce Table 1 / the confusion matrix) ----
with tab_rf:
    st.subheader("Train a Random Forest — reproduce the confusion matrix / Table 1")
    st.caption("The tabular model from the paper: a within-field stratified split with "
               "RandomForest(n_estimators=400, class_weight='balanced'), on the richer "
               "within-field feature set (geometry + motion + value features incl. local "
               "mean/std). Trains in seconds on CPU and shows per-class precision / recall / "
               "F1 plus the confusion matrix.")
    fields = list_fields()
    if not fields:
        st.info("Put data files in the data/ folder.")
    else:
        c1, c2, c3 = st.columns(3)
        rf_field = c1.selectbox("Field (must have annotations)", fields, key="rf_field")
        rf_cols = numeric_columns(rf_field)
        rf_col = c2.selectbox("Value column", rf_cols,
                              index=guess_value_index(rf_cols), key="rf_col")
        rf_tf = c3.slider("Test fraction", 0.10, 0.50, 0.25, 0.05, key="rf_tf",
                          help="Share of labeled points held out for testing (paper used 0.25).")
        if st.button("Train Random Forest", type="primary", key="rf_btn"):
            with st.spinner("Preprocessing + training Random Forest…"):
                try:
                    cm, prf, acc, pred_df = train_random_forest(rf_field, rf_col, rf_tf)
                except Exception as e:
                    st.error(f"Could not train: {e}"); st.stop()
            short = ["Clean", "Op", "Global", "Local"]
            names = ["Clean", "Operational Error", "Global Outlier", "Local Outlier"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Macro F1 (present classes)", f"{prf['macro_f1_present']:.3f}")
            m2.metric("Point accuracy", f"{100*acc:.1f}%")
            m3.metric("Test points", f"{int(cm.sum()):,}")

            per = pd.DataFrame({
                "Class": names,
                "Precision": [prf[f"class_{i}_precision"] for i in range(4)],
                "Recall": [prf[f"class_{i}_recall"] for i in range(4)],
                "F1": [prf[f"class_{i}_f1"] for i in range(4)],
                "Support": [int(prf[f"class_{i}_support"]) for i in range(4)],
            })
            st.markdown("**Per-class metrics (held-out test)**")
            st.dataframe(per.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
                         use_container_width=True, hide_index=True)

            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
            for ax, norm, ttl in [(axes[0], False, "Counts"), (axes[1], True, "Row-normalized")]:
                mm = cm.astype(float)
                if norm:
                    rs = mm.sum(1, keepdims=True); rs[rs == 0] = 1; mm = mm / rs
                ax.imshow(mm, cmap="Blues", vmin=0, vmax=(1 if norm else None))
                ax.set_xticks(range(4)); ax.set_yticks(range(4))
                ax.set_xticklabels(short, fontsize=8); ax.set_yticklabels(short, fontsize=8)
                ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(ttl, fontsize=10)
                thr = (0.5 if norm else (cm.max() / 2 if cm.max() else 0.5))
                for i in range(4):
                    for j in range(4):
                        v = mm[i, j]
                        ax.text(j, i, (f"{v:.2f}" if norm else f"{int(cm[i, j])}"),
                                ha="center", va="center", fontsize=8,
                                color="white" if v > thr else "black")
            fig.tight_layout()
            st.pyplot(fig)

            cm_df = pd.DataFrame(cm, index=[f"true_{n}" for n in short],
                                 columns=[f"pred_{n}" for n in short])
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Per-class metrics (CSV)", per.to_csv(index=False).encode(),
                               file_name=f"rf_metrics_{Path(rf_field).stem}.csv", key="rf_dl_m")
            d2.download_button("⬇️ Confusion matrix (CSV)", cm_df.to_csv().encode(),
                               file_name=f"rf_confusion_{Path(rf_field).stem}.csv", key="rf_dl_c")
            d3.download_button("⬇️ Test predictions (CSV)", pred_df.to_csv(index=False).encode(),
                               file_name=f"rf_predictions_{Path(rf_field).stem}.csv", key="rf_dl_p")
            st.caption("Counts depend on the train/test split; the seed is fixed (42), so re-running "
                       "gives identical numbers. This reproduces the paper's Random Forest confusion "
                       "matrix and per-class metrics for the selected field.")

# ---- Cross-sensor leave-one-field-out (RF + XGBoost) ----
with tab_lofo:
    st.subheader("Cross-sensor leave-one-field-out — Random Forest & XGBoost")
    st.caption("Pick the fields, then for each fold one field is held out as the "
               "TEST set and the model is trained on the others (cross-sensor "
               "transfer). The training fields are split with K-means (train + "
               "validation regions); everything is scaled with "
               "value' = (value − field_median) / saved_global_scale.")
    fields = list_fields()
    if len(fields) < 2:
        st.info("Put at least two annotated fields in the data/ folder.")
    else:
        sel = st.multiselect("Fields to use (each is held out once)", fields,
                             default=fields, key="lofo_fields")
        specs = []
        for f in sel:
            cols = numeric_columns(f)
            c = st.selectbox(f"Value column for {Path(f).name}", cols,
                             index=guess_value_index(cols), key=f"lofo_col_{f}")
            specs.append((f, c))
        c1, c2, c3 = st.columns(3)
        use_rf = c1.checkbox("Random Forest", value=True, key="lofo_rf")
        use_xgb = c2.checkbox("XGBoost", value=True, key="lofo_xgb")
        n_clusters = c3.slider("K-means clusters (per training field)", 2, 5, 2,
                               key="lofo_k")
        run_disabled = len(specs) < 2 or not (use_rf or use_xgb)
        if st.button("Run cross-sensor benchmark", type="primary",
                     disabled=run_disabled, key="lofo_btn"):
            models = tuple(m for m, on in (("rf", use_rf), ("xgb", use_xgb)) if on)
            from reta_ml.lofo_benchmark import run_lofo, render_markdown_report
            status = st.empty()
            with st.spinner("Running leave-one-field-out folds…"):
                try:
                    res = run_lofo([f for f, _ in specs], [c for _, c in specs],
                                   models=models, n_clusters=n_clusters,
                                   progress=lambda m: status.write(f"• {m}"))
                except Exception as e:
                    st.error(f"Benchmark failed: {e}"); st.stop()
            status.empty()

            # Summary: test macro-F1 per fold per model.
            mnames = list(next(iter(res["folds"].values()))["models"].keys())
            summ = pd.DataFrame([
                {"Held-out field": fld,
                 **{mn: d["models"][mn]["test"]["macro_f1_present"] for mn in mnames}}
                for fld, d in res["folds"].items()
            ])
            st.markdown("#### Test macro-F1 (present classes) per fold")
            st.dataframe(summ.style.format({mn: "{:.3f}" for mn in mnames}),
                         use_container_width=True, hide_index=True)

            import matplotlib.pyplot as plt
            short = ["Clean", "Op", "Global", "Local"]
            for fld, d in res["folds"].items():
                with st.expander(f"Hold out: {fld}  (train on {', '.join(d['train_fields'])})",
                                 expanded=True):
                    st.caption(f"train {d['n_train']:,} · validation {d['n_val']:,} · "
                               f"test {d['n_test']:,} points · "
                               f"train classes {d['train_classes_present']} · "
                               f"test classes {d['test_classes_present']}")
                    for mn, m in d["models"].items():
                        t = m["test"]
                        st.markdown(f"**{mn}** — test accuracy {t['accuracy']:.3f}, "
                                    f"macro-F1(present) {t['macro_f1_present']:.3f} "
                                    f"(validation macro-F1 {m['validation']['macro_f1_present']:.3f})")
                        per = pd.DataFrame([
                            {"Class": cn, "Precision": pc["precision"],
                             "Recall": pc["recall"], "F1": pc["f1"], "Support": pc["support"]}
                            for cn, pc in t["per_class"].items() if pc["support"] > 0
                        ])
                        cc1, cc2 = st.columns([3, 2])
                        cc1.dataframe(per.style.format(
                            {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
                            use_container_width=True, hide_index=True)
                        cm = np.array(t["confusion_matrix"], float)
                        fig, ax = plt.subplots(figsize=(3.4, 3.0))
                        mnorm = cm.copy(); rs = mnorm.sum(1, keepdims=True); rs[rs == 0] = 1
                        ax.imshow(mnorm / rs, cmap="Blues", vmin=0, vmax=1)
                        ax.set_xticks(range(4)); ax.set_yticks(range(4))
                        ax.set_xticklabels(short, fontsize=7); ax.set_yticklabels(short, fontsize=7)
                        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
                        for i in range(4):
                            for j in range(4):
                                ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center",
                                        fontsize=7, color="white" if (mnorm/rs)[i, j] > 0.5 else "black")
                        fig.tight_layout(); cc2.pyplot(fig)

            # ── Spatial maps: (a) variable  (b) manual annotation  (c) RF cross-field ──
            st.markdown("---")
            st.markdown("#### 🗺️ Spatial maps — annotated vs. RF cross-field predictions")
            from reta_ml.lofo_benchmark import holdout_field_predictions as _holdout_preds
            from matplotlib.lines import Line2D
            from matplotlib.gridspec import GridSpec

            _COLOR  = {0: "#c8cdd2", 1: "#e07b00", 2: "#000000", 3: "#1a6faf"}
            _NAME   = {0: "Clean", 1: "Operational Error", 2: "Global Outlier", 3: "Local Outlier"}
            _MARKER = {0: "o", 1: "D", 2: "s", 3: "^"}
            _STYLE  = {
                0: dict(s=1.6,  alpha=0.40, c=_COLOR[0], edgecolors="none",    lw=0.0, marker=_MARKER[0]),
                1: dict(s=12.0, alpha=0.92, c=_COLOR[1], edgecolors="none",    lw=0.0, marker=_MARKER[1]),
                3: dict(s=12.0, alpha=0.92, c=_COLOR[3], edgecolors="none",    lw=0.0, marker=_MARKER[3]),
                2: dict(s=38.0, alpha=1.0,  c=_COLOR[2], edgecolors="#ffffff", lw=0.7, marker=_MARKER[2]),
            }

            for _i, (_fpath, _fcol) in enumerate(specs):
                _fld_stem = Path(_fpath).stem
                _train_specs_i = [(f2, c2) for _j, (f2, c2) in enumerate(specs) if _j != _i]
                with st.expander(f"🗺️ {_fld_stem}", expanded=True):
                    _map_status = st.empty()
                    with st.spinner(f"Generating map for {_fld_stem}…"):
                        _pr = _holdout_preds(
                            (_fpath, _fcol), _train_specs_i, model="rf", seed=42,
                            progress=lambda m: _map_status.write(f"• {m}"))
                    _map_status.empty()

                    _latL, _lonL, _valL = _pr["lat"], _pr["lon"], _pr["value"]
                    _yl, _rf_pred = _pr["y_true"], _pr["y_pred"]
                    _ratio = 1.0 / np.cos(np.radians(np.nanmean(_latL)))
                    _vmin, _vmax = np.nanpercentile(_valL, [2, 98])

                    _map_fig = plt.figure(figsize=(16, 5.5))
                    _gs = GridSpec(1, 3, figure=_map_fig, wspace=0.10)

                    _axv = _map_fig.add_subplot(_gs[0, 0])
                    _sc = _axv.scatter(_lonL, _latL, s=3, c=_valL, cmap="viridis",
                                       vmin=_vmin, vmax=_vmax, linewidths=0)
                    _axv.set_title(f"(a) {_fcol}", fontsize=14, fontweight="bold", loc="left")
                    _axv.set_aspect(_ratio); _axv.set_xticks([]); _axv.set_yticks([])
                    _map_fig.colorbar(_sc, ax=_axv, fraction=0.046, pad=0.02).ax.tick_params(labelsize=9)

                    for _cell, _title, _labels in [
                        (_gs[0, 1], "(b) Manual annotation", _yl),
                        (_gs[0, 2], f"(c) RF cross-field (acc {_pr['accuracy']:.2f})", _rf_pred),
                    ]:
                        _ax = _map_fig.add_subplot(_cell)
                        for _cls in [0, 1, 3, 2]:
                            _mask = _labels == _cls
                            if _mask.any():
                                _sty = _STYLE[_cls]
                                _ax.scatter(_lonL[_mask], _latL[_mask], s=_sty["s"],
                                            c=_sty["c"], alpha=_sty["alpha"],
                                            edgecolors=_sty["edgecolors"],
                                            linewidths=_sty["lw"], marker=_sty["marker"])
                        _ax.set_title(_title, fontsize=14, fontweight="bold", loc="left")
                        _ax.set_aspect(_ratio); _ax.set_xticks([]); _ax.set_yticks([])

                    _handles = [
                        Line2D([0], [0], marker=_MARKER[_cls], linestyle="", markersize=10,
                               markerfacecolor=_COLOR[_cls],
                               markeredgecolor=("#ffffff" if _cls == 2 else "none"),
                               markeredgewidth=(0.7 if _cls == 2 else 0.0),
                               label=_NAME[_cls]) for _cls in range(4)]
                    _map_fig.legend(handles=_handles, loc="lower center", ncol=4,
                                    frameon=False, fontsize=13, bbox_to_anchor=(0.5, 0.01))
                    _map_fig.subplots_adjust(bottom=0.12, top=0.88)
                    st.pyplot(_map_fig)
                    plt.close(_map_fig)
                    st.caption(f"RF trained on: {', '.join(_pr['train_fields'])}  ·  "
                               f"cross-field accuracy {_pr['accuracy']:.3f}")

            st.markdown("---")
            report_md = render_markdown_report(res)
            from reta_ml.lofo_benchmark import render_docx_report
            import json as _json
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ Detailed report (Markdown)", report_md.encode(),
                               file_name="lofo_report.md", key="lofo_dl_md")
            d2.download_button("⬇️ Raw results (JSON)",
                               _json.dumps(res, indent=2).encode(),
                               file_name="lofo_results.json", key="lofo_dl_json")
            try:
                docx_bytes = render_docx_report(res)
                d3.download_button("⬇️ Detailed report (Word)", docx_bytes,
                                   file_name="lofo_report.docx", key="lofo_dl_docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception:
                d3.info("Install `python-docx` to enable Word export")
            with st.expander("📄 Detailed report (with code snippets)"):
                st.markdown(report_md)

# ---- Benchmark ----
with tab_bench:
    st.subheader("Leave-one-field-out benchmark")
    st.caption("Compares RETA, Random Forest, XGBoost, HistGradientBoosting, HeteroGAT, ThreeGNNEnsemble.")
    reports = sorted((REPO_ROOT / "reports" / "benchmark").glob("*/BENCHMARK.md"), reverse=True) \
        if (REPO_ROOT / "reports" / "benchmark").exists() else []
    if reports:
        st.markdown(f"**Latest report:** `{reports[0].parent.name}`")
        st.markdown(reports[0].read_text())
    else:
        st.info("No benchmark yet. Run `python scripts/benchmark.py` (it can be heavy on CPU; fast on your GPU).")
