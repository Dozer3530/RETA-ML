# RETA-ML

A MACHINE LEARNING FRAMEWORK FOR AUTOMATED ANOMALY DETECTION IN PRECISION AGRICULTURE GEOSPATIAL DATA

Zachary Komarnisky · Felippe H. S. Karp — Olds College of Agriculture & Technology
Presented at the *17th International Conference on Precision Agriculture (ICPA) /
11th Brazilian Congress on Precision and Digital Agriculture (ConBAP), 2026.*

This is the public repo to the RETA-ML project. It ships the trained models and the
framework code — enough to inspect the models, run them on your own sensor data, and
build on the work.

---

## What is the idea?

> A model trained on some sensors can clean data from a sensor it has **never seen** —
> but only the errors that *move with the machine*. Spatial anomalies don't transfer.

On-the-go soil and yield sensors are fast but produce errors — headland turns, overlaps,
speed changes, spatial spikes — that quietly corrupt precision-ag maps. Cleaning that
data is manual, slow, and different for every sensor. We ask whether one model can learn
cleaning once and apply it across sensors, and we measure exactly where that transfer
holds and where it breaks.

## What we found (leave-one-sensor-out)

Trained on two sensors, tested on the held-out third, macro-averaged over three folds:

| Class | RF&nbsp;P | RF&nbsp;R | **RF&nbsp;F1** | XGB&nbsp;P | XGB&nbsp;R | XGB&nbsp;F1 |
|---|---|---|---|---|---|---|
| Clean | 0.96 | 0.94 | **0.95** | 0.96 | 0.85 | 0.90 |
| Operational | 0.46 | 0.63 | **0.52** | 0.30 | 0.65 | 0.40 |
| Local | 0.44 | 0.24 | **0.30** | 0.32 | 0.38 | 0.34 |

Random Forest macro-accuracy **0.90**, XGBoost 0.83. Clean data and operational errors
(the kinematic ones) transfer across sensors; local spatial anomalies do not — they look
different on every sensor, which motivates a spatially-explicit (graph) model next.
*(Global-outlier class excluded from reporting: only 14 examples across all fields.)*

---

## Repository layout

```
reta_ml/     the framework — preprocessing, features, models, evaluation
models/      trained models + a model card
app.py       Streamlit app to load a field, run a model, and inspect results
requirements.txt / environment.yml   dependencies
```

## Trained models (`models/`)

| File | What it is |
|---|---|
| `random_forest.joblib` | Random Forest (400 trees, class-balanced), the poster's headline model. `joblib.load(...)`. |
| `xgboost.json` | XGBoost (400 trees, depth 6). `XGBClassifier().load_model(...)`. |
| `heterogat_best_model.pt` | HeteroGAT graph neural network checkpoint (torch) — the spatially-explicit next-step model. |
| `model_card.json` | Feature list, config, training data, and per-class counts. |

The RF/XGBoost models are trained on **all** expert-labeled points across the three sensors (49,665 points; the annotated
dataset itself is not distributed in this repository) with the locked-in configuration: 13 sensor-agnostic features
(field-relative motion + fixed-radius 15/30/45 m spatial z-scores), seed 42, no scaler.

---

## Quick start

```bash
# 1. install
pip install -r requirements.txt          # or: conda env create -f environment.yml

# 2. use a trained model in a few lines
python - <<'PY'
import joblib, json
from reta_ml.load import load_table
from reta_ml.preprocess import preprocess_pipeline, recompute_value_stats
from reta_ml.lofo_benchmark import _feature_matrix

feats = json.load(open("models/feature_columns.json"))
rf = joblib.load("models/random_forest.joblib")

raw = load_table("your_field.gpkg", validate=False)   # bring your own sensor data
raw["value"] = raw["<your_value_column>"]
proc = preprocess_pipeline(raw, main_variable="value", local_stats_mode="fixedbands")
proc = recompute_value_stats(proc, main_variable="value", local_stats_mode="fixedbands")
pred = rf.predict(_feature_matrix(proc, feats))     # 0=Clean 1=Operational 2=Global 3=Local
print(pred[:20])
PY

# 3. or explore interactively
streamlit run app.py
```

---

## Method, briefly

- **Features (13, sensor-agnostic):** field-relative speed / distance / acceleration,
  turn & short-segment flags, whole-field z-score, fixed-radius spatial z-scores at
  15 / 30 / 45 m, and ratio to the transect mean. Deliberately *relative* and
  *fixed-radius* so the same feature vector is comparable across a soil sensor and a
  combine.
- **Models:** Random Forest and XGBoost (class-balanced), with a HeteroGAT graph neural
  network as the spatially-explicit next step.
- **Evaluation:** Leave-One-Sensor-Out — train on two sensors, test on the third — the
  honest test of cross-sensor transfer.

## Citation

If you use this work, please cite the ICPA/ConBAP 2026 paper (Komarnisky & Karp, 2026).

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

Supported by an NSERC CCI Mobilize Grant (CCMOB-2024-00038), the Olds College Office of
Research Services, and the Werklund School of Agriculture Technology.
