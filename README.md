# Instagram Engagement Predictor

A clean Streamlit deployment for predicting raw Instagram engagement.

## Files

- `app.py` — Streamlit application.
- `model.npz` — compact exported Random Forest tree arrays.
- `model_meta.json` — model metadata and feature definitions.
- `train_model.py` — reproducible training/export script.
- `requirements.txt` — runtime dependencies only.
- `requirements-train.txt` — dependencies for retraining.
- `data/` — the two supplied CSV datasets.

## Why this version is deployment-safe

The Streamlit app does **not** load a pickled scikit-learn estimator. The Random Forest is exported as compact NumPy arrays and traversed directly by the app. This avoids the scikit-learn pickle/internal-module compatibility problem that affected the previous `model_artifacts.joblib`.

The deployed app therefore only needs Streamlit, NumPy and pandas.

## Streamlit deployment

Put every file in this repository and deploy `app.py` as the main file.

No `model_artifacts.joblib` is needed.

## Retraining

From the project root:

```bash
python train_model.py
```

This replaces `model.npz` and `model_meta.json`.

## Model

- Algorithm: Random Forest Regressor
- Trees: 150
- Max depth: 12
- Min samples leaf: 2
- Max features: 0.8
- Random state: 42
