import json
import joblib
import numpy as np
import pandas as pd

def load_model_and_assets():
    """
    Load model, scaler, features, and threshold from artifacts/.
    - lgbm_model.pkl      (required)
    - scaler.pkl          (optional but recommended)
    - features.pkl        (required)
    - threshold.json      (required; supports {'threshold': x} or {'best_threshold': x})
    """
    try:
        model = joblib.load("artifacts/lgbm_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("artifacts/lgbm_model.pkl not found.")

    try:
        scaler = joblib.load("artifacts/scaler.pkl")
    except FileNotFoundError:
        scaler = None  # allow running without scaler

    try:
        feature_list = joblib.load("artifacts/features.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("artifacts/features.pkl not found.")

    try:
        with open("artifacts/threshold.json", "r") as f:
            data = json.load(f)
            threshold = data.get("threshold", data.get("best_threshold", None))
            if threshold is None:
                raise ValueError("threshold.json must contain 'threshold' or 'best_threshold'.")
            threshold = float(threshold)
    except FileNotFoundError:
        raise FileNotFoundError("artifacts/threshold.json not found.")

    return model, scaler, feature_list, threshold

def prepare_input_row(row: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Ensure row contains exactly the model features (missing filled with 0.0).
    """
    X = row.copy()
    for c in features:
        if c not in X.columns:
            X[c] = 0.0
    return X[features]

def scale_numeric(X: pd.DataFrame, scaler=None, binary_cols=None):
    """
    Apply scaler to numeric columns, skipping binary flags if provided.
    If scaler is None, returns X unchanged.
    """
    if scaler is None:
        return X

    if binary_cols is None:
        binary_cols = ["LowRiskFlag", "OCF_Deficit_2of3"]

    num_cols = [c for c in X.columns if c not in binary_cols]
    X_scaled = X.copy()
    if len(num_cols) > 0:
        X_scaled[num_cols] = scaler.transform(X[num_cols])
    return X_scaled

def predict_proba_label_safe(model, X_scaled: pd.DataFrame, threshold: float):
    """
    Predict probability and label. Works for binary classifiers with predict_proba.
    """
    y_proba = model.predict_proba(X_scaled)
    if y_proba.ndim == 1:
        # Some models return shape (n_samples,)
        proba = float(y_proba[0])
    else:
        proba = float(y_proba[:, 1][0])
    label = int(proba >= threshold)
    return proba, label
