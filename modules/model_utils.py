import streamlit as st
import joblib, json
import numpy as np
import pandas as pd

@st.cache_resource(show_spinner=False)
def load_model_and_assets():
    model = joblib.load("artifacts/lgbm_model.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    features = joblib.load("artifacts/features.pkl")
    with open("artifacts/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    try:
        with open("artifacts/constraints.json") as f:
            constraints = json.load(f)
    except Exception:
        constraints = {}
    return model, scaler, features, float(threshold), constraints

def prepare_input_row(row: pd.DataFrame, features: list) -> pd.DataFrame:
    # Ensure columns order and missing columns are handled
    X = row.copy()
    missing = [c for c in features if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[features]
    return X

def scale_numeric(X: pd.DataFrame, binary_cols=None, scaler=None):
    if binary_cols is None:
        binary_cols = ["LowRiskFlag","OCF_Deficit_2of3"]
    num_cols = [c for c in X.columns if c not in binary_cols]
    X_scaled = X.copy()
    if scaler is not None and len(num_cols) > 0:
        X_scaled[num_cols] = scaler.transform(X[num_cols])
    return X_scaled

def predict_proba_label(model, X_scaled: pd.DataFrame, threshold: float):
    proba = model.predict_proba(X_scaled)[:,1]
    label = (proba >= threshold).astype(int)
    return float(proba[0]), int(label[0])

def score_dataframe(model, df: pd.DataFrame, scaler, features: list, threshold: float):
    # Score a whole dataframe (vectorized)
    X = df.copy()
    missing = [c for c in features if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[features]
    binary_cols = ["LowRiskFlag","OCF_Deficit_2of3"]
    num_cols = [c for c in X.columns if c not in binary_cols]
    X_scaled = X.copy()
    if scaler is not None and len(num_cols) > 0:
        X_scaled[num_cols] = scaler.transform(X[num_cols])
    proba = model.predict_proba(X_scaled)[:,1]
    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["Default_Proba"] = proba
    out["Default_Pred"] = pred
    return out
