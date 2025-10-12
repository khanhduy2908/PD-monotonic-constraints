import json
import joblib
import numpy as np
import pandas as pd

def load_model_and_assets():
    try:
        model = joblib.load("artifacts/lgbm_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("artifacts/lgbm_model.pkl not found.")
    try:
        scaler = joblib.load("artifacts/scaler.pkl")
    except FileNotFoundError:
        scaler = None
    try:
        feature_list = joblib.load("artifacts/features.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("artifacts/features.pkl not found.")
    try:
        with open("artifacts/threshold.json", "r") as f:
            data = json.load(f)
            threshold = float(data.get("threshold", data.get("best_threshold", None)))
            if threshold is None:
                raise ValueError
    except FileNotFoundError:
        raise FileNotFoundError("artifacts/threshold.json not found.")
    except Exception:
        raise ValueError("threshold.json must contain 'threshold' or 'best_threshold'.")
    return model, scaler, feature_list, threshold

def prepare_input_row(row: pd.DataFrame, features: list) -> pd.DataFrame:
    X = row.copy()
    for c in features:
        if c not in X.columns:
            X[c] = 0.0
    return X[features]

def scale_numeric(X: pd.DataFrame, scaler=None, binary_cols=None):
    if scaler is None:
        return X
    if binary_cols is None:
        binary_cols = ["LowRiskFlag", "OCF_Deficit_2of3"]
    Xs = X.copy()
    num_cols = [c for c in X.columns if c not in binary_cols]
    if len(num_cols) > 0:
        Xs[num_cols] = scaler.transform(X[num_cols])
    return Xs

def predict_proba_label_safe(model, X_scaled: pd.DataFrame, threshold: float):
    y_proba = model.predict_proba(X_scaled)
    proba = float(y_proba[:, 1][0]) if y_proba.ndim > 1 else float(y_proba[0])
    label = int(proba >= threshold)
    return proba, label

def predict_period_for_ticker(df: pd.DataFrame, ticker: str, years: list, model, scaler, features: list, threshold: float) -> pd.DataFrame:
    rows = []
    df2 = df.copy()
    df2["Ticker"] = df2["Ticker"].astype(str).str.upper()
    for y in years:
        r = df2[(df2["Ticker"] == str(ticker).upper()) & (df2["Year"].astype(int) == int(y))]
        if r.empty:
            continue
        r = r.iloc[[0]]
        X = prepare_input_row(r, features)
        Xs = scale_numeric(X, scaler=scaler)
        proba, label = predict_proba_label_safe(model, Xs, threshold)
        rows.append({"Ticker": str(ticker).upper(), "Year": int(y), "Default_Proba": proba, "Pred": int(label)})
    if not rows:
        raise ValueError("No rows found for the selected ticker and period.")
    return pd.DataFrame(rows).sort_values("Year")

def quick_diagnostics(df: pd.DataFrame, features: list, scaler) -> dict:
    d = {}
    missing = [c for c in features if c not in df.columns]
    d["missing_features"] = missing
    d["scaler_present"] = scaler is not None
    return d
