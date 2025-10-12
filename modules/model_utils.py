import pickle
import json
import numpy as np

def load_model_and_assets():
    with open("artifacts/lgbm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("artifacts/features.pkl", "rb") as f:
        feature_list = pickle.load(f)
    with open("artifacts/threshold.json", "r") as f:
        threshold = json.load(f)["best_threshold"]
    return model, scaler, feature_list, threshold

def predict_proba_single(model, scaler, x_input, threshold):
    x_scaled = scaler.transform(x_input)
    proba = model.predict_proba(x_scaled)[0, 1]
    label = int(proba >= threshold)
    return proba, label
