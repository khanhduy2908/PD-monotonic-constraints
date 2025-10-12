import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from modules.data_utils import load_master_data
from modules.viz_utils import (
    plot_default_distribution_year,
    plot_default_rate_by_sector,
    plot_probability_histogram,
    plot_roc_auc_plotly,
    plot_precision_recall_plotly
)

# =========================
# Load assets
# =========================
df_master = load_master_data()

with open("artifacts/features.pkl", "rb") as f:
    model_features = joblib.load(f)
scaler = joblib.load("artifacts/scaler.pkl")
model = joblib.load("artifacts/lgbm_model.pkl")
with open("artifacts/threshold.json", "r") as f:
    threshold = json.load(f)["best_threshold"]

# =========================
# Header
# =========================
st.set_page_config(page_title="Corporate Default Risk Scoring System", layout="wide")
st.title("Corporate Default Risk Scoring System")
st.caption("Internal credit risk scoring platform using pre-trained LightGBM model")

# =========================
# Data Section
# =========================
st.subheader("Master Dataset")
try:
    st.dataframe(df_master.head(50), use_container_width=True)
    st.caption(f"Dataset loaded successfully: {df_master.shape[0]} rows × {df_master.shape[1]} columns")
except Exception as e:
    st.error(str(e))
    st.stop()

# =========================
# EDA Section
# =========================
st.subheader("Exploratory Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(plot_default_distribution_year(df_master), use_container_width=True)
with col2:
    st.plotly_chart(plot_default_rate_by_sector(df_master), use_container_width=True)

# =========================
# Scoring Section
# =========================
st.subheader("Firm-level Default Scoring")

col_ticker, col_year = st.columns(2)
with col_ticker:
    ticker_input = st.text_input("Enter Ticker (e.g., VNM, HPG):").strip().upper()
with col_year:
    year_input = st.number_input("Enter Fiscal Year:", min_value=2000, max_value=2100, step=1)

if st.button("Run Default Scoring"):
    if ticker_input and year_input:
        firm_row = df_master[(df_master["Ticker"] == ticker_input) & (df_master["Year"] == year_input)]
        if firm_row.empty:
            st.warning("No data found for this Ticker and Year.")
        else:
            X_firm = firm_row[model_features]
            X_firm_scaled = X_firm.copy()
            num_cols = [c for c in model_features if c not in ["LowRiskFlag", "OCF_Deficit_2of3"]]
            X_firm_scaled[num_cols] = scaler.transform(X_firm[num_cols])

            proba = model.predict_proba(X_firm_scaled)[:, 1][0]
            label = int(proba >= threshold)

            st.metric(
                label=f"Predicted Default Probability ({ticker_input} - {year_input})",
                value=f"{proba:.2%}",
                delta="High Risk" if label == 1 else "Low Risk",
                delta_color="inverse" if label == 0 else "normal"
            )
    else:
        st.info("Please enter both Ticker and Year to score.")

# =========================
# Evaluation Dashboard
# =========================
st.subheader("Model Evaluation")
st.plotly_chart(plot_roc_auc_plotly(model, df_master[model_features], df_master["Default"]), use_container_width=True)
st.plotly_chart(plot_precision_recall_plotly(model, df_master[model_features], df_master["Default"]), use_container_width=True)
