import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_utils import load_master_data, require_columns
from modules.model_utils import (
    load_model_and_assets,
    prepare_input_row,
    scale_numeric,
    predict_proba_label_safe,
)
from modules.viz_utils import (
    plot_default_distribution_year,
    plot_default_rate_by_sector,
    plot_probability_histogram,
    plot_roc_auc_plotly,
    plot_precision_recall_plotly,
)

st.set_page_config(
    page_title="Corporate Default Risk Scoring System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== Header =====
st.title("Corporate Default Risk Scoring System")
st.markdown(
    "This internal tool provides real-time corporate default risk scoring using a pre-trained LightGBM model. "
    "It is designed for financial institutions to assess firm-level probability of default."
)

# ===== Load dataset (required) =====
st.subheader("Master Dataset")
try:
    df_master = load_master_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except pd.errors.EmptyDataError:
    st.error("Dataset file exists but is empty. Please upload a valid dataset to the repository.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading dataset: {e}")
    st.stop()

# Validate minimal columns for downstream use
required_cols = ["Ticker", "Year"]
missing = [c for c in required_cols if c not in df_master.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}.")
    st.stop()

st.dataframe(df_master.head(50), use_container_width=True)
st.caption(f"Dataset loaded successfully: {df_master.shape[0]:,} rows × {df_master.shape[1]:,} columns")

# Optional EDA (if Default/Sector exist)
st.subheader("Exploratory Data Overview")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_default_distribution_year(df_master), use_container_width=True)
with c2:
    st.plotly_chart(plot_default_rate_by_sector(df_master), use_container_width=True)

# ===== Load model assets (required) =====
try:
    model, scaler, feature_list, threshold = load_model_and_assets()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unable to load model artifacts: {e}")
    st.stop()

# ===== Scoring interface =====
st.subheader("Firm-Level Default Scoring")
left, right = st.columns(2)
with left:
    ticker_input = st.text_input("Enter Ticker (e.g., VNM, HPG)", "").strip().upper()
with right:
    year_input = st.number_input("Enter Fiscal Year", min_value=1990, max_value=2100, step=1)

run_btn = st.button("Run Default Scoring", type="primary", use_container_width=True)

if run_btn:
    if not ticker_input or not year_input:
        st.info("Please enter both Ticker and Year.")
    else:
        row = df_master[
            (df_master["Ticker"].astype(str).str.upper() == ticker_input)
            & (df_master["Year"] == int(year_input))
        ]
        if row.empty:
            st.warning("No data found for the specified Ticker and Year.")
        else:
            # Prepare X
            try:
                X = prepare_input_row(row, feature_list)
            except Exception as e:
                st.error(f"Failed to assemble feature row: {e}")
                st.stop()

            # Scale & Predict
            try:
                X_scaled = scale_numeric(X, scaler=scaler)
                proba, label = predict_proba_label_safe(model, X_scaled, threshold)
            except Exception as e:
                st.error(f"Scoring failed: {e}")
                st.stop()

            # Display result
            st.metric(
                label=f"Predicted Default Probability ({ticker_input} - {int(year_input)})",
                value=f"{proba:.2%}",
            )
            st.markdown(
                f"**Classification:** {'Default (High Risk)' if label == 1 else 'Non-Default (Low Risk)'}  \n"
                f"**Decision Threshold:** {threshold:.2f}"
            )

# ===== Evaluation (if Default exists) =====
if "Default" in df_master.columns:
    st.subheader("Model Evaluation")
    try:
        X_eval = df_master.copy()
        # Ensure feature completeness
        for c in feature_list:
            if c not in X_eval.columns:
                X_eval[c] = 0.0
        X_eval = X_eval[feature_list]

        # For evaluation, we apply scaler if available
        try:
            X_eval_scaled = scale_numeric(X_eval, scaler=scaler)
        except Exception:
            X_eval_scaled = X_eval

        y_true = df_master["Default"].astype(int)

        roc_fig = plot_roc_auc_plotly(model, X_eval_scaled, y_true)
        pr_fig = plot_precision_recall_plotly(model, X_eval_scaled, y_true)

        a, b = st.columns(2)
        with a:
            st.plotly_chart(roc_fig, use_container_width=True)
        with b:
            st.plotly_chart(pr_fig, use_container_width=True)

        # Optional probability histogram (predict on full dataset)
        try:
            y_proba = model.predict_proba(X_eval_scaled)[:, 1]
            tmp = df_master.copy()
            tmp["Default_Proba"] = y_proba
            st.subheader("Predicted Probability Distribution")
            st.plotly_chart(plot_probability_histogram(tmp), use_container_width=True)
        except Exception:
            pass

    except Exception as e:
        st.info(f"Evaluation skipped due to error: {e}")
else:
    st.info("Column 'Default' not found in dataset. Skipping evaluation charts.")
