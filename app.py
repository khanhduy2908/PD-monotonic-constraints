import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_utils import load_master_data
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

# =========================
# Header
# =========================
st.title("Corporate Default Risk Scoring System")
st.markdown(
    "This internal tool provides real-time corporate default risk scoring using a pre-trained LightGBM model. "
    "It is designed for financial institutions to assess firm-level probability of default."
)

# =========================
# Load dataset
# =========================
st.subheader("Master Dataset")
try:
    df_master = load_master_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except pd.errors.EmptyDataError:
    st.error("Dataset file exists but is empty. Please add a valid dataset to the repository.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading dataset: {e}")
    st.stop()

# Validate minimal columns
required_cols = ["Ticker", "Year"]
missing = [c for c in required_cols if c not in df_master.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

st.dataframe(df_master.head(50), use_container_width=True)
st.caption(f"Dataset loaded successfully: {df_master.shape[0]:,} rows × {df_master.shape[1]:,} columns")

# =========================
# EDA (with stable keys to avoid duplicate element ids)
# =========================
st.subheader("Exploratory Data Overview")
eda_col1, eda_col2 = st.columns(2)
with eda_col1:
    st.plotly_chart(
        plot_default_distribution_year(df_master),
        use_container_width=True,
        key="eda_default_year",
    )
with eda_col2:
    st.plotly_chart(
        plot_default_rate_by_sector(df_master),
        use_container_width=True,
        key="eda_default_sector",
    )

# =========================
# Load model assets
# =========================
try:
    model, scaler, feature_list, threshold = load_model_and_assets()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Unable to load model artifacts: {e}")
    st.stop()

# =========================
# Firm-level Scoring (Ticker + Year selection)
# =========================
st.subheader("Firm-Level Default Scoring")

# Build controls: ticker list from dataset, year list filtered by ticker
tickers = (
    df_master["Ticker"]
    .dropna()
    .astype(str)
    .str.upper()
    .sort_values()
    .unique()
    .tolist()
)

sc_col1, sc_col2, sc_col3 = st.columns([1.2, 1.0, 0.8])

with sc_col1:
    ticker_sel = st.selectbox(
        "Select Ticker",
        options=tickers if len(tickers) > 0 else ["—"],
        index=0 if len(tickers) > 0 else 0,
        key="sel_ticker",
    )

# years for selected ticker
years_for_ticker = (
    df_master[df_master["Ticker"].astype(str).str.upper() == ticker_sel]["Year"]
    .dropna()
    .astype(int)
    .sort_values()
    .unique()
    .tolist()
)

with sc_col2:
    year_sel = st.selectbox(
        "Select Fiscal Year",
        options=years_for_ticker if len(years_for_ticker) > 0 else [],
        index=(len(years_for_ticker) - 1) if len(years_for_ticker) > 0 else 0,
        key="sel_year",
    )

with sc_col3:
    run_btn = st.button("Run Default Scoring", type="primary", use_container_width=True, key="btn_score")

if run_btn:
    # Locate the exact row
    row = df_master[
        (df_master["Ticker"].astype(str).str.upper() == ticker_sel)
        & (df_master["Year"].astype(int) == int(year_sel))
    ]
    if row.empty:
        st.warning("No data found for the selected Ticker and Fiscal Year.")
    else:
        # Keep only the first match if duplicates exist
        row = row.iloc[[0]]

        # Assemble feature vector, scale, predict
        try:
            X = prepare_input_row(row, feature_list)
            X_scaled = scale_numeric(X, scaler=scaler)
            proba, label = predict_proba_label_safe(model, X_scaled, threshold)
        except Exception as e:
            st.error(f"Scoring failed: {e}")
        else:
            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.metric("Predicted Default Probability", f"{proba:.2%}")
            with met_col2:
                st.metric("Classification", "Default (High Risk)" if label == 1 else "Non-Default (Low Risk)")
            with met_col3:
                st.metric("Decision Threshold", f"{threshold:.2f}")

            with st.expander("Feature values used for scoring"):
                st.dataframe(X, use_container_width=True)

# =========================
# Evaluation (optional; requires 'Default' label)
# =========================
if "Default" in df_master.columns:
    st.subheader("Model Evaluation")
    try:
        X_eval = df_master.copy()
        # Ensure every feature is present
        for c in feature_list:
            if c not in X_eval.columns:
                X_eval[c] = 0.0
        X_eval = X_eval[feature_list]

        # Scale for evaluation
        try:
            X_eval_scaled = scale_numeric(X_eval, scaler=scaler)
        except Exception:
            X_eval_scaled = X_eval

        y_true = df_master["Default"].astype(int)

        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            st.plotly_chart(
                plot_roc_auc_plotly(model, X_eval_scaled, y_true),
                use_container_width=True,
                key="eval_roc",
            )
        with ev_col2:
            st.plotly_chart(
                plot_precision_recall_plotly(model, X_eval_scaled, y_true),
                use_container_width=True,
                key="eval_pr",
            )

        # Probability histogram
        try:
            y_proba = model.predict_proba(X_eval_scaled)[:, 1]
            tmp = df_master.copy()
            tmp["Default_Proba"] = y_proba
            st.subheader("Predicted Probability Distribution")
            st.plotly_chart(
                plot_probability_histogram(tmp),
                use_container_width=True,
                key="eval_hist",
            )
        except Exception:
            pass

    except Exception as e:
        st.info(f"Evaluation skipped due to error: {e}")
else:
    st.info("Column 'Default' not found in dataset. Skipping evaluation charts.")
