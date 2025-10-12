import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modules.ui_components import page_title, section_header, kpi_row
from modules.data_utils import load_master_data, filter_by_ticker_period
from modules.feature_engineering import preprocess_and_create_features
from modules.model_utils import (
    load_model_and_assets,
    prepare_input_row,
    scale_numeric,
    predict_proba_label_safe,
    predict_period_for_ticker,
    quick_diagnostics,
)
from modules.viz_utils import (
    plot_default_distribution_year,
    plot_default_rate_by_sector,
    plot_probability_histogram,
    plot_roc_auc_plotly,
    plot_precision_recall_plotly,
    plot_pd_line_forecast,
    plot_pd_risk_bucket_bar,
)

st.set_page_config(page_title="Corporate Default Risk Scoring System",
                   layout="wide", initial_sidebar_state="expanded")

page_title(
    "Corporate Default Risk Scoring System",
    "Institutional-grade dashboard for firm-level default risk forecasting using a pre-trained monotonic tree-based model."
)

try:
    raw_df = load_master_data()
    df_master = preprocess_and_create_features(raw_df)
except Exception as e:
    st.error(f"Failed to load/preprocess dataset: {e}")
    st.stop()

for col in ["Ticker", "Year"]:
    if col not in df_master.columns:
        st.error(f"Dataset missing required column: {col}"); st.stop()

with st.sidebar:
    st.header("Input Panel")
    tickers = df_master["Ticker"].astype(str).str.upper().sort_values().unique().tolist()
    ticker_sel = st.selectbox("Select Ticker", options=tickers, index=0 if tickers else 0)
    y_list = df_master[df_master["Ticker"].astype(str).str.upper()==ticker_sel]["Year"].astype(int).sort_values().unique().tolist()
    if not y_list:
        st.error("No available years for this ticker."); st.stop()
    default_start = max(min(y_list), y_list[-1]-3)
    default_end   = y_list[-1]
    start_year = st.number_input("Start Year", min_value=min(y_list), max_value=max(y_list), value=default_start, step=1, key="start_year")
    end_year   = st.number_input("End Year",   min_value=min(y_list), max_value=max(y_list), value=default_end,   step=1, key="end_year")
    run_btn = st.button("Run Forecast", type="primary", use_container_width=True)

try:
    model, scaler, feature_list, threshold = load_model_and_assets()
except Exception as e:
    st.error(f"Unable to load model artifacts: {e}")
    st.stop()

diag = quick_diagnostics(df_master, feature_list, scaler)
if diag.get("missing_features"):
    st.warning(f"Warning: Missing features in dataset: {diag['missing_features'][:10]}{'...' if len(diag['missing_features'])>10 else ''}")

section_header("Exploratory Data Overview (Selected Ticker & Period)")
subset = filter_by_ticker_period(df_master, ticker_sel, int(start_year), int(end_year))
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(plot_default_distribution_year(subset), use_container_width=True, key="eda_year_subset")
with c2:
    st.plotly_chart(plot_default_rate_by_sector(subset), use_container_width=True, key="eda_sector_subset")

section_header("Firm-Level PD Forecast (Multi-Year)")
if run_btn:
    try:
        years = list(range(int(start_year), int(end_year)+1))
        forecast_df = predict_period_for_ticker(df=df_master, ticker=ticker_sel, years=years,
                                                model=model, scaler=scaler, features=feature_list, threshold=threshold)
        latest = forecast_df.sort_values("Year").iloc[-1]
        kpi_row([
            ("Latest Year", f"{int(latest['Year'])}"),
            ("Predicted Default Probability (Latest)", f"{latest['Default_Proba']:.2%}"),
            ("Decision Threshold", f"{threshold:.2f}")
        ])
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(plot_pd_line_forecast(forecast_df, threshold=threshold),
                            use_container_width=True, key="pd_line_forecast")
        with cc2:
            st.plotly_chart(plot_pd_risk_bucket_bar(forecast_df, threshold=threshold),
                            use_container_width=True, key="pd_risk_bucket")
        st.subheader("Predicted Probability Distribution (Forecast Horizon)")
        st.plotly_chart(plot_probability_histogram(forecast_df), use_container_width=True, key="pd_hist_forecast")
    except Exception as e:
        st.error(f"Forecast failed: {e}")
else:
    st.info("Set Ticker and Year range on the left, then click Run Forecast.")

if "Default" in df_master.columns:
    section_header("Model Evaluation (Full Labeled Dataset)")
    try:
        X_eval = df_master.copy()
        for c in feature_list:
            if c not in X_eval.columns:
                X_eval[c] = 0.0
        X_eval = X_eval[feature_list]
        try:
            X_eval_scaled = scale_numeric(X_eval, scaler=scaler)
        except Exception:
            X_eval_scaled = X_eval
        y_true = df_master["Default"].astype(int)
        e1, e2 = st.columns(2)
        with e1:
            st.plotly_chart(plot_roc_auc_plotly(model, X_eval_scaled, y_true), use_container_width=True, key="roc_full")
        with e2:
            st.plotly_chart(plot_precision_recall_plotly(model, X_eval_scaled, y_true), use_container_width=True, key="pr_full")
    except Exception as e:
        st.info(f"Evaluation skipped: {e}")
else:
    st.info("Column 'Default' not found in dataset. Skipping evaluation charts.")
