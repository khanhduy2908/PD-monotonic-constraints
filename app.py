import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from modules.data_utils import load_master_data, filter_by_ticker_period
from modules.model_utils import (
    load_model_and_assets,
    prepare_input_row,
    scale_numeric,
    predict_proba_label_safe,
    predict_period_for_ticker,
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

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Corporate Default Risk Scoring System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Header ----------------
st.title("Corporate Default Risk Scoring System")
st.markdown(
    "Institutional-grade dashboard for firm-level default risk forecasting using a pre-trained, "
    "monotonic tree-based model. The tool loads model artifacts from the repository and never retrains online."
)

# ---------------- Load data ----------------
try:
    df_master = load_master_data()
    df_master = preprocess_and_create_features(df_master)
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except pd.errors.EmptyDataError:
    st.error("Dataset file exists but is empty. Please add a valid dataset (CSV/XLSX) to the repository.")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error while loading dataset: {e}")
    st.stop()

required_cols = ["Ticker", "Year"]
missing = [c for c in required_cols if c not in df_master.columns]
if missing:
    st.error(f"Dataset is missing required columns: {missing}")
    st.stop()

# ---------------- Sidebar: Input Panel ----------------
with st.sidebar:
    st.header("Input Panel")
    tickers = (
        df_master["Ticker"].dropna().astype(str).str.upper().sort_values().unique().tolist()
    )
    ticker_sel = st.selectbox("Select Ticker", options=tickers, index=0 if tickers else 0, key="sel_ticker")

    # full available years for this ticker
    years_for_ticker = (
        df_master[df_master["Ticker"].astype(str).str.upper() == ticker_sel]["Year"]
        .dropna().astype(int).sort_values().unique().tolist()
    )
    if len(years_for_ticker) == 0:
        st.error("No years available for this ticker in dataset."); st.stop()

    default_start = max(min(years_for_ticker), sorted(years_for_ticker)[-1] - 3)
    default_end   = max(years_for_ticker)

    start_year = st.number_input("Start Year", min_value=min(years_for_ticker),
                                 max_value=max(years_for_ticker), value=default_start, step=1, key="start_year")
    end_year   = st.number_input("End Year",   min_value=min(years_for_ticker),
                                 max_value=max(years_for_ticker), value=default_end, step=1, key="end_year")

    if start_year > end_year:
        st.warning("Start Year must be <= End Year.")

    run_btn = st.button("Run Forecast", type="primary", use_container_width=True, key="btn_run")

# ---------------- Load model artifacts ----------------
try:
    model, scaler, feature_list, threshold = load_model_and_assets()
except FileNotFoundError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Unable to load model artifacts: {e}"); st.stop()

# ---------------- Filtering: only user-selected subset ----------------
subset = filter_by_ticker_period(df_master, ticker_sel, int(start_year), int(end_year))

# ====== Section 1: Executive Overview (EDA on selected subset only) ======
st.subheader("Exploratory Data Overview (Selected Ticker & Period)")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        plot_default_distribution_year(subset),
        use_container_width=True,
        key="eda_year_subset",
    )
with c2:
    st.plotly_chart(
        plot_default_rate_by_sector(subset),
        use_container_width=True,
        key="eda_sector_subset",
    )

# ====== Section 2: Firm-level PD Forecast over Period ======
st.subheader("Firm-Level PD Forecast (Multi-Year)")
if run_btn:
    # Forecast for each year in [start_year, end_year]
    try:
        forecast_df = predict_period_for_ticker(
            df=df_master,
            ticker=ticker_sel,
            years=list(range(int(start_year), int(end_year)+1)),
            model=model,
            scaler=scaler,
            features=feature_list,
            threshold=threshold,
        )
    except Exception as e:
        st.error(f"Forecast failed: {e}")
    else:
        # KPI row
        latest_row = forecast_df.sort_values("Year").iloc[-1]
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Latest Year", f"{int(latest_row['Year'])}")
        with k2:
            st.metric("Predicted Default Probability (Latest)", f"{latest_row['Default_Proba']:.2%}")
        with k3:
            st.metric("Decision Threshold", f"{threshold:.2f}")

        # Charts
        cc1, cc2 = st.columns(2)
        with cc1:
            st.plotly_chart(
                plot_pd_line_forecast(forecast_df),
                use_container_width=True,
                key="pd_line_forecast",
            )
        with cc2:
            st.plotly_chart(
                plot_pd_risk_bucket_bar(forecast_df, threshold=threshold),
                use_container_width=True,
                key="pd_risk_bucket",
            )

        # Probability histogram of only the forecast horizon
        st.subheader("Predicted Probability Distribution (Forecast Horizon)")
        st.plotly_chart(
            plot_probability_histogram(forecast_df.rename(columns={"Default_Proba": "Default_Proba"})),
            use_container_width=True,
            key="pd_hist_forecast",
        )
else:
    st.info("Set Ticker and Year range on the left, then click **Run Forecast**.")

# ====== Section 3: Model Evaluation (optional, whole dataset if labeled) ======
if "Default" in df_master.columns:
    st.subheader("Model Evaluation (Full Labeled Dataset)")
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
            st.plotly_chart(
                plot_roc_auc_plotly(model, X_eval_scaled, y_true),
                use_container_width=True,
                key="eval_roc_full",
            )
        with e2:
            st.plotly_chart(
                plot_precision_recall_plotly(model, X_eval_scaled, y_true),
                use_container_width=True,
                key="eval_pr_full",
            )
    except Exception as e:
        st.info(f"Evaluation skipped: {e}")
else:
    st.info("Column 'Default' not found in dataset. Skipping evaluation charts.")
