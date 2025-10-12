import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---- Local modules (giữ nguyên theo cấu trúc của bạn) ----
from modules.data_utils import load_master_data, label_default_from_financials, preprocess_and_create_features
# Nếu bạn đang dùng tách file feature_engineering.py cho interactions, vẫn giữ import trên data_utils là đủ.


# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Corporate Default Risk Scoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Corporate Default Risk Scoring System")
st.markdown(
    """
This internal dashboard provides **institutional-grade default risk forecasting** for corporates,
using a **pre-trained LightGBM model** with monotonic constraints and calibrated financial features.
"""
)


# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def load_artifacts():
    """Load model, features, scaler, threshold from artifacts/ safely."""
    art_dir = Path("artifacts")
    errors = []

    # model
    model = None
    model_path = art_dir / "lgbm_model.pkl"
    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as e:
            errors.append(f"Failed to load model: {e}")
    else:
        errors.append("lgbm_model.pkl not found in artifacts/")

    # features
    features = None
    feat_path = art_dir / "features.pkl"
    if feat_path.exists():
        try:
            features = joblib.load(feat_path)
            if not isinstance(features, (list, tuple)):
                errors.append("features.pkl must contain a Python list of feature names.")
                features = None
        except Exception as e:
            errors.append(f"Failed to load features.pkl: {e}")
    else:
        errors.append("features.pkl not found in artifacts/")

    # scaler (optional)
    scaler = None
    scaler_path = art_dir / "scaler.pkl"
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            errors.append(f"Failed to load scaler.pkl: {e}")

    # threshold
    threshold = None
    th_path = art_dir / "threshold.json"
    if th_path.exists():
        try:
            with open(th_path, "r") as f:
                th = json.load(f)
            # accept either "threshold" or "best_threshold"
            threshold = float(th.get("threshold", th.get("best_threshold", 0.5)))
        except Exception as e:
            errors.append(f"Failed to parse threshold.json: {e}")
            threshold = 0.5
    else:
        errors.append("threshold.json not found in artifacts/. Fallback to 0.5.")
        threshold = 0.5

    return model, features, scaler, threshold, errors


def align_features_for_inference(df: pd.DataFrame, features: list, scaler):
    """
    - Bảo đảm có đủ cột feature (thiếu → thêm 0.0)
    - Sắp xếp đúng thứ tự theo features.pkl
    - Scale numeric (nếu scaler có) — giữ nguyên cột nhị phân phổ biến
    """
    X = df.copy()

    # Ensure all feature columns exist
    for c in features:
        if c not in X.columns:
            X[c] = 0.0

    # Order columns as features list
    X = X[features]

    # Apply scaler if available
    if scaler is not None:
        X_scaled = X.copy()
        # nếu bạn đã dùng các binary trong training, có thể chỉ định:
        bin_cols = {"LowRiskFlag", "OCF_Deficit_2of3"}
        num_cols = [c for c in X.columns if c not in bin_cols]
        try:
            X_scaled[num_cols] = scaler.transform(X[num_cols])
            return X_scaled
        except Exception:
            # nếu scaler lỗi do shape mismatch → dùng X nguyên bản để không chặn flow
            return X
    return X


@st.cache_data(show_spinner=False)
def prepare_dataframe():
    """
    Load -> Label realistic default -> Feature engineering (bao gồm interactions do bạn đã xử lý trong modules).
    """
    df_raw = load_master_data()  # bạn đã trỏ về ./bctc_final.csv trong data_utils
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_labeled = label_default_from_financials(df_raw)
    df_feat = preprocess_and_create_features(df_labeled)
    # enforce basic types
    if "Year" in df_feat.columns:
        df_feat["Year"] = df_feat["Year"].astype(int, errors="ignore")
    if "Ticker" in df_feat.columns:
        df_feat["Ticker"] = df_feat["Ticker"].astype(str).str.upper()
    return df_feat


# =========================
# Load data & artifacts
# =========================
df_master = prepare_dataframe()
if df_master.empty:
    st.error("No dataset found or dataset is empty. Make sure `bctc_final.csv` is present and readable.")
    st.stop()

required_cols = {"Ticker", "Year"}
missing_cols = required_cols - set(df_master.columns)
if missing_cols:
    st.error(f"Dataset missing required columns: {sorted(list(missing_cols))}")
    st.stop()

model, feature_list, scaler, threshold, art_errors = load_artifacts()
if art_errors:
    st.warning(" | ".join(art_errors))
if (model is None) or (feature_list is None):
    st.error("Model or feature list not available. Please check artifacts folder.")
    st.stop()


# =========================
# Sidebar (Ticker & Period)
# =========================
st.sidebar.header("Input Panel")

tickers = sorted(df_master["Ticker"].unique().tolist())
ticker_sel = st.sidebar.selectbox("Select Ticker", options=tickers, index=0)

years_all = sorted(df_master["Year"].unique().tolist())
default_range = (min(years_all), max(years_all))
start_year, end_year = st.sidebar.select_slider(
    "Select Year Range",
    options=years_all,
    value=default_range
)

run_btn = st.sidebar.button("Run Forecast", type="primary", use_container_width=True)


# =========================
# Subset theo lựa chọn
# =========================
subset = df_master[
    (df_master["Ticker"] == ticker_sel) &
    (df_master["Year"].between(start_year, end_year))
].sort_values("Year")

if subset.empty:
    st.warning(f"No data for {ticker_sel} between {start_year}–{end_year}.")
    st.stop()


# =========================
# Overview KPIs
# =========================
st.subheader(f"Overview — {ticker_sel} ({start_year}–{end_year})")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Years in Selection", f"{subset['Year'].nunique()}")
with c2:
    st.metric("Any Default Label in Selection",
              "Yes" if "Default" in subset.columns and subset["Default"].max() == 1 else "No")
with c3:
    sector_value = subset["Sector"].iloc[0] if "Sector" in subset.columns else "N/A"
    st.metric("Sector", sector_value)


# =========================
# EDA — Financial & Ratios
# =========================
st.subheader("Exploratory Data Analysis")

def plot_financial_trends(df_):
    cols_present = [c for c in ["Total_Assets", "Equity", "Revenue", "Net_Profit"] if c in df_.columns]
    if not cols_present:
        return None
    fig = px.line(
        df_, x="Year", y=cols_present, markers=True,
        title="Financial Trends", template="plotly_white"
    )
    fig.update_layout(legend_title_text="Metric", xaxis=dict(dtick=1))
    return fig

def plot_ratio_trends(df_):
    cols_present = [c for c in ["Debt_to_Assets", "ROA", "ROE"] if c in df_.columns]
    if not cols_present:
        return None
    fig = px.line(
        df_, x="Year", y=cols_present, markers=True,
        title="Leverage & Profitability Ratios", template="plotly_white"
    )
    fig.update_layout(legend_title_text="Ratio", xaxis=dict(dtick=1))
    return fig

fig1 = plot_financial_trends(subset)
if fig1: st.plotly_chart(fig1, use_container_width=True)

fig2 = plot_ratio_trends(subset)
if fig2: st.plotly_chart(fig2, use_container_width=True)


# =========================
# PD Forecast (multi-year)
# =========================
st.subheader("Default Probability Forecast (Multi-Year)")

if run_btn:
    # Align features & predict for selected period
    X_subset = align_features_for_inference(subset, feature_list, scaler)
    try:
        y_proba = model.predict_proba(X_subset)[:, 1]
    except Exception:
        # LightGBM Booster vs sklearn API fallback
        try:
            y_proba = model.predict(X_subset)
            if y_proba.ndim == 1:
                # assume already probas
                pass
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

    pred_df = subset[["Ticker", "Year"]].copy()
    pred_df["Default_Proba"] = y_proba
    pred_df = pred_df.sort_values("Year")

    # KPIs
    latest_row = pred_df.iloc[-1]
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Latest Year", f"{int(latest_row['Year'])}")
    with k2:
        st.metric("Predicted Default Probability (Latest)", f"{latest_row['Default_Proba']:.2%}")
    with k3:
        st.metric("Decision Threshold", f"{threshold:.2f}")

    # Charts
    fig_pd = px.line(
        pred_df, x="Year", y="Default_Proba", markers=True,
        title=f"PD Forecast by Year — {ticker_sel}", template="plotly_white"
    )
    fig_pd.update_yaxes(tickformat=".0%", range=[0, 1])
    # add threshold line
    fig_pd.add_hline(y=threshold, line_dash="dash", line_color="#e63946",
                     annotation_text=f"Threshold {threshold:.2f}")
    st.plotly_chart(fig_pd, use_container_width=True)

    # Risk bucket bar
    def bucket(p):
        if p >= threshold: return "High Risk"
        if p >= max(0.5*threshold, 0.15): return "Medium Risk"
        return "Low Risk"
    pred_df["Risk_Bucket"] = pred_df["Default_Proba"].apply(bucket)

    fig_bucket = px.bar(
        pred_df, x="Year", y="Default_Proba", color="Risk_Bucket",
        title="Risk Classification by Year", template="plotly_white"
    )
    fig_bucket.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(fig_bucket, use_container_width=True)

    # Histogram
    hist = px.histogram(
        pred_df, x="Default_Proba", nbins=20, title="Predicted Default Probability Distribution",
        template="plotly_white"
    )
    hist.update_xaxes(range=[0, 1])
    st.plotly_chart(hist, use_container_width=True)
else:
    st.info("Set Ticker and Year range on the left, then click **Run Forecast**.")


# =========================
# Sector Benchmark
# =========================
st.subheader("Sector Benchmark Risk")

if "Sector" in df_master.columns and "Default" in df_master.columns:
    sector_summary = (
        df_master.groupby("Sector")["Default"]
        .mean()
        .reset_index()
        .rename(columns={"Default": "Default Rate"})
        .sort_values("Default Rate", ascending=False)
    )
    fig_sector = px.bar(
        sector_summary, x="Sector", y="Default Rate",
        title="Default Rate by Sector (Full Dataset)", template="plotly_white"
    )
    fig_sector.update_layout(xaxis_tickangle=-40, yaxis_tickformat=".0%")
    st.plotly_chart(fig_sector, use_container_width=True)
else:
    st.info("Sector or Default column not found. Sector benchmark is skipped.")
