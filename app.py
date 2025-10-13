# app.py  —  Single-page, bank-grade UI
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==== Utils (theo repo của bạn) ====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import (
    preprocess_and_create_features,
    default_financial_feature_list,
)
from utils.feature_selection import select_features_for_model
from utils.model_scoring import (
    load_lgbm_model,
    model_feature_names,
    predict_pd,
    explain_shap,
    run_stress_test,
)
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd
from utils.drift_monitoring import compute_psi_table
from utils.visualization import (
    default_distribution_by_year,
    default_distribution_by_sector,
)

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Corporate Default Risk Scoring Portal",
    layout="wide",
    page_icon="🧮",
)

# ====== STYLES (nhẹ, không cần asset riêng) ======
st.markdown(
    """
    <style>
      .block-container {padding-top:1.5rem; padding-bottom:2rem;}
      .metric {text-align:center}
      .stExpander {border-radius:10px; border:1px solid #E6E8EB;}
      .stButton>button {border-radius:8px;}
      .badge {display:inline-block; padding:4px 8px; border-radius:8px; background:#F2F4F7; font-size:12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====== HELPERS UI ======
def gauge_pd(pd_value: float) -> go.Figure:
    """Plotly gauge cho PD%."""
    v = float(pd_value) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={'suffix':"%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#1f77b4'},
                'steps': [
                    {'range': [0, 10], 'color': '#E8F1FB'},
                    {'range': [10, 30], 'color': '#CFE3F7'},
                    {'range': [30, 100], 'color': '#F9E3E3'},
                ],
                'threshold': {'line': {'color': 'red', 'width': 3}, 'thickness': 0.8, 'value': v}
            },
            domain={'x': [0,1], 'y':[0,1]}
        )
    )
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def show_policy_badge(th):
    st.markdown(
        f"<span class='badge'>Policy: Low &lt; {th['low']:.0%} • Medium &lt; {th['medium']:.0%}</span>",
        unsafe_allow_html=True,
    )

# =====================================================================================
# SIDEBAR — INPUT PANEL (tất cả nhập liệu & cấu hình đều ở đây)
# =====================================================================================
st.sidebar.title("Artifacts & Inputs")

# (1) Model & Policy files
model_path = st.sidebar.text_input("LightGBM model (.pkl)", value="models/lgbm_model.pkl")
threshold_path = st.sidebar.text_input("threshold.json", value="models/threshold.json")

# (2) Data source
st.sidebar.subheader("Data")
data_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
use_demo = st.sidebar.checkbox("Hoặc dùng bctc_final.csv trong repo", value=True)

# (3) Selections (sau khi có data)
st.sidebar.subheader("Scoring Parameters")
ticker_sel = st.sidebar.text_input("Ticker (gõ mã để lọc)", value="")
year_sel = st.sidebar.text_input("Year (ví dụ 2023)", value="")
sector_override = st.sidebar.text_input("Override Sector (để trống nếu dùng sector tự động)", value="")

# (4) Stress Test
st.sidebar.subheader("Stress Test (optional)")
enable_stress = st.sidebar.checkbox("Bật Stress Test", value=False)
stress_magnitude = st.sidebar.slider("Shock magnitude (%)", -50, 50, 10, 5)

# (5) Drift Monitoring
st.sidebar.subheader("Drift Monitoring")
enable_psi = st.sidebar.checkbox("Bật tính PSI (cần models/train_reference.*)", value=True)

# (6) RUN buttons
run_prepare = st.sidebar.button("1) Build features")
run_score = st.sidebar.button("2) Score & Explain")
run_stress = st.sidebar.button("3) Run Stress Test", disabled=not enable_stress)
run_drift = st.sidebar.button("4) Compute PSI", disabled=not enable_psi)

# =====================================================================================
# MAIN — OUTPUTS ONLY
# =====================================================================================

st.title("Corporate Default Risk Scoring Portal")
st.caption("Single-page • Sidebar for inputs • Main page for results • LightGBM scoring • Policy by sector • SHAP • Stress test • Drift/PSI")

# -------------------------------------------------------------------------------------
# STEP 1 — LOAD DATA & BUILD FEATURES
# -------------------------------------------------------------------------------------
if run_prepare:
    try:
        if data_file is not None:
            raw = pd.read_csv(data_file)
        elif use_demo and os.path.exists("bctc_final.csv"):
            raw = pd.read_csv("bctc_final.csv")
        else:
            st.error("❌ Không có dữ liệu. Hãy upload CSV hoặc bật 'dùng bctc_final.csv'.")
            raw = None

        if raw is not None:
            st.subheader("1) Data Overview")
            st.write("Raw preview", raw.head(5))

            # Cleaning → Feature engineering
            cleaned = clean_and_log_transform(raw)
            features_df = preprocess_and_create_features(cleaned)

            st.success(f"✅ Prepared dataset: {features_df.shape[0]} rows × {features_df.shape[1]} cols")
            st.dataframe(features_df.head(10), use_container_width=True)

            # Save to session
            st.session_state["features_df"] = features_df

            # Overview charts
            c1, c2 = st.columns(2)
            with c1:
                p1, b1, t1 = default_distribution_by_year(features_df)
                st.plotly_chart(p1, use_container_width=True)
                st.plotly_chart(b1, use_container_width=True)
            with c2:
                b2, p2, b3, t2 = default_distribution_by_sector(features_df)
                st.plotly_chart(b2, use_container_width=True)
                st.plotly_chart(p2, use_container_width=True)

    except Exception as e:
        st.exception(e)

# -------------------------------------------------------------------------------------
# STEP 2 — SCORE (PD) + POLICY BAND + SHAP
# -------------------------------------------------------------------------------------
if run_score:
    if "features_df" not in st.session_state:
        st.error("❌ Chưa có features. Vui lòng chạy '1) Build features' trước.")
    else:
        features_df = st.session_state["features_df"]
        # Detect ticker/year from user inputs
        df_tick = features_df["Ticker"].astype(str)
        tickers = sorted(df_tick.unique().tolist())
        # Gợi ý ticker nếu user nhập 1 phần
        chosen_ticker = None
        if ticker_sel:
            matches = [t for t in tickers if ticker_sel.upper() in str(t).upper()]
            if len(matches)>0:
                chosen_ticker = matches[0]
        if chosen_ticker is None and len(tickers)>0:
            chosen_ticker = tickers[0]

        # Year
        if year_sel:
            try:
                chosen_year = int(year_sel)
            except:
                chosen_year = None
        else:
            # mặc định lấy năm lớn nhất của ticker
            if chosen_ticker:
                years = features_df.loc[df_tick==chosen_ticker, "Year"].dropna().astype(int)
                chosen_year = int(years.max()) if not years.empty else None
            else:
                chosen_year = None

        # Sector detect
        sector_values = features_df.loc[
            (df_tick==chosen_ticker) & (features_df["Year"]==chosen_year),
            "Sector"
        ].astype(str).unique()
        detected_sector = sector_values[0] if len(sector_values)>0 else ""
        sector_used = sector_override.strip() or detected_sector

        # UI summary
        st.subheader("2) Company Selection")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker", chosen_ticker if chosen_ticker else "-")
        c2.metric("Year", chosen_year if chosen_year else "-")
        c3.metric("Sector", sector_used if sector_used else "-")
        c4.metric("Rows", f"{features_df.shape[0]}")

        # Load model & thresholds
        try:
            model = load_lgbm_model(model_path)
        except Exception as e:
            st.error(f"Không load được model: {e}")
            model = None

        thresholds = load_thresholds(threshold_path)

        # Build feature vector theo model
        candidate_features = default_financial_feature_list()
        model_feats = model_feature_names(model) if model is not None else None
        final_features = select_features_for_model(features_df, candidate_features, model_feats)

        if model is None or len(final_features)==0:
            st.error("❌ Model hoặc feature set không hợp lệ.")
        else:
            # Lấy đúng dòng dữ liệu
            row = features_df[(features_df["Ticker"].astype(str)==str(chosen_ticker)) & (features_df["Year"]==chosen_year)]
            if row.empty:
                st.error("❌ Không tìm thấy dữ liệu cho Ticker/Year đã chọn.")
            else:
                x = row.iloc[0]
                X_df = pd.DataFrame([x[final_features].values], columns=final_features)
                X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

                # PD & Policy
                pd_value = predict_pd(model, X_df)
                th = thresholds_for_sector(thresholds, sector_used)
                band = classify_pd(pd_value, th)

                # Hiển thị kết quả
                st.subheader("3) PD Scoring & Policy Band")
                c1, c2, c3 = st.columns([1,1,2])
                with c1:
                    st.metric("PD", f"{pd_value:.2%}")
                with c2:
                    st.metric("Risk Band", band)
                with c3:
                    show_policy_badge(th)
                st.plotly_chart(gauge_pd(pd_value), use_container_width=True)

                # Lưu session cho các bước sau
                st.session_state["chosen_ticker"] = chosen_ticker
                st.session_state["chosen_year"] = chosen_year
                st.session_state["sector_used"] = sector_used
                st.session_state["final_features"] = final_features
                st.session_state["model"] = model
                st.session_state["pd_value"] = float(pd_value)
                st.session_state["X_row"] = X_df
                st.session_state["raw_row"] = x

                # SHAP
                st.subheader("4) SHAP — Top Drivers")
                shap_df = explain_shap(model, X_df, top_n=10)
                if shap_df.empty:
                    st.info("SHAP không khả dụng cho mô hình hiện tại.")
                else:
                    st.dataframe(shap_df, use_container_width=True)

# -------------------------------------------------------------------------------------
# STEP 3 — STRESS TEST
# -------------------------------------------------------------------------------------
if run_stress:
    need_keys = ["raw_row", "final_features", "model"]
    if not all(k in st.session_state for k in need_keys):
        st.error("❌ Hãy chạy '2) Score & Explain' trước khi stress test.")
    else:
        st.subheader("5) Stress Testing")
        base_row = st.session_state["raw_row"]
        feats = st.session_state["final_features"]
        model = st.session_state["model"]

        # Mặc định chọn 3 biến đầu tiên numeric
        numeric_feats = [f for f in feats if isinstance(base_row.get(f, 0.0), (int, float, np.floating))]
        default_pick = numeric_feats[:3]
        st.write("💡 Gợi ý biến có thể shock:", default_pick)

        # Shock toàn bộ biến đã chọn cùng một biên độ
        shocks = {f: stress_magnitude / 100.0 for f in default_pick}
        out = run_stress_test(model, base_row, feats, shocks)

        # Bảng kết quả
        st.dataframe(out[["Scenario", "PD", "Delta_PD"] + default_pick], use_container_width=True)

        # Biểu đồ ΔPD
        fig = go.Figure()
        base_pd = float(out.loc[out["Scenario"]=="Base","PD"].values[0])
        for _, r in out.iterrows():
            if r["Scenario"] == "Base": 
                continue
            fig.add_trace(go.Bar(name=r["Scenario"], x=["ΔPD vs Base"], y=[r["PD"]-base_pd]))
        fig.update_layout(barmode="group", title="ΔPD by scenario", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------------------
# STEP 4 — DRIFT / PSI
# -------------------------------------------------------------------------------------
if run_drift:
    if "features_df" not in st.session_state:
        st.error("❌ Chưa có features. Hãy chạy '1) Build features'.")
    else:
        st.subheader("6) Drift Monitoring (PSI)")
        cur = st.session_state["features_df"]
        feats = st.session_state.get("final_features", default_financial_feature_list())

        # Tìm train_reference.* trong models/
        ref = None
        for path in ["models/train_reference.parquet", "models/train_reference.csv"]:
            if os.path.exists(path):
                try:
                    ref = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
                    break
                except Exception:
                    pass

        if ref is None:
            st.warning("Không tìm thấy models/train_reference.parquet|csv → không tính được PSI.")
        else:
            common = [f for f in feats if f in cur.columns and f in ref.columns]
            cur_df = cur[common].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            ref_df = ref[common].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            psi_table = compute_psi_table(ref_df, cur_df, common, buckets=10)
            st.dataframe(psi_table, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Stable", int((psi_table["status"]=="Stable").sum()))
            c2.metric("Moderate", int((psi_table["status"]=="Moderate").sum()))
            c3.metric("Shift", int((psi_table["status"]=="Shift").sum()))

# Footer
st.markdown("---")
st.caption("© Corporate Risk Analytics • Single-page portal • Sidebar input → Main output • LightGBM only")
