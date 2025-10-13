import streamlit as st
import pandas as pd
import numpy as np
import os, json

from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import preprocess_and_create_features, default_financial_feature_list
from utils.feature_selection import select_features_for_model
from utils.model_scoring import load_lgbm_model, model_feature_names, predict_pd, explain_shap, run_stress_test
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd
from utils.drift_monitoring import compute_psi_table
from utils.visualization import default_distribution_by_year, default_distribution_by_sector

st.set_page_config(page_title="Corporate Default Risk Scoring Portal", layout="wide")
st.markdown("<h2 style='margin-bottom:0'>Corporate Default Risk Scoring Portal</h2>", unsafe_allow_html=True)
st.caption("Single-page, bank-grade • LightGBM scoring • SHAP explainability • Stress testing • Drift monitoring")

st.sidebar.header("Artifacts")
model_path = st.sidebar.text_input("LightGBM model (.pkl)", value="models/lgbm_model.pkl")
threshold_path = st.sidebar.text_input("threshold.json", value="models/threshold.json")
constraints_path = st.sidebar.text_input("constraints.json (optional)", value="models/constraints.json")

# 1) Upload & features
with st.expander("1) Upload data & build features", expanded=True):
    up = st.file_uploader("Upload CSV dataset", type=["csv"])
    if up is not None:
        raw = pd.read_csv(up)
        st.write("Raw preview:", raw.head(5))

        cleaned = clean_and_log_transform(raw)
        features_df = preprocess_and_create_features(cleaned)
        st.success(f"Prepared dataset: {features_df.shape[0]} rows × {features_df.shape[1]} cols")
        st.dataframe(features_df.head(10))

        c1, c2 = st.columns(2)
        with c1:
            p1, b1, t1 = default_distribution_by_year(features_df)
            st.plotly_chart(p1, use_container_width=True)
            st.plotly_chart(b1, use_container_width=True)
        with c2:
            b2, p2, b3, t2 = default_distribution_by_sector(features_df)
            st.plotly_chart(b2, use_container_width=True)
            st.plotly_chart(p2, use_container_width=True)

        st.session_state["features_df"] = features_df
    else:
        st.info("Upload your CSV to proceed.")

# 2) Select Ticker & Year
with st.expander("2) Select Ticker & Year", expanded=True):
    if "features_df" not in st.session_state:
        st.warning("Please upload data above.")
    else:
        df = st.session_state["features_df"]
        tickers = sorted(df["Ticker"].dropna().unique().tolist())
        ticker = st.selectbox("Ticker", tickers, index=0 if tickers else None)
        sector_detected = df.loc[df["Ticker"]==ticker, "Sector"].dropna().astype(str).unique()
        sector = sector_detected[0] if len(sector_detected)>0 else ""
        st.info(f"Detected Sector: **{sector or 'Unknown'}**")
        sector = st.text_input("Override Sector (optional)", value=sector)
        years = sorted(df.loc[df["Ticker"]==ticker, "Year"].dropna().unique().tolist())
        year = st.selectbox("Year", years, index=len(years)-1 if years else 0)
        st.session_state["sel"] = {"ticker": ticker, "sector": sector, "year": year}

# 3) Model & Policy
with st.expander("3) Model & Policy", expanded=True):
    if "features_df" not in st.session_state or "sel" not in st.session_state:
        st.warning("Please complete steps (1) and (2).")
    else:
        try:
            model = load_lgbm_model(model_path)
            st.success("LightGBM model loaded.")
        except Exception as e:
            model = None
            st.error(f"Cannot load model: {e}")

        thresholds = load_thresholds(threshold_path)
        st.write("Active thresholds (sample):", {k: thresholds[k] for i,k in enumerate(thresholds.keys()) if i<6})

        constraints = None
        if os.path.exists(constraints_path):
            try:
                with open(constraints_path, "r", encoding="utf-8") as f:
                    constraints = json.load(f)
                st.caption("Monotonic constraints loaded.")
            except Exception:
                constraints = None

        df = st.session_state["features_df"]
        candidate_features = default_financial_feature_list()
        model_feats = model_feature_names(model) if model is not None else None
        final_features = select_features_for_model(df, candidate_features, model_feats)
        st.write(f"Features used for scoring ({len(final_features)}):", final_features[:20], "..." if len(final_features)>20 else "")
        st.session_state["model"] = model
        st.session_state["thresholds"] = thresholds
        st.session_state["constraints"] = constraints
        st.session_state["final_features"] = final_features

# 4) PD Scoring & Policy Band
with st.expander("4) PD Scoring & Policy Band", expanded=True):
    state_ok = all(k in st.session_state for k in ["features_df","sel","model","thresholds","final_features"]) and st.session_state["model"] is not None
    if not state_ok:
        st.warning("Please ensure model & features are loaded in (3).")
    else:
        df = st.session_state["features_df"]
        sel = st.session_state["sel"]
        feats = st.session_state["final_features"]
        model = st.session_state["model"]
        th_all = st.session_state["thresholds"]

        row = df[(df["Ticker"]==sel["ticker"]) & (df["Year"]==sel["year"])]
        if row.empty:
            st.error("No data for selected Ticker & Year.")
        else:
            x = row.iloc[0]
            X_df = pd.DataFrame([x[feats].values], columns=feats).replace([np.inf,-np.inf], np.nan).fillna(0.0)

            pd_value = predict_pd(model, X_df)
            th = thresholds_for_sector(th_all, sel["sector"])
            band = classify_pd(pd_value, th)

            c1, c2, c3 = st.columns(3)
            c1.metric("PD", f"{pd_value:.2%}")
            c2.metric("Risk Band", band)
            c3.metric("Policy", f"Low<{th['low']:.0%} • Med<{th['medium']:.0%}")

            st.session_state["row_selected"] = x
            st.session_state["pd_value"] = float(pd_value)
            st.session_state["policy_th"] = th
            st.success("Scoring complete.")

# 5) SHAP
with st.expander("5) SHAP Explainability — Top Drivers", expanded=True):
    ready = all(k in st.session_state for k in ["row_selected","final_features","model"])
    if not ready:
        st.info("Score first to enable SHAP.")
    else:
        x = st.session_state["row_selected"]
        feats = st.session_state["final_features"]
        model = st.session_state["model"]
        X_df = pd.DataFrame([x[feats].values], columns=feats).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        top = explain_shap(model, X_df, top_n=10)
        if top.empty:
            st.warning("SHAP explanation unavailable.")
        else:
            st.dataframe(top)

# 6) Stress Testing
with st.expander("6) Stress Testing", expanded=True):
    ready = all(k in st.session_state for k in ["row_selected","final_features","model","pd_value"])
    if not ready:
        st.info("Score first to enable stress testing.")
    else:
        x = st.session_state["row_selected"]
        feats = st.session_state["final_features"]
        model = st.session_state["model"]

        numeric_feats = [f for f in feats if isinstance(x.get(f,0.0), (int,float))]
        pick = st.multiselect("Select variables to shock (≤5)", numeric_feats, default=numeric_feats[:3])
        magnitude = st.slider("Shock magnitude (%)", -50, 50, 10, 5)
        shocks = {f: magnitude/100.0 for f in pick}

        if st.button("Run stress test"):
            out = run_stress_test(model, x, feats, shocks)
            st.dataframe(out[["Scenario","PD","Delta_PD"] + pick])

# 7) Drift Monitoring
with st.expander("7) Drift Monitoring (PSI)", expanded=True):
    if "features_df" not in st.session_state:
        st.info("Upload data to compute PSI.")
    else:
        df = st.session_state["features_df"]
        feats = st.session_state.get("final_features", default_financial_feature_list())

        ref_candidates = ["models/train_reference.parquet", "models/train_reference.csv"]
        ref = None
        for path in ref_candidates:
            if os.path.exists(path):
                try:
                    if path.endswith(".parquet"):
                        ref = pd.read_parquet(path)
                    else:
                        ref = pd.read_csv(path)
                    break
                except Exception:
                    ref = None
        if ref is None:
            st.warning("No training reference found. Place models/train_reference.parquet or .csv to enable PSI.")
        else:
            common = [f for f in feats if f in df.columns and f in ref.columns]
            score_df = df[common].replace([np.inf,-np.inf], np.nan).fillna(0.0)
            ref_df = ref[common].replace([np.inf,-np.inf], np.nan).fillna(0.0)
            psi_table = compute_psi_table(ref_df, score_df, common, buckets=10)
            st.dataframe(psi_table.head(50))
            stable = int((psi_table["status"]=="Stable").sum())
            moderate = int((psi_table["status"]=="Moderate").sum())
            shift = int((psi_table["status"]=="Shift").sum())
            c1,c2,c3 = st.columns(3)
            c1.metric("Stable", stable); c2.metric("Moderate", moderate); c3.metric("Shift", shift)

st.caption("© Corporate Risk Analytics — Single-page Portal")