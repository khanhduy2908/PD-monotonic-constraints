import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ===== Utils from your repo =====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import preprocess_and_create_features, default_financial_feature_list
from utils.feature_selection import select_features_for_model
from utils.model_scoring import load_lgbm_model, model_feature_names, explain_shap
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd
from utils.drift_monitoring import compute_psi_table
from utils.visualization import default_distribution_by_year, default_distribution_by_sector

# ===================== Page config & styling =====================
st.set_page_config(page_title="Corporate Default Risk Scoring Portal", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.0rem; padding-bottom:2rem;}
h1,h2,h3 {font-weight:600;}
.badge {display:inline-block; padding:4px 8px; border-radius:8px; background:#F2F4F7; font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ===================== Helpers =====================
def _safe_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _force_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return _safe_df(X)

def build_X_for_model(row: pd.Series, model, fallback_features: list) -> pd.DataFrame:
    """
    Align features to model's expected order; add missing with 0.0; drop extras.
    This prevents LightGBMError (validate_features) on Streamlit Cloud.
    """
    m_feats = model_feature_names(model)
    if not m_feats or len(m_feats) == 0:
        # fall back to provided list
        feats = fallback_features
    else:
        feats = list(m_feats)

    # add missing columns with 0.0
    data = {}
    for f in feats:
        data[f] = float(row.get(f, 0.0))
    X = pd.DataFrame([data], columns=feats)
    return _force_numeric(X)

def gauge_pd(pd_value: float) -> go.Figure:
    v = float(pd_value) * 100
    fig = go.Figure(go.Indicator(
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
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def policy_badge(th):
    st.markdown(
        f"<span class='badge'>Policy: Low &lt; {th['low']:.0%} • Medium &lt; {th['medium']:.0%}</span>",
        unsafe_allow_html=True
    )

def load_train_reference():
    for p in ["models/train_reference.parquet", "models/train_reference.csv"]:
        if os.path.exists(p):
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                pass
    return None

def compute_feature_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    # chỉ giữ những feature có thật trong df để tránh KeyError
    valid_feats = [f for f in features if f in df.columns]
    if not valid_feats:
        return pd.DataFrame(columns=["mean", "std"])
    sub = df[valid_feats].replace([np.inf, -np.inf], np.nan)
    stats = pd.DataFrame({"mean": sub.mean(skipna=True), "std": sub.std(ddof=0, skipna=True)})
    stats["std"] = stats["std"].replace(0, np.nan)
    return stats

# Feature groups to determine direction of risk in shocks
RISK_UP_FEATURES = {
    "Debt_to_Assets", "Debt_to_Equity", "Total_Debt_to_EBITDA", "Net_Debt_to_Equity", "Long_Term_Debt_to_Assets"
}
RISK_DOWN_FEATURES = {
    "ROE", "ROA", "Current_Ratio", "Quick_Ratio", "Interest_Coverage", "EBITDA_to_Interest", "Operating_Income_to_Debt"
}

def apply_market_shock_vector(base_vec: np.ndarray, features: list, stats: pd.DataFrame, k_sigma: float) -> np.ndarray:
    shocked = base_vec.copy()
    for i, f in enumerate(features):
        if f not in stats.index:
            continue
        s = stats.loc[f, "std"]
        if not np.isfinite(s) or s == 0:
            continue
        if f in RISK_UP_FEATURES:
            shocked[i] = shocked[i] + k_sigma * float(s)
        elif f in RISK_DOWN_FEATURES:
            shocked[i] = shocked[i] - k_sigma * float(s)
    return shocked

def monte_carlo_cvar_pd(model, base_vec: np.ndarray, features: list, stats: pd.DataFrame, sims: int = 2000, alpha: float = 0.95) -> dict:
    sigmas = np.array([stats.loc[f, "std"] if (f in stats.index and np.isfinite(stats.loc[f, "std"]) and stats.loc[f,"std"]>0) else 0.0 for f in features])
    dirs = np.array([1.0 if f in RISK_UP_FEATURES else (-1.0 if f in RISK_DOWN_FEATURES else 0.0) for f in features], dtype=float)
    shocks = np.random.normal(loc=0.0, scale=sigmas, size=(sims, len(features))) * dirs
    sims_mat = base_vec.reshape(1, -1) + shocks
    X = _safe_df(pd.DataFrame(sims_mat, columns=features))
    # predict_proba in batch, using model.booster under the hood
    if hasattr(model, "predict_proba"):
        pd_sims = model.predict_proba(X)[:, 1]
    else:
        pd_sims = model.predict(X).astype(float)
    var = float(np.quantile(pd_sims, alpha))
    cvar = float(pd_sims[pd_sims >= var].mean()) if (pd_sims >= var).any() else var
    return {"VaR": var, "CVaR": cvar, "PD_sims": pd_sims}

# ===================== Data & Artifacts bootstrap =====================
@st.cache_data(show_spinner=False)
def load_prepared_dataset():
    if not os.path.exists("bctc_final.csv"):
        raise FileNotFoundError("bctc_final.csv not found in repository root.")
    raw = pd.read_csv("bctc_final.csv")
    cleaned = clean_and_log_transform(raw)
    features_df = preprocess_and_create_features(cleaned)
    return features_df

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_lgbm_model("models/lgbm_model.pkl")
    thresholds = load_thresholds("models/threshold.json")
    return model, thresholds

# ===================== Sidebar (inputs only) =====================
st.sidebar.header("Inputs")

# Load artifacts and data once
try:
    features_df = load_prepared_dataset()
except Exception as e:
    st.error(f"Dataset load error: {e}")
    st.stop()

try:
    model, thresholds = load_artifacts()
except Exception as e:
    st.error(f"Artifacts load error: {e}")
    st.stop()

# Feature list to align with model
candidate_features = default_financial_feature_list()
model_feats = model_feature_names(model)
final_features = select_features_for_model(features_df, candidate_features, model_feats)

# Ticker / Year selectors
all_tickers = sorted(features_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0 if all_tickers else None)

years_avail = sorted(features_df.loc[features_df["Ticker"].astype(str)==ticker, "Year"].dropna().astype(int).unique().tolist())
default_year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=default_year_idx)

# ===================== Main (outputs only) =====================
st.title("Corporate Default Risk Scoring Portal")
st.caption("Single-page. Sidebar inputs only. Main dashboard shows PD, policy band, SHAP, three stress scenarios side-by-side, and PSI.")

# Company selection
row_sel = features_df[(features_df["Ticker"].astype(str)==ticker) & (features_df["Year"]==year)]
if row_sel.empty:
    st.error("No matching record for selected Ticker & Year.")
    st.stop()

x = row_sel.iloc[0]
sector_detected = str(x.get("Sector", "")) if pd.notna(x.get("Sector", "")) else ""
exchange_detected = str(x.get("Exchange", "")) if pd.notna(x.get("Exchange", "")) else ""

# Build base feature vector aligned to model
X_base = build_X_for_model(x, model=model, fallback_features=final_features)
base_vec = X_base.values[0]  # numpy vector

# ========== A) Company Overview ==========
st.subheader("A. Company Overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Year", int(year))
c3.metric("Sector", sector_detected if sector_detected else "-")
c4.metric("Exchange", exchange_detected if exchange_detected else "-")

with st.expander("Sector & Year distribution (context)"):
    c5,c6 = st.columns(2)
    p1, b1, _ = default_distribution_by_year(features_df)
    b2, p2, _, _ = default_distribution_by_sector(features_df)
    with c5:
        st.plotly_chart(b1, use_container_width=True)
    with c6:
        st.plotly_chart(p2, use_container_width=True)

# ========== B) PD Scoring & Policy ==========
st.subheader("B. Default Probability (PD) & Policy Band")
# Predict PD baseline — use model.predict_proba on aligned frame (avoids LightGBMError)
if hasattr(model, "predict_proba"):
    pd_base = float(model.predict_proba(X_base)[:, 1][0])
else:
    pd_base = float(model.predict(X_base)[0])

th = thresholds_for_sector(thresholds, sector_detected)
band = classify_pd(pd_base, th)

c7,c8,c9 = st.columns([1,1,2])
with c7: st.metric("PD", f"{pd_base:.2%}")
with c8: st.metric("Policy Band", band)
with c9: policy_badge(th)
st.plotly_chart(gauge_pd(pd_base), use_container_width=True)

# ========== C) SHAP Explainability ==========
st.subheader("C. Model Explainability (SHAP)")
shap_df = explain_shap(model, X_base, top_n=10)
if shap_df.empty:
    st.info("SHAP explanation is not available for this model or input.")
else:
    st.dataframe(shap_df, use_container_width=True)

# ========== D) Stress Testing — show THREE scenarios side-by-side ==========
st.subheader("D. Stress Testing (Three scenarios side-by-side)")
# Stats for shocks
stats = compute_feature_stats(features_df, list(X_base.columns))

def scenario_market(pd_base_val, k_sigma: float, label: str):
    shocked_vec = apply_market_shock_vector(base_vec, list(X_base.columns), stats, k_sigma=k_sigma)
    Xs = _safe_df(pd.DataFrame([shocked_vec], columns=X_base.columns))
    if hasattr(model, "predict_proba"):
        pd_s = float(model.predict_proba(Xs)[:, 1][0])
    else:
        pd_s = float(model.predict(Xs)[0])
    table = pd.DataFrame([{"Scenario": label, "PD": pd_s, "ΔPD": pd_s - pd_base_val}])
    fig = go.Figure()
    fig.add_trace(go.Bar(name="PD", x=[label], y=[pd_s]))
    fig.update_layout(title=f"{label}", showlegend=False)
    return table, fig

def scenario_mc_cvar(pd_base_val, label="Monte Carlo CVaR 95%"):
    mc = monte_carlo_cvar_pd(model, base_vec, list(X_base.columns), stats, sims=2000, alpha=0.95)
    table = pd.DataFrame([
        {"Scenario": label, "PD": mc["CVaR"], "ΔPD": mc["CVaR"] - pd_base_val},
        {"Scenario": "VaR 95%", "PD": mc["VaR"], "ΔPD": mc["VaR"] - pd_base_val},
    ])
    hist = np.histogram(mc["PD_sims"], bins=30)
    fig = go.Figure()
    centers = (hist[1][1:] + hist[1][:-1]) / 2
    fig.add_trace(go.Bar(x=centers, y=hist[0]))
    fig.add_vline(x=mc["VaR"], line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=mc["CVaR"], line_width=2, line_dash="dot", line_color="black")
    fig.update_layout(title=label, xaxis_title="PD", yaxis_title="Frequency")
    return table, fig

cA, cB, cC = st.columns(3)
tA, fA = scenario_market(pd_base, k_sigma=1.0, label="Market Shock (1σ)")
tB, fB = scenario_market(pd_base, k_sigma=2.0, label="Systemic Crisis (2σ)")
tC, fC = scenario_mc_cvar(pd_base, label="Monte Carlo CVaR 95%")

with cA:
    st.table(tA.style.format({"PD":"{:.2%}","ΔPD":"{:+.2%}"}))
    st.plotly_chart(fA, use_container_width=True)
with cB:
    st.table(tB.style.format({"PD":"{:.2%}","ΔPD":"{:+.2%}"}))
    st.plotly_chart(fB, use_container_width=True)
with cC:
    st.table(tC.style.format({"PD":"{:.2%}","ΔPD":"{:+.2%}"}))
    st.plotly_chart(fC, use_container_width=True)

# ========== E) Drift Monitoring (PSI) — always on if reference exists ==========
st.subheader("E. Drift Monitoring (PSI)")
ref = load_train_reference()
if ref is None:
    st.info("Training reference not found. Place models/train_reference.parquet or .csv to enable PSI.")
else:
    # Align columns to model features order; then compute PSI on common features
    common = [f for f in X_base.columns if f in features_df.columns and f in ref.columns]
    score_df = _safe_df(features_df[common])
    ref_df = _safe_df(ref[common])
    psi_table = compute_psi_table(ref_df, score_df, common, buckets=10)
    st.dataframe(psi_table, use_container_width=True)
    s1 = int((psi_table["status"]=="Stable").sum())
    s2 = int((psi_table["status"]=="Moderate").sum())
    s3 = int((psi_table["status"]=="Shift").sum())
    d1,d2,d3 = st.columns(3)
    d1.metric("Stable", s1); d2.metric("Moderate", s2); d3.metric("Shift", s3)

# ========== F) Executive Summary ==========
st.subheader("F. Executive Summary")
summary_lines = [
    f"Ticker {ticker}, Year {year}, Sector {sector_detected or '-'}, Exchange {exchange_detected or '-'}.",
    f"Baseline PD {pd_base:.2%}, policy band: {classify_pd(pd_base, th)}.",
    f"Stress results: +{(tA['ΔPD'].iloc[0]):+.2%} (1σ), +{(tB['ΔPD'].iloc[0]):+.2%} (2σ), CVaR uplift {float(tC.loc[tC['Scenario']=='Monte Carlo CVaR 95%','ΔPD'].iloc[0]):+.2%}.",
]
if ref is not None:
    summary_lines.append("PSI computed against training reference; features with 'Shift' status require review.")
st.write("\n".join(f"- {s}" for s in summary_lines))

st.markdown("---")
st.caption("© Corporate Risk Analytics — single-page portal. English UI. Sidebar inputs → main outputs. LightGBM only.")
