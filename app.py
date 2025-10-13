# app.py — Bank-grade, single-page portal (English UI)
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ==== Utils from your repo ====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import (
    preprocess_and_create_features,
    default_financial_feature_list,
)
from utils.feature_selection import select_features_for_model
from utils.model_scoring import (
    load_lgbm_model,
    model_feature_names,
    explain_shap,
)
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd
from utils.drift_monitoring import compute_psi_table
from utils.visualization import (
    default_distribution_by_year,
    default_distribution_by_sector,
)
from utils.stress_testing import (
    detect_sector_alias, exchange_intensity,
    apply_sector_crisis_row, apply_systemic_shock_row,
    systemic_sigma, monte_carlo_cvar_pd
)

# ===================== Page config & minimal styling =====================
st.set_page_config(page_title="Corporate Default Risk Scoring", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
h1,h2,h3 {font-weight: 600;}
.badge {display:inline-block; padding:4px 8px; border-radius:8px; background:#F2F4F7; font-size:12px;}
.small {font-size:12px; color:#6b7280;}
.stMetric {text-align:center}
</style>
""", unsafe_allow_html=True)

# ===================== Helpers: safety & alignment =====================
def _safe_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def _force_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return _safe_df(X)

def build_X_for_model(row: pd.Series, model, fallback_features: list) -> pd.DataFrame:
    """
    Align features to model's expected order; add missing with 0.0; drop extras.
    Prevents LightGBM validate_features error.
    """
    m_feats = model_feature_names(model)
    feats = list(m_feats) if m_feats else list(fallback_features)
    data = {f: float(row.get(f, 0.0)) for f in feats}
    X = pd.DataFrame([data], columns=feats)
    return _force_numeric(X)

def gauge_pd(pd_value: float) -> go.Figure:
    v = float(pd_value) * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=v,
        number={'suffix': "%"},
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
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def policy_badge(th):
    st.markdown(f"<span class='badge'>Policy: Low &lt; {th['low']:.0%} • Medium &lt; {th['medium']:.0%}</span>", unsafe_allow_html=True)

def load_train_reference():
    """Try models/train_reference.parquet|csv; else fallback: build from historical years != selected year."""
    for p in ["models/train_reference.parquet", "models/train_reference.csv"]:
        if os.path.exists(p):
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                pass
    return None

def compute_feature_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    valid_feats = [f for f in features if f in df.columns]
    if not valid_feats:
        return pd.DataFrame(columns=["mean", "std"])
    sub = df[valid_feats].replace([np.inf, -np.inf], np.nan)
    stats = pd.DataFrame({"mean": sub.mean(skipna=True), "std": sub.std(ddof=0, skipna=True)})
    stats["std"] = stats["std"].replace(0, np.nan)
    return stats

# ===================== Sector-aware stress setup =====================
# Directional sets: ↑ increases risk; ↓ decreases risk.
RISK_UP_FEATURES = {
    "Debt_to_Assets", "Debt_to_Equity", "Total_Debt_to_EBITDA",
    "Net_Debt_to_Equity", "Long_Term_Debt_to_Assets"
}
RISK_DOWN_FEATURES = {
    "ROA", "ROE", "Current_Ratio", "Quick_Ratio",
    "Interest_Coverage", "EBITDA_to_Interest", "Operating_Income_to_Debt"
}

# Sector-specific stress definitions (deterministic shocks in σ-units)
SECTOR_SCENARIOS = {
    "Technology":       {"Revenue_CAGR_3Y": -1.0, "EBITDA_to_Interest": -1.0, "ROE": -0.8},  # Tech shock
    "Telecom":          {"Revenue_CAGR_3Y": -0.8, "EBITDA_to_Interest": -1.0, "ROA": -0.6},
    "Materials":        {"Debt_to_Assets": +0.8, "Net_Profit_Margin": -1.0, "ROA": -0.6},    # Commodity crisis
    "Energy":           {"Debt_to_Assets": +1.0, "Net_Profit_Margin": -1.0, "ROE": -0.6},
    "Financials":       {"ROE": -1.2, "Interest_Coverage": -1.0, "Current_Ratio": -0.5},     # Financial shock
    "Real Estate":      {"Debt_to_Assets": +1.2, "Current_Ratio": -1.0, "EBITDA_to_Interest": -1.0},
    "Industrials":      {"Asset_Turnover": -1.0, "EBITDA_to_Interest": -0.8, "ROA": -0.6},   # Industrial downturn
    "Consumer":         {"Revenue_CAGR_3Y": -0.8, "Gross_Margin": -0.8, "ROE": -0.5},        # Demand shock
    "Utilities":        {"ROA": -0.4, "EBITDA_to_Interest": -0.6},                            # Mild regulatory shock
    # Default/fallback
    "__default__":      {"Revenue_CAGR_3Y": -0.7, "ROA": -0.5, "EBITDA_to_Interest": -0.6},
}

# UPCoM nuance: low observed default yet weak performance → lighten deterministic sector shock
EXCHANGE_SHOCK_ADJUST = {
    "UPCOM": 0.6,  # 60% intensity vs normal
    "HNX":   1.0,
    "HOSE":  1.0,
}

def sector_tag(sector_str: str) -> str:
    s = (sector_str or "").lower()
    if any(k in s for k in ["tech", "information", "software", "it"]): return "Technology"
    if "tele" in s: return "Telecom"
    if any(k in s for k in ["material", "metal", "mining", "cement"]): return "Materials"
    if any(k in s for k in ["energy", "oil", "gas", "coal"]): return "Energy"
    if any(k in s for k in ["bank", "finance", "insurance", "securities"]): return "Financials"
    if any(k in s for k in ["real estate", "property", "construction"]): return "Real Estate"
    if any(k in s for k in ["industrial", "manufacturing", "machinery"]): return "Industrials"
    if any(k in s for k in ["consumer", "retail", "food", "beverage"]): return "Consumer"
    if any(k in s for k in ["utilit"]): return "Utilities"
    return "__default__"

def systemic_crisis_sigma(sector_alias: str) -> float:
    # Allow slightly stronger systemic shock for Financials & Real Estate
    return 2.0 if sector_alias in {"Financials", "Real Estate"} else 1.8

def apply_sigma_shock_vector(base_vec, features, stats, sigma_map: dict, intensity_scale: float = 1.0):
    """
    Apply feature-level shocks defined in σ-units: new = base + kσ or base - kσ.
    Positive value => move in "risk-up" direction, negative => in "risk-down".
    """
    shocked = base_vec.copy()
    for i, f in enumerate(features):
        if f not in stats.index: 
            continue
        sigma = stats.loc[f, "std"]
        if not np.isfinite(sigma) or sigma == 0:
            continue
        k = sigma_map.get(f, 0.0) * intensity_scale
        if k == 0: 
            continue
        # If user specified +k for a "good" feature, move downwards; for "bad", move upwards
        if f in RISK_UP_FEATURES:
            shocked[i] = shocked[i] + (k * sigma)  # +k raises risk
        elif f in RISK_DOWN_FEATURES:
            shocked[i] = shocked[i] - (k * sigma)  # -k lowers "good" metric -> risk up
        else:
            # neutral features: apply sign directly
            shocked[i] = shocked[i] + (k * sigma)
    return shocked

def apply_market_shock_vector(base_vec, features, stats, k_sigma: float):
    """
    Generic market shock: +kσ to risk-up features, -kσ to risk-down features.
    """
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

def monte_carlo_cvar_pd(model, base_vec: np.ndarray, features: list, stats: pd.DataFrame, sims: int = 3000, alpha: float = 0.95) -> dict:
    sigmas = np.array([stats.loc[f, "std"] if (f in stats.index and np.isfinite(stats.loc[f, "std"]) and stats.loc[f,"std"]>0) else 0.0 for f in features])
    dirs = np.array([1.0 if f in RISK_UP_FEATURES else (-1.0 if f in RISK_DOWN_FEATURES else 0.0) for f in features], dtype=float)
    shocks = np.random.normal(loc=0.0, scale=sigmas, size=(sims, len(features))) * dirs
    sims_mat = base_vec.reshape(1, -1) + shocks
    X = _safe_df(pd.DataFrame(sims_mat, columns=features))
    if hasattr(model, "predict_proba"):
        pd_sims = model.predict_proba(X)[:, 1]
    else:
        pd_sims = model.predict(X).astype(float)
    var = float(np.quantile(pd_sims, alpha))
    cvar = float(pd_sims[pd_sims >= var].mean()) if (pd_sims >= var).any() else var
    return {"VaR": var, "CVaR": cvar, "PD_sims": pd_sims}

# ===================== Data & artifacts (cached) =====================
@st.cache_data(show_spinner=False)
def load_prepared_dataset():
    if not os.path.exists("bctc_final.csv"):
        raise FileNotFoundError("bctc_final.csv not found in repository root.")
    raw = pd.read_csv("bctc_final.csv")
    cleaned = clean_and_log_transform(raw)
    feats = preprocess_and_create_features(cleaned)
    return feats

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_lgbm_model("models/lgbm_model.pkl")
    thresholds = load_thresholds("models/threshold.json")
    return model, thresholds

# ===================== Sidebar (inputs + company snapshot) =====================
st.sidebar.header("Inputs")

# Load once
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

# Feature list aligned to model
candidate_features = default_financial_feature_list()
model_feats = model_feature_names(model)
final_features = select_features_for_model(features_df, candidate_features, model_feats)

# Selectors
all_tickers = sorted(features_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0 if all_tickers else None)

years_avail = sorted(features_df.loc[features_df["Ticker"].astype(str)==ticker, "Year"].dropna().astype(int).unique().tolist())
year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=year_idx)

# Sidebar company snapshot (auto)
row_sel = features_df[(features_df["Ticker"].astype(str)==ticker) & (features_df["Year"]==year)]
if row_sel.empty:
    st.sidebar.warning("No matching record for selected Ticker & Year.")
    st.stop()

x = row_sel.iloc[0]
sector_raw = str(x.get("Sector", "")) if pd.notna(x.get("Sector", "")) else ""
sector_alias = sector_tag(sector_raw)
exchange = (str(x.get("Exchange","")) or "").upper()
exchange_adj = EXCHANGE_SHOCK_ADJUST.get(exchange, 1.0)

# Key snapshot metrics
def _val(name): 
    return float(x.get(name, 0.0)) if pd.notna(x.get(name, 0.0)) else 0.0

_total_assets = _val("Total_Assets") if "Total_Assets" in x else _val("TOTAL ASSETS (Bn. VND)")
_equity = _val("Equity") if "Equity" in x else _val("OWNER'S EQUITY(Bn.VND)")
_total_debt = _val("Total_Debt")
_revenue = _val("Revenue")
_net_profit = _val("Net_Profit")
_roa = _val("ROA") ; _roe = _val("ROE")
_dte = _val("Debt_to_Equity") ; _dta = _val("Debt_to_Assets")

st.sidebar.markdown("### Company")
st.sidebar.write(f"**Ticker**: {ticker}")
st.sidebar.write(f"**Year**: {int(year)}")
st.sidebar.write(f"**Sector**: {sector_raw or '-'}  \n**Exchange**: {exchange or '-'}")
st.sidebar.markdown("### Snapshot (selected year)")
st.sidebar.write(f"Assets: { _total_assets:,.2f} • Equity: { _equity:,.2f}")
st.sidebar.write(f"Debt: { _total_debt:,.2f} • Revenue: { _revenue:,.2f}")
st.sidebar.write(f"Net Profit: { _net_profit:,.2f}")
st.sidebar.write(f"ROA: { _roa:.2%} • ROE: { _roe:.2%}")
st.sidebar.write(f"Debt/Equity: { _dte:.2f} • Debt/Assets: { _dta:.2f}")

# Build base vector
X_base = build_X_for_model(x, model=model, fallback_features=final_features)
base_vec = X_base.values[0]
stats_all = compute_feature_stats(features_df, list(X_base.columns))

# ===================== Main Dashboard =====================
st.title("Corporate Default Risk Scoring")
st.caption("Single-page. Sidebar inputs → Main outputs. Sector-specific stress + systemic crisis + Monte Carlo CVaR. Always-on PSI drift.")

# ---------- A) Company Overview ----------
st.subheader("A. Company Overview")
# 3y revenue & profit trend
hist = features_df[features_df["Ticker"].astype(str)==ticker].sort_values("Year")
trend = hist[["Year","Revenue","Net_Profit"]].dropna()
c1,c2 = st.columns([2,1])

with c1:
    if not trend.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Revenue", x=trend["Year"], y=trend["Revenue"]))
        fig.add_trace(go.Scatter(name="Net Profit", x=trend["Year"], y=trend["Net_Profit"], mode="lines+markers"))
        fig.update_layout(title="Revenue & Net Profit (multi-year)", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical series for revenue/profit.")

with c2:
    # Capital structure snapshot
    cap = pd.DataFrame({
        "Item": ["Equity", "Total Debt"],
        "Value": [ _equity, _total_debt ]
    })
    fig2 = px.pie(cap, names="Item", values="Value", hole=0.35, title="Capital Structure")
    st.plotly_chart(fig2, use_container_width=True)

# Key ratios
key_ratios = pd.DataFrame({
    "Metric": ["ROA","ROE","Debt_to_Assets","Debt_to_Equity","Current_Ratio","Quick_Ratio","Interest_Coverage","EBITDA_to_Interest"],
    "Value": [ _roa, _roe, _dta, _dte, _val("Current_Ratio"), _val("Quick_Ratio"), _val("Interest_Coverage"), _val("EBITDA_to_Interest")]
})
st.table(key_ratios.style.format({"Value": "{:,.4f}"}))

# ---------- B) PD & Policy ----------
st.subheader("B. Default Probability (PD) & Policy Band")
if hasattr(model, "predict_proba"):
    pd_base = float(model.predict_proba(X_base)[:,1][0])
else:
    pd_base = float(model.predict(X_base)[0])
th = thresholds_for_sector(load_thresholds("models/threshold.json"), sector_raw)
band = classify_pd(pd_base, th)
c3,c4,c5 = st.columns([1,1,2])
with c3: st.metric("PD", f"{pd_base:.2%}")
with c4: st.metric("Policy Band", band)
with c5: policy_badge(th)
st.plotly_chart(gauge_pd(pd_base), use_container_width=True)

# ---------- C) SHAP ----------
st.subheader("C. Model Explainability (SHAP)")
shap_df = explain_shap(model, X_base, top_n=10)
if shap_df.empty:
    st.info("SHAP explanation is not available for this model or input.")
else:
    st.dataframe(shap_df, use_container_width=True)

# ---------- D) Stress Testing (three scenarios side-by-side) ----------
st.subheader("D. Stress Testing")

# Reference for covariance (Monte Carlo)
reference = load_train_reference()
reference = reference if reference is not None else features_df

# Sector alias & exchange nuance
sector_alias = detect_sector_alias(sector_raw)
ex_int = exchange_intensity(exchange)

# 1) Sector Crisis (sector-specific; deterministic)
X_sector = apply_sector_crisis_row(X_base, sector_alias=sector_alias, exch_intensity=ex_int)
if hasattr(model, "predict_proba"):
    pd_sector = float(model.predict_proba(X_sector)[:,1][0])
else:
    pd_sector = float(model.predict(X_sector)[0])

# 2) Systemic Crisis (σ-based; common to all sectors)
k_sys = systemic_sigma(sector_alias)
X_sys = apply_systemic_shock_row(X_base, reference_df=reference, k_sigma=k_sys)
if hasattr(model, "predict_proba"):
    pd_sys = float(model.predict_proba(X_sys)[:,1][0])
else:
    pd_sys = float(model.predict(X_sys)[0])

# 3) Monte Carlo CVaR 95% (correlated, clipped)
mc = monte_carlo_cvar_pd(model, X_base, reference_df=reference, sims=5000, alpha=0.95)
pd_var, pd_cvar = mc["VaR"], mc["CVaR"]

# ---- Layout: 2x2 charts (no tables) ----
cA, cB = st.columns(2)
with cA:
    st.markdown("**Sector Crisis (sector-specific)**")
    figA = go.Figure()
    figA.add_trace(go.Bar(x=["Sector Crisis"], y=[pd_sector]))
    figA.update_layout(yaxis=dict(tickformat=".0%"), title=f"Sector: {sector_alias}")
    st.plotly_chart(figA, use_container_width=True)

with cB:
    st.markdown(f"**Systemic Crisis ({k_sys:.1f}σ)**")
    figB = go.Figure()
    figB.add_trace(go.Bar(x=[f"Systemic {k_sys:.1f}σ"], y=[pd_sys]))
    figB.update_layout(yaxis=dict(tickformat=".0%"), title="Market-wide shock")
    st.plotly_chart(figB, use_container_width=True)

cC, cD = st.columns(2)
with cC:
    st.markdown("**Monte Carlo CVaR 95%**")
    hist = np.histogram(mc["PD_sims"], bins=40)
    centers = (hist[1][1:]+hist[1][:-1])/2
    figC = go.Figure()
    figC.add_trace(go.Bar(x=centers, y=hist[0]))
    figC.add_vline(x=pd_var, line_width=2, line_dash="dash", line_color="red")
    figC.add_vline(x=pd_cvar, line_width=2, line_dash="dot", line_color="black")
    figC.update_layout(
        title="PD distribution (VaR 95% in red, CVaR 95% in black)",
        xaxis_title="PD", yaxis_title="Frequency"
    )
    st.plotly_chart(figC, use_container_width=True)

with cD:
    # Key numbers as clean metrics (no tables)
    st.metric("Sector Crisis PD", f"{pd_sector:.2%}")
    st.metric("Systemic Crisis PD", f"{pd_sys:.2%}")
    st.metric("Monte Carlo VaR 95%", f"{pd_var:.2%}")
    st.metric("Monte Carlo CVaR 95%", f"{pd_cvar:.2%}")

# ---------- E) PSI Drift (always on; fallback to in-sample baseline) ----------
st.subheader("E. Drift Monitoring (PSI)")
reference = load_train_reference()
if reference is None:
    # Fallback: historical baseline excluding selected year (to ensure PSI always shows)
    hist_other_years = features_df[(features_df["Ticker"].astype(str)==ticker) & (features_df["Year"]!=year)]
    ref_df = hist_other_years[final_features].copy() if not hist_other_years.empty else features_df[final_features].copy()
    ref_df = _safe_df(ref_df)
    st.info("Training reference not found. Using historical snapshot as baseline for PSI.")
else:
    # Align to features available
    ref_df = reference.copy()
    common_cols = [f for f in final_features if f in ref_df.columns]
    ref_df = _safe_df(ref_df[common_cols])
# Current population = entire (prepared) dataset on common features
common = [f for f in final_features if f in features_df.columns and f in ref_df.columns]
score_df = _safe_df(features_df[common])
psi_table = compute_psi_table(ref_df[common], score_df[common], common, buckets=10)
st.dataframe(psi_table, use_container_width=True)
s1 = int((psi_table["status"]=="Stable").sum())
s2 = int((psi_table["status"]=="Moderate").sum())
s3 = int((psi_table["status"]=="Shift").sum())
m1,m2,m3 = st.columns(3)
m1.metric("Stable", s1); m2.metric("Moderate", s2); m3.metric("Shift", s3)

# ---------- F) Executive Summary ----------
st.subheader("F. Executive Summary")
summary = [
    f"Ticker {ticker}, Year {int(year)}, Sector '{sector_raw or '-'}', Exchange '{exchange or '-'}'.",
    f"Baseline PD {pd_base:.2%} ⇒ Policy band: {band}.",
    f"Sector crisis ΔPD: {t_sector['ΔPD'].iloc[0]:+.2%}; Systemic crisis ΔPD: {t_sys['ΔPD'].iloc[0]:+.2%}; CVaR uplift: {float(t_mc.loc[t_mc['Scenario']=='Monte Carlo CVaR 95%','ΔPD'].iloc[0]):+.2%}.",
    "PSI computed against training or historical baseline; features flagged 'Shift' require review."
]
st.write("\n".join(f"- {line}" for line in summary))

st.markdown("---")
st.caption("© Corporate Risk Analytics — single-page portal. English UI. Sidebar snapshot → main outputs. LightGBM-only scoring.")
