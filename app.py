import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==== Your utils (already present in repo) ====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import preprocess_and_create_features
from utils.feature_selection import select_features_for_model
from utils.model_scoring import load_lgbm_model, model_feature_names, explain_shap
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd

# ===================== Page config & styles =====================
st.set_page_config(page_title="Corporate Default Risk Scoring", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 0.8rem; padding-bottom: 1.2rem;}
h1,h2,h3 {font-weight: 650;}
.small {font-size:12px; color:#6b7280;}
.metric-card {background:#F8FAFC;border:1px solid #E5E7EB;border-radius:10px;padding:10px 12px;margin-bottom:8px;}
hr {margin: 0.6rem 0;}
</style>
""", unsafe_allow_html=True)

# ===================== Helpers =====================
ID_LABEL_COLS = {"Year","Ticker","Sector","Exchange","Default"}

def read_csv_smart(path: str) -> pd.DataFrame:
    """Read CSV with robust encoding fallbacks."""
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.shape[1] == 0:
                raise ValueError("CSV has no columns (empty or bad delimiter).")
            return df
        except Exception:
            continue
    raise RuntimeError(f"Unable to read {path} with common encodings.")

def to_float(x):
    try:
        if pd.isna(x): return np.nan
        if isinstance(x, str): x = x.replace(",", "")
        return float(x)
    except Exception:
        return np.nan

def fmt_money(x):
    return "-" if (x is None or not np.isfinite(x)) else f"{x:,.2f}"

def fmt_ratio(x):
    if (x is None) or (not np.isfinite(x)): return "-"
    return f"{x:.2%}" if -1.5 <= float(x) <= 1.5 else f"{x:,.4f}"

def safe_df(X: pd.DataFrame) -> pd.DataFrame:
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def force_numeric(X: pd.DataFrame) -> pd.DataFrame:
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return safe_df(X)

def model_align_row(row: pd.Series, model, fallbacks: list) -> pd.DataFrame:
    """Map row -> 1xN as expected by model (add missing=0, drop extras, order correct)."""
    m_feats = model_feature_names(model)
    feats = list(m_feats) if m_feats else list(fallbacks)
    data = {f: float(row.get(f, 0.0)) for f in feats}
    X = pd.DataFrame([data], columns=feats)
    return force_numeric(X)

def align_features_to_model(X_df: pd.DataFrame, model):
    """Ensure X_df columns exactly match model.feature_name_ (order & count)."""
    model_feats = list(getattr(model, "feature_name_", []) or [])
    if not model_feats:
        return force_numeric(X_df.copy())
    X = X_df.copy()
    for col in model_feats:
        if col not in X.columns:
            X[col] = 0.0
    X = X[model_feats]
    return force_numeric(X)

def load_train_reference():
    for p in ("models/train_reference.parquet", "models/train_reference.csv"):
        if os.path.exists(p):
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                pass
    return None

# ===================== Stress testing lib (self-contained) =====================
def sector_alias_map(sector_raw: str) -> str:
    s = (sector_raw or "").lower()
    if any(k in s for k in ["tech","information","software","it"]): return "Technology"
    if "tele" in s: return "Telecom"
    if any(k in s for k in ["material","metal","mining","cement","basic res"]): return "Materials"
    if any(k in s for k in ["energy","oil","gas","coal"]): return "Energy"
    if any(k in s for k in ["bank","finance","insurance","securities"]): return "Financials"
    if any(k in s for k in ["real estate","property","construction"]): return "Real Estate"
    if any(k in s for k in ["industrial","manufacturing","machinery"]): return "Industrials"
    if any(k in s for k in ["consumer","retail","food","beverage"]): return "Consumer"
    if any(k in s for k in ["utilit"]): return "Utilities"
    return "__default__"

def normalize_sector_for_factors(sector_raw: str) -> str:
    s = (sector_raw or "").lower()
    if "basic" in s and "resource" in s: return "Steel"
    if "material" in s or "steel" in s or "metal" in s: return "Steel"
    return "__default__"

EXCHANGE_INTENSITY = {"UPCOM": 0.6, "HNX": 1.0, "HOSE": 1.0, "HSX": 1.0}

# ===================== Stress Testing Updates =====================

def apply_multipliers_once(Xrow: pd.DataFrame, mults: dict) -> pd.DataFrame:
    X = Xrow.copy()
    for feat, mult in mults.items():
        if feat in X.columns:
            X[feat] = float(X[feat].iloc[0]) * float(mult)
    return X

def score_pd(model, Xrow: pd.DataFrame) -> float:
    Xrow = align_features_to_model(Xrow, model)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(Xrow)[:, 1][0])
    return float(model.predict(Xrow)[0])

# Define sector-specific stress scenarios (customized for realistic impacts)
SECTOR_SCENARIOS = {
    "Technology": {
        "Tech Crunch": {"Revenue_CAGR_3Y": 0.70, "ROA": 0.75, "Sentiment Score": 0.85},
        "Supply Chain Disruption": {"Asset_Turnover": 0.80, "Receivables_Turnover": 0.85},
    },
    "Financials": {
        "Credit Loss Surge": {"ROA": 0.60, "Debt_to_Equity": 1.2},
        "Interest Rate Hike": {"Interest_Coverage": 0.75, "Debt_to_Assets": 1.1},
    },
    "Energy": {
        "Oil Price Drop": {"Revenue_CAGR_3Y": 0.75, "Net_Profit_Margin": 0.65},
        "Policy Change": {"ROA": 0.60, "EBITDA_to_Interest": 0.70},
    },
    "Real Estate": {
        "Housing Downturn": {"Revenue_CAGR_3Y": 0.60, "Debt_to_Assets": 1.2},
        "Credit Tightening": {"Interest_Coverage": 0.70, "Net_Profit_Margin": 0.65},
    },
    "__default__": {
        "Pandemic Shock": {"ROA": 0.70, "EBITDA_to_Interest": 0.75},
        "Supply Chain Shock": {"Revenue_CAGR_3Y": 0.75, "Sentiment Score": 0.80},
    }
}

# Run stress testing scenarios for sector-specific crises
def run_scenarios(model, X_base: pd.DataFrame, scenarios: dict) -> pd.DataFrame:
    rows = []
    for name, mults in scenarios.items():
        Xs = apply_multipliers_once(X_base, mults)
        pd_val = score_pd(model, Xs)
        rows.append({"Scenario": name, "PD": pd_val})
    return pd.DataFrame(rows).sort_values("PD", ascending=False)

# Retrieve sector scenario based on sector
sector_scenarios = SECTOR_SCENARIOS.get(sector_raw, SECTOR_SCENARIOS["__default__"])

# Apply the defined stress scenarios
df_sector = run_scenarios(model, X_base, sector_scenarios)

# ===================== Financial Overview - Additional Visualizations =====================
st.subheader("Financial Overview")

# Revenue & Net Profit over the years
rev_series = raw_df[["Year", "Net Sales", "Net Profit For the Year"]].dropna()
rev_series = rev_series.rename(columns={"Net Sales": "Revenue", "Net Profit For the Year": "Net_Profit"})

fig_rev = go.Figure()
fig_rev.add_trace(go.Bar(x=rev_series["Year"], y=rev_series["Revenue"], name="Revenue"))
fig_rev.add_trace(go.Scatter(x=rev_series["Year"], y=rev_series["Net_Profit"], name="Net Profit", mode="lines+markers", yaxis="y2"))
fig_rev.update_layout(
    title="Revenue & Net Profit (multi-year)",
    yaxis=dict(title="Revenue"),
    yaxis2=dict(title="Net Profit", overlaying="y", side="right"),
    height=380
)
st.plotly_chart(fig_rev, use_container_width=True)

# Capital structure (Debt vs Equity)
fig_cap = go.Figure(data=[go.Pie(labels=["Total Debt", "Equity"], values=[debt_raw, equity_raw], hole=0.5)])
fig_cap.update_layout(title="Capital Structure", height=380)
st.plotly_chart(fig_cap, use_container_width=True)

# Key Financial Ratios (ROA, ROE, Debt-to-Equity)
key_ratios = pd.DataFrame({
    "Metric": ["ROA", "ROE", "Debt_to_Assets", "Debt_to_Equity", "Current_Ratio", "Quick_Ratio", "Interest_Coverage"],
    "Value": [roa, roe, dta, dte, current_ratio, quick_ratio, interest_coverage]
})
key_ratios["Value"] = key_ratios["Value"].apply(fmt_ratio)
st.dataframe(key_ratios, use_container_width=True, hide_index=True)

# ===================== Monte Carlo CVaR =====================
st.markdown("**Monte Carlo CVaR 95%**")
mc_results = mc_cvar_pd(model, X_base, feats_df, sims=5000, alpha=0.95)

if isinstance(mc_results, dict) and "PD_sims" in mc_results:
    pd_var = mc_results["VaR"]
    pd_cvar = mc_results["CVaR"]
    st.metric("VaR 95% (PD)", f"{pd_var:.2%}")
    st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}")
else:
    st.warning("Monte Carlo CVaR simulation failed.")
