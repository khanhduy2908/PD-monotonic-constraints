# ========================================
# app.py — Professional Bank-Grade PD Stress Testing Dashboard
# Version: 2025.10
# ========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import shap
import warnings
warnings.filterwarnings("ignore")

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="PD Stress Test Dashboard",
    page_icon="📊",
    layout="wide"
)
st.markdown(
    """
    <style>
    .stMetric label, .stMarkdown {font-family: "Helvetica Neue", sans-serif;}
    .stMetric {font-size: 14px;}
    .small {font-size: 12px; color: gray;}
    </style>
    """, unsafe_allow_html=True
)

# ---------- HELPER: Plotly without warnings ----------
def show_plotly(fig, key: str):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("bctc_final.csv")
    return df

@st.cache_resource
def load_model():
    with open("lgbm_model.pkl", "rb") as f:
        return pickle.load(f)

feats_df = load_data()
model = load_model()

# ---------- SIDEBAR ----------
st.sidebar.title("Select Company Profile")
all_tickers = sorted(feats_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0, key="sb_ticker")

years_avail = sorted(
    feats_df.loc[feats_df["Ticker"].astype(str)==ticker, "Year"]
    .dropna().astype(int).unique().tolist()
)
year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=year_idx, key=f"sb_year_{ticker}")

row_raw = feats_df[(feats_df["Ticker"].astype(str)==ticker) & (feats_df["Year"]==year)]
if row_raw.empty:
    st.error("No data available for this selection.")
    st.stop()
row_raw = row_raw.iloc[0]
sector_raw = str(row_raw.get("Sector", "Other"))
exchange = str(row_raw.get("Exchange", "HOSE")).upper()

st.title(f"📊 Probability of Default (PD) & Stress Testing — {ticker} ({year})")
st.caption(f"Sector: **{sector_raw}** • Exchange: **{exchange}**")

# ---------- Helper: extract features ----------
X_base = pd.DataFrame([row_raw.drop(["Ticker","Year","Sector","Exchange"], errors="ignore")])
X_base = X_base.select_dtypes(include=[np.number]).fillna(0)

# =====================================
# A. OVERVIEW SECTION
# =====================================
st.subheader("A. Financial Overview")

overview_cols = [
    ("Total_Assets", "Total Assets (₫B)"),
    ("Total_Liabilities", "Liabilities (₫B)"),
    ("Equity", "Equity (₫B)"),
    ("Revenue", "Revenue (₫B)"),
    ("Net_Income", "Net Income (₫B)"),
]
vals = {lbl: float(row_raw.get(col, np.nan)) for col, lbl in overview_cols}

c1,c2,c3,c4,c5 = st.columns(5)
for (col, lbl), c in zip(overview_cols, [c1,c2,c3,c4,c5]):
    val = vals[lbl]
    c.metric(lbl, f"{val:,.0f}" if np.isfinite(val) else "-")

# =====================================
# B. DEFAULT PROBABILITY (PD) — MULTI-FACTOR
# =====================================
st.subheader("B. Default Probability (PD) & Policy Band")

def _logit(p, eps=1e-9):
    p = float(np.clip(p, eps, 1-eps)); return np.log(p/(1-p))
def _sigmoid(z):
    z = float(z)
    if z >= 35: return 1.0
    if z <= -35: return 0.0
    return 1.0 / (1.0 + np.exp(-z))

# --- Base model PD ---
pd_model = float(model.predict_proba(X_base)[:,1][0]) if hasattr(model,"predict_proba") else float(model.predict(X_base)[0])

# --- Overrides per ticker (manual control for high-risk firms) ---
TICKER_OVERRIDES = {
    "HAG": {"logit_boost": 2.2, "severity_boost": 0.5, "pd_floor": 0.40},
    "ROS": {"logit_boost": 1.8, "severity_boost": 0.4, "pd_floor": 0.35},
    "FLC": {"logit_boost": 1.6, "severity_boost": 0.3, "pd_floor": 0.30},
    "TGG": {"logit_boost": 1.5, "severity_boost": 0.3, "pd_floor": 0.25},
}

# --- Configs ---
PD_CFG = {
    "exchange_logit_mult": {"UPCOM": 1.10, "HNX": 0.45, "HOSE": 0.00, "__default__": 0.20},
    "sector_tilt": {"Real Estate": 0.60, "Materials": 0.25, "Financials": 0.00, "Technology": 0.00, "__default__": 0.05},
    "pd_floor": {"UPCOM": 0.12, "HNX": 0.07, "HOSE": 0.03, "__default__": 0.05},
    "pd_cap": 0.98
}

sector_bucket = sector_raw if sector_raw in PD_CFG["sector_tilt"] else "Other"
logit0 = _logit(pd_model)
adj = 0.0
adj += PD_CFG["exchange_logit_mult"].get(exchange, PD_CFG["exchange_logit_mult"]["__default__"])
adj += PD_CFG["sector_tilt"].get(sector_bucket, PD_CFG["sector_tilt"]["__default__"])

# --- Overrides ---
ovr = TICKER_OVERRIDES.get(str(ticker), {})
adj += float(ovr.get("logit_boost", 0.0))
pd_floor = float(ovr.get("pd_floor", PD_CFG["pd_floor"].get(exchange, PD_CFG["pd_floor"]["__default__"])))
pd_cap = PD_CFG["pd_cap"]

pd_final = float(np.clip(_sigmoid(logit0 + adj), pd_floor, pd_cap))
pd_final = max(pd_final, pd_floor)  # enforce floor

c1,c2 = st.columns(2)
c1.metric("PD (Post Adjustment)", f"{pd_final:.2%}")
c2.metric("Risk Level", "High" if pd_final >= 0.80 else "Medium" if pd_final >= 0.40 else "Low")

# --- PD Gauge ---
fig_g = go.Figure(go.Indicator(
    mode="gauge+number",
    value=pd_final*100,
    number={'suffix': '%'},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': '#1f77b4'},
        'steps': [
            {'range': [0, 40], 'color': '#E8F1FB'},
            {'range': [40, 80], 'color': '#CFE3F7'},
            {'range': [80, 100], 'color': '#F9E3E3'}
        ],
        'threshold': {'line': {'color': 'red', 'width': 3}, 'value': pd_final*100}
    }
))
fig_g.update_layout(height=250, margin=dict(l=10,r=10,t=10,b=10))
show_plotly(fig_g, "pd_gauge")

# =====================================
# C. KEY FINANCIAL RATIOS (with SHAP)
# =====================================
st.subheader("C. Key Financial Ratios & Model Explanation")

key_ratios = [
    "ROA","ROE","Debt_to_Assets","Debt_to_Equity",
    "Current_Ratio","Quick_Ratio","Revenue_CAGR_3Y","Net_Profit_Margin"
]
ratios_df = pd.DataFrame({
    "Ratio": key_ratios,
    "Value": [row_raw.get(r, np.nan) for r in key_ratios]
})
st.dataframe(ratios_df, hide_index=True, use_container_width=True, key="ratios_table")

# --- SHAP summary (safe) ---
try:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_base)
    if isinstance(shap_values, list): shap_values = shap_values[1]
    shap_df = pd.DataFrame({
        "Feature": X_base.columns,
        "SHAP": shap_values[0]
    })
    shap_df["absSHAP"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("absSHAP", ascending=False).head(10)
    fig_sh = go.Figure()
    fig_sh.add_trace(go.Bar(
        x=shap_df["absSHAP"], y=shap_df["Feature"],
        orientation="h", text=[f"{v:.4f}" for v in shap_df["absSHAP"]],
        textposition="outside", marker_color="#2E86C1"
    ))
    fig_sh.update_layout(title="Top 10 SHAP Drivers of PD",
                         height=360, margin=dict(l=10,r=10,t=40,b=40))
    show_plotly(fig_sh, "shap_chart")
except Exception:
    st.info("SHAP not available for this model or input.")

# =====================================
# D. STRESS TESTING — AUTOMATED (SECTOR + SYSTEMIC)
# =====================================
st.subheader("D. Stress Testing — Sector & Systemic Impacts")

# --- Crisis library ---
SECTOR_CRISES = {
    "Materials": [
        ("Steel Price Collapse", {"ROA":0.90,"ROE":0.90,"Net_Profit_Margin":0.85}),
        ("Energy Cost Surge", {"Gross_Margin":0.92,"Net_Profit_Margin":0.90}),
    ],
    "Real Estate": [
        ("Credit Tightening", {"Debt_to_Equity":1.12,"Current_Ratio":0.90}),
        ("Property Price Drop", {"Net_Profit_Margin":0.88,"ROE":0.90}),
    ],
    "Technology": [
        ("Valuation Reset", {"Net_Profit_Margin":0.93,"ROE":0.93}),
        ("Supply Chain Disruption", {"Inventory_Turnover":0.88,"Receivables_Turnover":0.92}),
    ],
    "Consumer Discretionary": [
        ("COVID Demand Shock", {"Revenue_CAGR_3Y":0.85,"ROA":0.88}),
    ],
    "Other": [("Generic Sector Shock", {"ROA":0.95,"Net_Profit_Margin":0.95})]
}

SYSTEMIC_CRISES = [
    ("Global Financial Crisis", {"ROA":0.90,"ROE":0.88}),
    ("Interest Rate +300bps", {"Debt_to_Equity":1.12,"Debt_to_Assets":1.06}),
    ("US–China Tariffs", {"Gross_Margin":0.96,"Net_Profit_Margin":0.95}),
]

sector_bucket = sector_raw if sector_raw in SECTOR_CRISES else "Other"
base_pd = pd_final
logit_base = _logit(base_pd)

def scenario_pd(multiplier, severity=1.0):
    bump = 0.8 * severity
    return _sigmoid(logit_base + bump)

# --- Severity auto by ticker ---
sev_base = 1.0
if exchange == "UPCOM": sev_base += 0.2
if ticker in TICKER_OVERRIDES:
    sev_base += TICKER_OVERRIDES[ticker].get("severity_boost", 0.0)

# --- Sector scenarios ---
rows_sector = []
for name, _ in SECTOR_CRISES[sector_bucket]:
    pd_new = scenario_pd(_, sev_base)
    rows_sector.append({"Scenario": name, "PD": pd_new, "ΔPD%": (pd_new-base_pd)/base_pd*100})
df_sector = pd.DataFrame(rows_sector)

# --- Systemic scenarios ---
rows_sys = []
for name, _ in SYSTEMIC_CRISES:
    pd_new = scenario_pd(_, sev_base*0.8)
    rows_sys.append({"Scenario": name, "PD": pd_new, "ΔPD%": (pd_new-base_pd)/base_pd*100})
df_sys = pd.DataFrame(rows_sys)

# --- Plots ---
c1,c2 = st.columns(2)
if not df_sector.empty:
    f1 = go.Figure(go.Bar(x=df_sector["Scenario"], y=df_sector["ΔPD%"],
        text=[f"{v:.1f}%" for v in df_sector["ΔPD%"]], textposition="outside"))
    f1.update_layout(title=f"Sector Impact — {sector_bucket}", yaxis_title="ΔPD (%)",
                     height=360, margin=dict(l=10,r=10,t=40,b=80))
    show_plotly(f1, "chart_sector")
if not df_sys.empty:
    f2 = go.Figure(go.Bar(x=df_sys["Scenario"], y=df_sys["ΔPD%"],
        text=[f"{v:.1f}%" for v in df_sys["ΔPD%"]], textposition="outside"))
    f2.update_layout(title="Systemic Impact", yaxis_title="ΔPD (%)",
                     height=360, margin=dict(l=10,r=10,t=40,b=80))
    show_plotly(f2, "chart_systemic")

# --- Metrics summary ---
k1,k2 = st.columns(2)
k1.metric("Max PD under Sector Stress", f"{df_sector['PD'].max():.2%}")
k2.metric("Max PD under Systemic Stress", f"{df_sys['PD'].max():.2%}")
