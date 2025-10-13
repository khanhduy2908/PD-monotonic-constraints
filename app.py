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
from utils.model_scoring import load_lgbm_model, model_feature_names, predict_pd, explain_shap
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd
from utils.drift_monitoring import compute_psi_table
from utils.visualization import default_distribution_by_year, default_distribution_by_sector

# ===================== Page config & styling =====================
st.set_page_config(page_title="Corporate Default Risk Scoring Portal", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.25rem; padding-bottom:2rem;}
h1,h2,h3 {font-weight:600;}
.stMetric {text-align:center}
.stExpander {border-radius:10px; border:1px solid #E6E8EB;}
.badge {display:inline-block; padding:4px 8px; border-radius:8px; background:#F2F4F7; font-size:12px;}
.table-title {font-weight:600; margin-top:0.25rem; margin-bottom:0.25rem;}
</style>
""", unsafe_allow_html=True)

# ===================== Helpers =====================
def _safe(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

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
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def policy_badge(th):
    st.markdown(f"<span class='badge'>Policy: Low &lt; {th['low']:.0%} • Medium &lt; {th['medium']:.0%}</span>", unsafe_allow_html=True)

def load_train_reference():
    for p in ["models/train_reference.parquet", "models/train_reference.csv"]:
        if os.path.exists(p):
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                pass
    return None

def compute_feature_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    sub = df[features].replace([np.inf, -np.inf], np.nan)
    stats = pd.DataFrame({
        "mean": sub.mean(skipna=True),
        "std": sub.std(ddof=0, skipna=True)
    })
    stats["std"] = stats["std"].replace(0, np.nan)
    return stats

# Which features move PD upward when increasing (↑ bad) vs downward (↑ good)
RISK_UP_FEATURES = {
    "Debt_to_Assets", "Debt_to_Equity", "Total_Debt_to_EBITDA", "Net_Debt_to_Equity", "Long_Term_Debt_to_Assets"
}
RISK_DOWN_FEATURES = {
    "ROE", "ROA", "Current_Ratio", "Quick_Ratio", "Interest_Coverage", "EBITDA_to_Interest", "Operating_Income_to_Debt"
}

def apply_market_shock(base_row: pd.Series, features: list, stats: pd.DataFrame, sigma_multiple: float) -> pd.Series:
    """Deterministic shock: move risk-up features by +k*σ, risk-down features by -k*σ."""
    x = base_row.copy()
    for f in features:
        if f not in stats.index:
            continue
        s = stats.loc[f, "std"]
        if not np.isfinite(s) or s == 0:
            continue
        if f in RISK_UP_FEATURES:
            x[f] = float(x[f]) + sigma_multiple * float(s)
        elif f in RISK_DOWN_FEATURES:
            x[f] = float(x[f]) - sigma_multiple * float(s)
        # else: leave unchanged
    return x

def monte_carlo_cvar_pd(model, base_row: pd.Series, features: list, stats: pd.DataFrame, sims: int = 2000, alpha: float = 0.95) -> dict:
    """Independent shocks ~ N(0, σ^2), directional signs based on risk effect sets."""
    mu = np.zeros(len(features))
    sigmas = np.array([stats.loc[f, "std"] if (f in stats.index and np.isfinite(stats.loc[f, "std"]) and stats.loc[f,"std"]>0) else 0.0 for f in features])
    sigmas = np.nan_to_num(sigmas, nan=0.0, posinf=0.0, neginf=0.0)

    base_vec = np.array([float(base_row.get(f, 0.0)) for f in features], dtype=float)
    # Direction multipliers: +1 for risk-up features, -1 for risk-down features, 0 otherwise
    dirs = np.array([1.0 if f in RISK_UP_FEATURES else (-1.0 if f in RISK_DOWN_FEATURES else 0.0) for f in features], dtype=float)

    # Generate shocks
    shocks = np.random.normal(loc=0.0, scale=sigmas, size=(sims, len(features))) * dirs
    sims_mat = base_vec.reshape(1, -1) + shocks

    # Predict PD for all simulations (batch for speed)
    X = pd.DataFrame(sims_mat, columns=features)
    X = _safe(X)
    if hasattr(model, "predict_proba"):
        pd_sims = model.predict_proba(X)[:, 1]
    else:
        pd_sims = model.predict(X).astype(float)

    var = np.quantile(pd_sims, alpha)
    cvar = pd_sims[pd_sims >= var].mean() if (pd_sims >= var).any() else var
    return {"VaR": float(var), "CVaR": float(cvar), "PD_sims": pd_sims}

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

# Prepare feature list in the exact model order if available
candidate_features = default_financial_feature_list()
model_feats = model_feature_names(model)
final_features = select_features_for_model(features_df, candidate_features, model_feats)

# Ticker / Year selectors (auto)
all_tickers = sorted(features_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0 if all_tickers else None)

years_avail = sorted(features_df.loc[features_df["Ticker"].astype(str)==ticker, "Year"].dropna().astype(int).unique().tolist())
default_year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=default_year_idx)

# Stress scenario
scenario = st.sidebar.selectbox(
    "Stress Test Scenario",
    [
        "Baseline",
        "Market Shock (1σ)",
        "Market Shock (2σ)",
        "Monte Carlo CVaR 95%"
    ],
    index=0
)

# Optional PSI toggle
enable_psi = st.sidebar.checkbox("Compute PSI (if train_reference exists)", value=True)

# ===================== Main (outputs only) =====================
st.title("Corporate Default Risk Scoring Portal")
st.caption("Single-page. Sidebar for inputs. Main for outputs. LightGBM scoring, policy by sector, SHAP explainability, market stress, and Monte Carlo CVaR.")

# Company selection & sector detect
row_sel = features_df[(features_df["Ticker"].astype(str)==ticker) & (features_df["Year"]==year)]
if row_sel.empty:
    st.error("No matching record for selected Ticker & Year.")
    st.stop()

x = row_sel.iloc[0]
sector_detected = str(x.get("Sector", "")) if pd.notna(x.get("Sector", "")) else ""
exchange_detected = str(x.get("Exchange", "")) if pd.notna(x.get("Exchange", "")) else ""
X_base = pd.DataFrame([x[final_features].values], columns=final_features)
X_base = _safe(X_base)

# ========== A) Company Overview ==========
st.subheader("A. Company Overview")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Ticker", ticker)
c2.metric("Year", int(year))
c3.metric("Sector", sector_detected if sector_detected else "-")
c4.metric("Exchange", exchange_detected if exchange_detected else "-")

# Optional high-level portfolio visuals (useful context for RM)
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
# Predict PD
pd_base = predict_pd(model, X_base)
# Thresholds by sector (fallback to __default__)
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

# ========== D) Stress Testing ==========
st.subheader("D. Stress Testing")
# Precompute feature stats for stress engines
stats = compute_feature_stats(features_df, final_features)

stress_summary = None
if scenario.startswith("Market Shock"):
    k = 1.0 if "1σ" in scenario else 2.0
    x_shocked = apply_market_shock(x, final_features, stats, sigma_multiple=k)
    X_shock = pd.DataFrame([x_shocked[final_features].values], columns=final_features)
    X_shock = _safe(X_shock)
    pd_shock = predict_pd(model, X_shock)

    stress_summary = pd.DataFrame([
        {"Scenario": "Baseline", "PD": pd_base, "Delta_PD": 0.0},
        {"Scenario": f"Market Shock ({int(k)}σ)", "PD": pd_shock, "Delta_PD": pd_shock - pd_base}
    ])

    st.table(stress_summary.style.format({"PD":"{:.2%}", "Delta_PD":"{:+.2%}"}))

    fig = go.Figure()
    fig.add_trace(go.Bar(name="PD", x=stress_summary["Scenario"], y=stress_summary["PD"]))
    fig.update_layout(title="PD under Market Shock", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif scenario.startswith("Monte Carlo"):
    mc = monte_carlo_cvar_pd(model, x, final_features, stats, sims=2000, alpha=0.95)
    stress_summary = pd.DataFrame([
        {"Metric": "Baseline PD", "Value": pd_base},
        {"Metric": "VaR (95%) PD", "Value": mc["VaR"]},
        {"Metric": "CVaR (95%) PD", "Value": mc["CVaR"]},
        {"Metric": "Delta vs Baseline (CVaR)", "Value": mc["CVaR"] - pd_base},
    ])
    st.table(stress_summary.style.format({"Value":"{:.2%}"}))

    # Distribution plot (simple histogram)
    hist = np.histogram(mc["PD_sims"], bins=30)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=((hist[1][1:]+hist[1][:-1])/2), y=hist[0]))
    fig.add_vline(x=mc["VaR"], line_width=2, line_dash="dash", line_color="red")
    fig.add_vline(x=mc["CVaR"], line_width=2, line_dash="dot", line_color="black")
    fig.update_layout(title="Monte Carlo PD distribution with VaR/CVaR", xaxis_title="PD", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Scenario: Baseline. No shock applied.")

# ========== E) Drift Monitoring (PSI) ==========
st.subheader("E. Drift Monitoring (PSI)")
if enable_psi:
    ref = load_train_reference()
    if ref is None:
        st.info("Training reference not found. Place models/train_reference.parquet or .csv to enable PSI.")
    else:
        common = [f for f in final_features if f in features_df.columns and f in ref.columns]
        score_df = _safe(features_df[common])
        ref_df = _safe(ref[common])
        psi_table = compute_psi_table(ref_df, score_df, common, buckets=10)
        st.dataframe(psi_table, use_container_width=True)
        s1 = int((psi_table["status"]=="Stable").sum())
        s2 = int((psi_table["status"]=="Moderate").sum())
        s3 = int((psi_table["status"]=="Shift").sum())
        d1,d2,d3 = st.columns(3)
        d1.metric("Stable", s1); d2.metric("Moderate", s2); d3.metric("Shift", s3)
else:
    st.info("PSI computation disabled in sidebar.")

# ========== F) Executive Summary ==========
st.subheader("F. Executive Summary")
summary_lines = [
    f"Ticker {ticker}, Year {year}, Sector {sector_detected or '-'}, Exchange {exchange_detected or '-'}.",
    f"Predicted PD {pd_base:.2%} with policy band: {band}.",
]
if scenario.startswith("Market Shock") and stress_summary is not None:
    delta = float(stress_summary.loc[stress_summary['Scenario']!= 'Baseline', 'Delta_PD'].values[0])
    summary_lines.append(f"Market stress ({scenario}) shifts PD by {delta:+.2%} versus baseline.")
elif scenario.startswith("Monte Carlo") and stress_summary is not None:
    cvar_delta = float(stress_summary.loc[stress_summary['Metric']=='Delta vs Baseline (CVaR)', 'Value'].values[0])
    summary_lines.append(f"Monte Carlo CVaR(95%) indicates tail PD uplift of {cvar_delta:+.2%} relative to baseline.")
if enable_psi:
    summary_lines.append("PSI was computed if reference data was available; features with 'Shift' status require review.")
st.write("\n".join(f"- {s}" for s in summary_lines))

st.markdown("---")
st.caption("© Corporate Risk Analytics — single-page portal. English UI. Sidebar inputs → main outputs. LightGBM only.")
