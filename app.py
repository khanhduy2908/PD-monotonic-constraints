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

def compute_feature_stats(df: pd.DataFrame, features: list) -> pd.DataFrame:
    valid = [f for f in features if f in df.columns]
    if not valid: return pd.DataFrame(columns=["mean","std"])
    sub = df[valid].replace([np.inf,-np.inf], np.nan)
    stats = pd.DataFrame({"mean": sub.mean(skipna=True), "std": sub.std(ddof=0, skipna=True)})
    stats["std"] = stats["std"].replace(0, np.nan)
    return stats

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

# Sector factor library (example for Steel/Materials)
SECTOR_FACTORS = {
    "Steel": {
        "Demand/Supply": {
            "Revenue_CAGR_3Y": 0.65,
            "Asset_Turnover": 0.80,
            "EBITDA_to_Interest": 0.70,
            "ROA": 0.85
        },
        "Steel Price": {
            "Gross_Margin": 0.60,
            "Net_Profit_Margin": 0.60,
            "ROE": 0.85,
            "EBITDA_to_Interest": 0.75
        },
        "Pandemic": {
            "Asset_Turnover": 0.70,
            "Receivables_Turnover": 0.70,
            "Revenue_CAGR_3Y": 0.60
        }
    },
    "__default__": {
        "Sector Shock": { "ROA": 0.85, "EBITDA_to_Interest": 0.80, "Revenue_CAGR_3Y": 0.85 }
    }
}

# Systemic factor library
SYSTEMIC_FACTORS = {
    "Interest Rate +300bps": {
        "Interest_Coverage": 0.60,
        "EBITDA_to_Interest": 0.60,
        "Operating_Income_to_Debt": 0.85
    },
    "Government Tightening": {
        "Current_Ratio": 0.90,
        "Quick_Ratio": 0.90,
        "ROA": 0.90,
        "Debt_to_Assets": 1.10
    }
}

# ===================== Load data & artifacts =====================
@st.cache_data(show_spinner=False)
def load_raw_and_features():
    if not os.path.exists("bctc_final.csv"):
        raise FileNotFoundError("bctc_final.csv not found in repository root.")
    raw = read_csv_smart("bctc_final.csv")            # raw for overview
    cleaned = clean_and_log_transform(raw.copy())     # pipeline input
    feats = preprocess_and_create_features(cleaned)   # engineered features for model
    return raw, feats

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_lgbm_model("models/lgbm_model.pkl")
    thresholds = load_thresholds("models/threshold.json")
    return model, thresholds

# ===================== Header =====================
st.title("Corporate Default Risk Scoring")
st.caption("English UI • Single page • LightGBM scoring • SHAP explainability • Sector & Systemic stress • Monte Carlo CVaR")

# ===================== Data init =====================
try:
    raw_df, feats_df = load_raw_and_features()
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

try:
    model, thresholds = load_artifacts()
except Exception as e:
    st.error(f"Artifacts error: {e}")
    st.stop()

numeric_cols = [c for c in feats_df.columns if pd.api.types.is_numeric_dtype(feats_df[c])]
candidate_features = [c for c in numeric_cols if c not in ID_LABEL_COLS]
model_feats = model_feature_names(model)
final_features = select_features_for_model(feats_df, candidate_features, model_feats)

# ===================== Sidebar Inputs & Profile =====================
all_tickers = sorted(feats_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0 if all_tickers else None)

years_avail = sorted(feats_df.loc[feats_df["Ticker"].astype(str)==ticker, "Year"].dropna().astype(int).unique().tolist())
year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=year_idx)

row_model = feats_df[(feats_df["Ticker"].astype(str)==ticker) & (feats_df["Year"]==year)]
if row_model.empty:
    st.warning("No record for selected Ticker & Year.")
    st.stop()
row_model = row_model.iloc[0]

row_raw = raw_df[(raw_df["Ticker"].astype(str)==ticker) & (raw_df["Year"]==year)]
row_raw = row_raw.iloc[0] if not row_raw.empty else pd.Series(dtype="object")

sector_raw = str(row_model.get("Sector","")) if pd.notna(row_model.get("Sector","")) else ""
sector_alias = sector_alias_map(sector_raw)
sector_norm_for_factors = normalize_sector_for_factors(sector_raw)
exchange = (str(row_model.get("Exchange","")) or "").upper()
ex_intensity = EXCHANGE_INTENSITY.get(exchange, 1.0)

# ---- Extract raw values (robust) ----
def get_raw(col_names, default=np.nan):
    for c in col_names:
        if c in row_raw.index:
            return to_float(row_raw[c])
    return default

assets_raw = get_raw(["TOTAL ASSETS (Bn. VND)","Total_Assets"])
equity_raw = get_raw(["OWNER'S EQUITY(Bn.VND)","Equity"])
curr_liab = get_raw(["Current liabilities (Bn. VND)","Current_Liabilities"], 0.0)
long_liab = get_raw(["Long-term liabilities (Bn. VND)","Long_Term_Liabilities"], 0.0)
short_bor = get_raw(["Short-term borrowings (Bn. VND)","Short_Term_Borrowings"], 0.0)

revenue_raw = get_raw(["Net Sales","Revenue"])
net_profit_raw = get_raw(["Net Profit For the Year","Net_Profit"])
oper_profit_raw = get_raw(["Operating Profit/Loss","Operating_Profit"])
interest_exp_raw = get_raw(["Interest Expenses","Interest_Expenses"], 0.0)
cash_raw = get_raw(["Cash and cash equivalents (Bn. VND)","Cash"], 0.0)
receivables_raw = get_raw(["Accounts receivable (Bn. VND)","Receivables"], 0.0)
inventories_raw = get_raw(["Net Inventories","Inventories"], 0.0)
current_assets_raw = get_raw(["CURRENT ASSETS (Bn. VND)","Current_Assets"], 0.0)

# ---- Debt & ratios (no double-count; bounded presentation) ----
def safe_div(a, b):
    try:
        return (float(a) / float(b)) if (b not in [0, None, np.nan]) else np.nan
    except Exception:
        return np.nan

total_liab_raw = (curr_liab or 0.0) + (long_liab or 0.0)
interest_bearing_debt = (short_bor or 0.0) + (long_liab or 0.0)
if "Total_Debt" in row_raw.index and pd.notna(row_raw.get("Total_Debt")):
    debt_raw = to_float(row_raw.get("Total_Debt"))
else:
    debt_raw = interest_bearing_debt

roa = safe_div(net_profit_raw, assets_raw)
roe = safe_div(net_profit_raw, equity_raw)

dta = safe_div(total_liab_raw, assets_raw)
if pd.notna(dta): dta = min(max(dta, 0.0), 0.999)  # show <= 100%

dte = safe_div(debt_raw, equity_raw)
if pd.notna(dte): dte = min(max(dte, 0.0), 0.999)  # show <= 100%

current_ratio = safe_div(current_assets_raw, curr_liab)
quick_ratio   = safe_div((cash_raw or 0.0) + (receivables_raw or 0.0), curr_liab)
interest_coverage  = safe_div(oper_profit_raw, interest_exp_raw)
ebitda_to_interest = np.nan  # add if Depreciation available

# ---- Sidebar profile ----
with st.sidebar:
    st.header("Company Profile")
    st.subheader(f"{ticker} — {int(year)}")
    st.markdown(f"**Sector:** {sector_raw or '-'}  \n**Exchange:** {exchange or '-'}")
    st.markdown("<div class='metric-card'>"
                f"Total Assets: <b>{fmt_money(assets_raw)}</b><br>"
                f"Equity: <b>{fmt_money(equity_raw)}</b><br>"
                f"Debt: <b>{fmt_money(debt_raw)}</b><br>"
                f"Revenue: <b>{fmt_money(revenue_raw)}</b><br>"
                f"Net Profit: <b>{fmt_money(net_profit_raw)}</b>"
                "</div>", unsafe_allow_html=True)
    st.markdown("<div class='metric-card'>"
                f"ROA: <b>{fmt_ratio(roa)}</b><br>"
                f"ROE: <b>{fmt_ratio(roe)}</b><br>"
                f"Debt/Equity: <b>{fmt_ratio(dte)}</b><br>"
                f"Debt/Assets: <b>{fmt_ratio(dta)}</b>"
                "</div>", unsafe_allow_html=True)

# ===================== Build model input =====================
X_base = model_align_row(row_model, model, fallbacks=final_features)
X_base = align_features_to_model(X_base, model)   # ensure exact shape as training
features_order = list(X_base.columns)

# ===================== A) Company Overview =====================
st.subheader("A. Company Financial Overview")
hist = raw_df[raw_df["Ticker"].astype(str)==ticker].sort_values("Year")
rev_series = hist[["Year","Net Sales","Net Profit For the Year"]].rename(
    columns={"Net Sales":"Revenue","Net Profit For the Year":"Net_Profit"}
).dropna(how="any")

col1, col2 = st.columns([2,1])
with col1:
    if not rev_series.empty:
        fig_rev = go.Figure()
        fig_rev.add_trace(go.Bar(x=rev_series["Year"], y=rev_series["Revenue"], name="Revenue"))
        fig_rev.add_trace(go.Scatter(x=rev_series["Year"], y=rev_series["Net_Profit"],
                                     name="Net Profit", mode="lines+markers", yaxis="y2"))
        fig_rev.update_layout(
            title="Revenue & Net Profit (multi-year)",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(title="Net Profit", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=380
        )
        st.plotly_chart(fig_rev, use_container_width=True)
    else:
        st.info("No historical series for this company.")

with col2:
    fig_cap = go.Figure(data=[go.Pie(labels=["Total Debt","Equity"], values=[debt_raw, equity_raw], hole=0.5)])
    fig_cap.update_layout(title="Capital Structure", height=380)
    st.plotly_chart(fig_cap, use_container_width=True)

st.markdown("### Key Financial Ratios")
key_ratios = pd.DataFrame({
    "Metric": ["ROA","ROE","Debt_to_Assets","Debt_to_Equity","Current_Ratio","Quick_Ratio","Interest_Coverage","EBITDA_to_Interest"],
    "Value": [roa, roe, dta, dte, current_ratio, quick_ratio, interest_coverage, ebitda_to_interest]
})
key_ratios["Value"] = key_ratios["Value"].apply(fmt_ratio)
st.dataframe(key_ratios, use_container_width=True, hide_index=True)

# ===================== B) PD & Policy =====================
st.subheader("B. Default Probability (PD) & Policy Band")
if hasattr(model, "predict_proba"):
    pd_base = float(model.predict_proba(X_base)[:,1][0])
else:
    pd_base = float(model.predict(X_base)[0])

thr = thresholds_for_sector(load_thresholds("models/threshold.json"), sector_raw)
band = classify_pd(pd_base, thr)

c1,c2,c3 = st.columns([1,1,2])
with c1: st.metric("PD", f"{pd_base:.2%}")
with c2: st.metric("Policy Band", band)
with c3:
    st.markdown(f"<span class='small'>Policy: Low &lt; {thr['low']:.0%} • Medium &lt; {thr['medium']:.0%}</span>", unsafe_allow_html=True)
fig_g = go.Figure(go.Indicator(mode="gauge+number", value=pd_base*100,
                               number={'suffix': "%"},
                               gauge={'axis': {'range': [0,100]},
                                      'bar': {'color': '#1f77b4'},
                                      'steps': [{'range':[0,10],'color':'#E8F1FB'},
                                                {'range':[10,30],'color':'#CFE3F7'},
                                                {'range':[30,100],'color':'#F9E3E3'}],
                                      'threshold': {'line': {'color':'red','width':3},'value':pd_base*100}}))
fig_g.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig_g, use_container_width=True)

# ===================== C) SHAP =====================
st.subheader("C. Model Explainability (SHAP)")
try:
    shap_df = explain_shap(model, X_base, top_n=10)
except Exception:
    shap_df = pd.DataFrame()
if shap_df.empty:
    st.info("SHAP explanation is not available for this model/input.")
else:
    st.dataframe(shap_df, use_container_width=True, hide_index=True)

# ===================== D) Stress Testing (Factor-level, sector & systemic) =====================
st.subheader("D. Stress Testing")

# Ensure sector_raw is valid and not empty
if not sector_raw or pd.isna(sector_raw.strip()):
    sector_raw = "__default__"  # Default value for invalid or empty sector

# Debugging: log the sector_raw value
st.write(f"Sector raw: {sector_raw}")  # Display sector for debugging purposes

# ------ Define realistic sector-specific crisis scenarios --------
# Define realistic sector-specific crisis scenarios
SECTOR_CRISIS_SCENARIOS = {
    "Technology": {
        "Tech Crunch": {
            "Revenue_CAGR_3Y": 0.70,  # Impacted by market slowdowns
            "ROA": 0.60,              # Decreased profitability
            "Net_Profit_Margin": 0.50, # Lower margins
            "Interest_Coverage": 0.70,
            "Debt_to_Equity": 1.05,
            "EBITDA_to_Interest": 0.65,
            "Sentiment Score": 0.70    # Sentiment downturn
        },
        "Supply Chain Disruption": {
            "Revenue_CAGR_3Y": 0.80,
            "ROA": 0.75,
            "Net_Profit_Margin": 0.70,
            "Interest_Coverage": 0.75,
            "Debt_to_Equity": 1.10,
            "EBITDA_to_Interest": 0.75,
            "Sentiment Score": 0.75
        },
        "Pandemic Shock": {
            "Revenue_CAGR_3Y": 0.60,
            "ROA": 0.55,
            "Net_Profit_Margin": 0.50,
            "Interest_Coverage": 0.60,
            "Debt_to_Equity": 1.15,
            "EBITDA_to_Interest": 0.60,
            "Sentiment Score": 0.65
        }
    },
    "Real Estate": {
        "Housing Downturn": {
            "Revenue_CAGR_3Y": 0.75,  # Impact of lower sales
            "ROA": 0.65,              # Profitability decrease
            "Net_Profit_Margin": 0.60,
            "Interest_Coverage": 0.70,
            "Debt_to_Equity": 1.30,
            "EBITDA_to_Interest": 0.80,
            "Sentiment Score": 0.60
        },
        "Credit Crunch": {
            "Revenue_CAGR_3Y": 0.70,
            "ROA": 0.60,
            "Net_Profit_Margin": 0.55,
            "Interest_Coverage": 0.65,
            "Debt_to_Equity": 1.25,
            "EBITDA_to_Interest": 0.75,
            "Sentiment Score": 0.65
        },
        "Government Policy Change": {
            "Revenue_CAGR_3Y": 0.80,
            "ROA": 0.70,
            "Net_Profit_Margin": 0.65,
            "Interest_Coverage": 0.75,
            "Debt_to_Equity": 1.20,
            "EBITDA_to_Interest": 0.85,
            "Sentiment Score": 0.70
        }
    },
    # Add more specific scenarios for other sectors here

    # Default fallback for sectors not defined in the dictionary
    "__default__": {
        "General Crisis": {
            "Revenue_CAGR_3Y": 0.70,
            "ROA": 0.65,
            "Net_Profit_Margin": 0.60,
            "Interest_Coverage": 0.65,
            "Debt_to_Equity": 1.20,
            "EBITDA_to_Interest": 0.70,
            "Sentiment Score": 0.60
        }
    }
}

# -------------------------------------------------------------
# Select appropriate sector-based stress scenarios based on user input
def build_sector_scenarios(sector_name: str) -> dict:
    if sector_name in SECTOR_CRISIS_SCENARIOS:
        return SECTOR_CRISIS_SCENARIOS[sector_name]
    else:
        # Default to general crisis scenarios if sector is unknown
        return SECTOR_CRISIS_SCENARIOS["__default__"]

# -------------------------------------------------------------
# Run stress test and apply dynamic factors to the model input
sector_scenarios = build_sector_scenarios(sector_raw)

# Scale the crisis multipliers based on selected severity and exchange intensity
sector_scenarios_scaled = {scenario: scale_multiplier(factor, sev, ex_intensity) for scenario, factor in sector_scenarios.items()}

# Run the scenario test and get PD values
df_sector = run_scenarios(model, X_base_row, sector_scenarios_scaled)

# Plot the sector scenario results
if not df_sector.empty:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_sector["Scenario"], y=df_sector["PD"]))
    fig.update_layout(title=f"Sector Crisis Impact — {sector_raw}", yaxis=dict(tickformat=".0%"), height=340)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No sector scenarios generated results.")
    
# ---------- Monte Carlo CVaR ----------
st.markdown("**Monte Carlo CVaR 95%**")

mc_results = mc_cvar_pd(model, X_base_row, feats_df, sims=sim_count, alpha=0.95)

if isinstance(mc_results, dict) and "PD_sims" in mc_results:
    pd_var = mc_results["VaR"]
    pd_cvar = mc_results["CVaR"]
    st.metric("VaR 95% (PD)", f"{pd_var:.2%}")
    st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}")
else:
    st.warning("Monte Carlo CVaR simulation failed.")
