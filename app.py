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

# ===================== D) Stress Testing (Real-world, sector-aware) =====================
st.subheader("D. Stress Testing — Real-world Crises by Sector")

# ---------- Sector normalization (broad coverage) ----------
def sector_normalize(s: str) -> str:
    x = (s or "").lower()
    # broad buckets (GICS-like + VN context)
    if any(k in x for k in ["tech", "software", "it", "semiconductor", "internet"]): return "Technology"
    if any(k in x for k in ["telecom", "communication services", "telco"]): return "Telecom"
    if any(k in x for k in ["bank", "securit", "insur", "financial"]): return "Financials"
    if any(k in x for k in ["real estate", "property", "construction", "developer"]): return "Real Estate"
    if any(k in x for k in ["steel", "material", "cement", "mining", "basic resources", "chem"]): return "Materials"
    if any(k in x for k in ["energy", "oil", "gas", "coal", "refining", "power gen"]): return "Energy"
    if any(k in x for k in ["industrial", "manufacturing", "machinery", "aviation", "aerospace"]): return "Industrials"
    if any(k in x for k in ["consumer discretionary", "retail", "auto", "apparel", "electronics retail"]): return "Consumer Discretionary"
    if any(k in x for k in ["consumer staples", "food", "beverage", "household", "staple"]): return "Consumer Staples"
    if any(k in x for k in ["health", "pharma", "biotech", "medical"]): return "Healthcare"
    if any(k in x for k in ["utility", "water", "electric", "gas util"]): return "Utilities"
    if any(k in x for k in ["transport", "airline", "airport", "shipping", "logistics"]): return "Transportation"
    if any(k in x for k in ["hotel", "travel", "tourism", "hospitality", "leisure"]): return "Hospitality & Travel"
    if any(k in x for k in ["agri", "fisher", "aquaculture", "seafood", "fishery"]): return "Agriculture & Fisheries"
    if any(k in x for k in ["auto", "oem", "components"]): return "Automotive"
    return "__default__"

sector_bucket = sector_normalize(sector_raw)

# ---------- Crisis Catalog (evidence-informed) ----------
# Multipliers are multiplicative shocks to features (1.0 = no change).
# Naming of features follows your engineered set; if missing -> safely ignored.
CRISIS_CATALOG = {
    "COVID-19 Pandemic (2020-2021)": {
        "when_hint": [2020, 2021],
        "desc": "Mobility collapse, services shutdown; airlines/hospitality hit nhất; dầu cầu giảm mạnh; supply/demand lệch.",
        "base_impacts": {  # general direction
            # profitability & margins
            "Net_Profit_Margin": 0.85, "Gross_Margin": 0.90, "ROA": 0.85, "ROE": 0.85,
            # coverage & liquidity
            "Interest_Coverage": 0.70, "EBITDA_to_Interest": 0.75, "Current_Ratio": 0.95, "Quick_Ratio": 0.95,
            # leverage/solvency (risk up)
            "Debt_to_Assets": 1.05, "Debt_to_Equity": 1.05, "Total_Debt_to_EBITDA": 1.15,
            # ops cycles
            "Asset_Turnover": 0.90, "Receivables_Turnover": 0.90, "Inventory_Turnover": 0.85,
            # top-line trajectory proxy
            "Revenue_CAGR_3Y": 0.85
        },
        "sector_sensitivity": {
            "Transportation": 1.8, "Hospitality & Travel": 1.8, "Energy": 1.3,
            "Consumer Discretionary": 1.2, "Industrials": 1.2, "__default__": 1.0
        }
    },
    "Global Financial Crisis (2008-2009)": {
        "when_hint": [2008, 2009],
        "desc": "Credit crunch, bất động sản & tài chính chịu tác động sâu; đầu tư & tín dụng co hẹp.",
        "base_impacts": {
            "Net_Profit_Margin": 0.90, "ROA": 0.90, "ROE": 0.88,
            "Interest_Coverage": 0.75, "EBITDA_to_Interest": 0.80,
            "Current_Ratio": 0.95, "Quick_Ratio": 0.95,
            "Debt_to_Assets": 1.10, "Operating_Income_to_Debt": 0.85,
            "Revenue_CAGR_3Y": 0.90
        },
        "sector_sensitivity": {
            "Financials": 1.6, "Real Estate": 1.5, "Materials": 1.2, "Industrials": 1.2, "__default__": 1.0
        }
    },
    "US–China Tariffs (2018–2019; 2025 updates)": {
        "when_hint": [2018, 2019, 2025],
        "desc": "Thuế Section 232/301; kim loại, linh kiện điện tử, dệt may, máy móc… chịu ma sát chi phí & cầu.",
        "base_impacts": {
            "Gross_Margin": 0.95, "Net_Profit_Margin": 0.93,
            "Asset_Turnover": 0.97, "Receivables_Turnover": 0.95, "Inventory_Turnover": 0.93,
            "Debt_to_Assets": 1.05, "Debt_to_Equity": 1.05
        },
        "sector_sensitivity": {
            "Materials": 1.5, "Technology": 1.2, "Consumer Discretionary": 1.2, "Industrials": 1.2, "__default__": 1.0
        }
    },
    "Energy Price Shock (Europe 2022)": {
        "when_hint": [2022],
        "desc": "Giá năng lượng tăng kỷ lục; chi phí đầu vào sản xuất/tiêu dùng tăng; margin & thanh khoản căng.",
        "base_impacts": {
            "Gross_Margin": 0.92, "Net_Profit_Margin": 0.90, "ROA": 0.92,
            "Current_Ratio": 0.93, "Quick_Ratio": 0.93,
            "Interest_Coverage": 0.85, "EBITDA_to_Interest": 0.88
        },
        "sector_sensitivity": {
            "Materials": 1.3, "Industrials": 1.3, "Consumer Staples": 1.2, "Utilities": 1.2, "__default__": 1.0
        }
    },
    "Supply Chain Disruptions (2021–2022)": {
        "when_hint": [2021, 2022],
        "desc": "Thiếu container, cảng tắc nghẽn, thiếu chip; hàng tồn tăng, vòng quay giảm, doanh thu trì trệ.",
        "base_impacts": {
            "Inventory_Turnover": 0.85, "Receivables_Turnover": 0.92, "Asset_Turnover": 0.93,
            "Revenue_CAGR_3Y": 0.92, "Gross_Margin": 0.95
        },
        "sector_sensitivity": {
            "Technology": 1.3, "Automotive": 1.4, "Industrials": 1.2, "Consumer Discretionary": 1.2, "__default__": 1.0
        }
    },
    "Tech Valuation Reset (2022–2023)": {
        "when_hint": [2022, 2023],
        "desc": "Lãi suất tăng nhanh → compress multiples; tăng chi phí vốn; tăng tiết giảm R&D/S&M.",
        "base_impacts": {
            "Net_Profit_Margin": 0.95, "ROE": 0.93, "ROA": 0.95,
            "Interest_Coverage": 0.90, "EBITDA_to_Interest": 0.92
        },
        "sector_sensitivity": {"Technology": 1.5, "__default__": 1.0}
    },
    "Oil Demand Crash (2020)": {
        "when_hint": [2020],
        "desc": "Cầu dầu giảm ~9.3 mb/d yoy; refining/energy upstream/downstream biến động mạnh.",
        "base_impacts": {
            "Gross_Margin": 0.92, "Net_Profit_Margin": 0.88, "ROA": 0.90, "ROE": 0.90,
            "Interest_Coverage": 0.80, "EBITDA_to_Interest": 0.82
        },
        "sector_sensitivity": {"Energy": 1.6, "Transportation": 1.3, "__default__": 1.0}
    }
}

# ---------- Helpers to build scenario multipliers ----------
RISK_UP = {"Debt_to_Assets","Debt_to_Equity","Total_Debt_to_EBITDA","Net_Debt_to_Equity","Long_Term_Debt_to_Assets"}
HEALTH_DOWN = {"ROA","ROE","Current_Ratio","Quick_Ratio","Interest_Coverage","EBITDA_to_Interest",
               "Operating_Income_to_Debt","Asset_Turnover","Receivables_Turnover","Inventory_Turnover",
               "Net_Profit_Margin","Gross_Margin","Revenue_CAGR_3Y"}

def combine_impacts(base_map: dict, sens: float, severity: float) -> dict:
    """Scale base impacts by sector sensitivity and user severity. Impacts are multiplicative multipliers."""
    out = {}
    for k, v in base_map.items():
        # Convert base v toward 1.0 depending on sens*severity:
        # If v<1 (bad for health metrics) → exponentiate the distance to 1.0
        # If v>1 (bad for risk-up metrics) → likewise.
        if v == 1.0:
            out[k] = 1.0
            continue
        dist = (v - 1.0)
        out[k] = 1.0 + dist * sens * severity
        # Keep within sane bounds
        out[k] = float(np.clip(out[k], 0.5, 1.8))
    return out

def apply_feature_multipliers(Xrow: pd.DataFrame, fmap: dict) -> pd.DataFrame:
    X = Xrow.copy()
    for f, mult in fmap.items():
        if f in X.columns and pd.notna(X.at[X.index[0], f]):
            try:
                X.at[X.index[0], f] = float(X.at[X.index[0], f]) * float(mult)
            except Exception:
                pass
    return X

def run_crisis_pd(model, Xrow: pd.DataFrame, crisis_name: str, sector_bucket: str, severity: float) -> tuple:
    c = CRISIS_CATALOG.get(crisis_name, None)
    if not c:
        return crisis_name, float("nan")
    base = c["base_impacts"]
    sens = c["sector_sensitivity"].get(sector_bucket, c["sector_sensitivity"].get("__default__", 1.0))
    fmap = combine_impacts(base, sens, severity)
    Xs = apply_feature_multipliers(Xrow, fmap)
    Xs = align_features_to_model(Xs, model)
    if hasattr(model, "predict_proba"):
        p = float(model.predict_proba(Xs)[:,1][0])
    else:
        p = float(model.predict(Xs)[0])
    return crisis_name, p

# ---------- Build UI controls ----------
st.markdown("**Choose crisis & severity**")
c1, c2, c3 = st.columns([2,1,1])
with c1:
    crisis_names = list(CRISIS_CATALOG.keys())
    # Suggest a crisis by Year hint
    default_idx = 0
    for i, name in enumerate(crisis_names):
        years_hint = CRISIS_CATALOG[name].get("when_hint", [])
        if years_hint and (int(year) in years_hint):
            default_idx = i
            break
    crisis = st.selectbox("Crisis", crisis_names, index=default_idx)
with c2:
    severity = st.slider("Severity (×)", min_value=0.5, max_value=2.0, value=1.0, step=0.05,
                         help="Scale the magnitude of shocks vs. catalog baseline.")
with c3:
    st.write(" ")
    st.caption(f"Sector bucket: **{sector_bucket}**")

# ---------- Compute PD under selected crisis ----------
X_base_comm = X_base.copy()  # fix prior undefined
sel_name, sel_pd = run_crisis_pd(model, X_base_comm, crisis, sector_bucket, severity)

# Also compute a dashboard for all crises at default severity=1
rows_all = []
for nm in CRISIS_CATALOG.keys():
    nm_, p_ = run_crisis_pd(model, X_base_comm, nm, sector_bucket, 1.0)
    rows_all.append({"Crisis": nm_, "PD": p_})
df_all_crises = pd.DataFrame(rows_all).sort_values("PD", ascending=False)

# ---------- Monte Carlo CVaR (using train reference if available) ----------
ref_df = load_train_reference()
try:
    ref_df = ref_df if isinstance(ref_df, pd.DataFrame) else feats_df  # fallback
    mc = mc_cvar_pd(model, X_base_comm, ref_df, sims=4000, alpha=0.95, clip_q=(0.01,0.99))
    pd_var = float(mc["VaR"]); pd_cvar = float(mc["CVaR"])
except Exception:
    mc = {"PD_sims": np.array([])}; pd_var = np.nan; pd_cvar = np.nan

# ---------- Render charts ----------
cA, cB = st.columns([1.2, 1])
with cA:
    st.markdown(f"**Selected crisis:** {sel_name}")
    st.metric("PD under selected crisis", f"{sel_pd:.2%}")
    if not df_all_crises.empty:
        figC = go.Figure()
        figC.add_trace(go.Bar(x=df_all_crises["Crisis"], y=df_all_crises["PD"]))
        figC.update_layout(title=f"PD under crises — sector: {sector_bucket}",
                           yaxis=dict(tickformat=".0%"), height=360, margin=dict(l=10,r=10,t=40,b=90))
        st.plotly_chart(figC, use_container_width=True)
    else:
        st.info("No crisis PDs computed.")

with cB:
    st.markdown("**Monte Carlo tail risk (95%)**")
    if isinstance(mc.get("PD_sims"), np.ndarray) and mc["PD_sims"].size:
        hist = np.histogram(mc["PD_sims"], bins=40)
        centers = (hist[1][1:] + hist[1][:-1]) / 2
        figT = go.Figure()
        figT.add_trace(go.Bar(x=centers, y=hist[0]))
        figT.add_vline(x=pd_var, line_width=2, line_dash="dash", line_color="red")
        figT.add_vline(x=pd_cvar, line_width=2, line_dash="dot", line_color="black")
        figT.update_layout(title="PD sims (VaR 95% red, CVaR 95% black)",
                           xaxis_title="PD", yaxis_title="Freq", height=300, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(figT, use_container_width=True)
    else:
        st.info("Monte Carlo distribution unavailable.")

# ---------- Small KPI panel ----------
k1, k2, k3 = st.columns(3)
with k1: st.metric("Max PD across crises", f"{df_all_crises['PD'].max():.2%}" if not df_all_crises.empty else "-")
with k2: st.metric("VaR 95% (PD)", f"{pd_var:.2%}" if np.isfinite(pd_var) else "-")
with k3: st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}" if np.isfinite(pd_cvar) else "-")

# ---------- Disclosure ----------
st.caption("Notes: Crisis impacts are multiplicative feature shocks calibrated from sector-level evidence "
           "and can be tuned via Severity. Sector mapping is broad; unknown sectors fall back to '__default__'.")
