# app.py — Corporate Default Risk Scoring (Single-page, Bank-grade)
# ----------------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==== Your utils (must exist in repo) ====
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

# ===================== Stress testing math (self-contained) =====================
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

# Robust board mapping
EXCHANGE_INTENSITY = {"UPCOM": 0.6, "HNX": 1.0, "HOSE": 1.0, "HSX": 1.0}

RISK_UP = {"Debt_to_Assets","Debt_to_Equity","Total_Debt_to_EBITDA","Net_Debt_to_Equity","Long_Term_Debt_to_Assets"}
RISK_DOWN = {"ROA","ROE","Current_Ratio","Quick_Ratio","Operating_Income_to_Debt"}

def apply_factor_map_once(Xrow: pd.DataFrame, factor: dict) -> pd.DataFrame:
    X = Xrow.copy()
    for f, mult in factor.items():
        if f in X.columns:
            X[f] = float(X[f].iloc[0]) * float(mult)
    return X

def run_factor_scenarios(model, Xrow_comm: pd.DataFrame, factors: dict) -> pd.DataFrame:
    rows = []
    for name, fmap in factors.items():
        Xs = apply_factor_map_once(Xrow_comm, fmap)
        Xs = align_features_to_model(Xs, model)
        pd_val = float(model.predict_proba(Xs)[:,1][0]) if hasattr(model, "predict_proba") else float(model.predict(Xs)[0])
        rows.append({"Scenario": name, "PD": pd_val})
    return pd.DataFrame(rows)

def shrink_cov(cov: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    d = np.diag(np.diag(cov))
    shrunk = (1 - alpha) * cov + alpha * d
    w, V = np.linalg.eigh(shrunk)
    w = np.clip(w, 1e-8, None)
    return (V * w) @ V.T

def mc_cvar_pd(model, Xrow: pd.DataFrame, reference_df: pd.DataFrame,
               sims: int = 5000, alpha: float = 0.95, clip_q=(0.01,0.99)) -> dict:
    assert Xrow.shape[0] == 1
    cols = list(Xrow.columns)
    ref = reference_df[cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    base = Xrow[cols].values.reshape(1,-1).astype(float)[0]
    cov = np.cov(ref.values.T)
    if not np.all(np.isfinite(cov)): cov = np.nan_to_num(cov, nan=0.0)
    cov = shrink_cov(cov, alpha=0.15)
    sims_mat = np.random.multivariate_normal(mean=base, cov=cov, size=sims)
    ql = ref.quantile(clip_q[0], numeric_only=True).values
    qh = ref.quantile(clip_q[1], numeric_only=True).values
    sims_mat = np.minimum(np.maximum(sims_mat, ql), qh)
    X = pd.DataFrame(sims_mat, columns=cols)
    X = align_features_to_model(force_numeric(X), model)
    if hasattr(model, "predict_proba"):
        pd_sims = model.predict_proba(X)[:,1]
    else:
        pd_sims = model.predict(X).astype(float)
    var = float(np.quantile(pd_sims, alpha))
    cvar = float(pd_sims[pd_sims >= var].mean()) if (pd_sims >= var).any() else var
    return {"PD_sims": pd_sims, "VaR": var, "CVaR": cvar}

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
st.caption("English UI • Single page • LightGBM scoring • SHAP explainability • Sector vs Systemic stress • Monte Carlo CVaR")

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

# --- Sector bucket chuẩn hoá dùng thống nhất ---
def _normalize_sector_bucket(s: str) -> str:
    x = (s or "").lower()
    if any(k in x for k in ["tech","software","it","semiconductor","internet"]): return "Technology"
    if any(k in x for k in ["telecom","communication services","telco"]): return "Telecom"
    if any(k in x for k in ["bank","insur","securit","financial"]): return "Financials"
    if any(k in x for k in ["real estate","property","construction","developer"]): return "Real Estate"
    if any(k in x for k in ["steel","material","cement","mining","basic res","chem"]): return "Materials"
    if any(k in x for k in ["energy","oil","gas","coal","refining","petro","power gen"]): return "Energy"
    if any(k in x for k in ["industrial","manufacturing","machinery","aviation","aerospace"]): return "Industrials"
    if any(k in x for k in ["consumer discretionary","retail","auto","apparel","electronics retail"]): return "Consumer Discretionary"
    if any(k in x for k in ["consumer staples","food","beverage","household","staple"]): return "Consumer Staples"
    if any(k in x for k in ["health","pharma","biotech","medical"]): return "Healthcare"
    if any(k in x for k in ["utility","water","electric","gas util"]): return "Utilities"
    if any(k in x for k in ["transport","airline","airport","shipping","logistics"]): return "Transportation"
    if any(k in x for k in ["hotel","travel","tourism","hospitality","leisure"]): return "Hospitality & Travel"
    if any(k in x for k in ["agri","fisher","aquaculture","seafood","fishery"]): return "Agriculture & Fisheries"
    if any(k in x for k in ["auto","oem","components"]): return "Automotive"
    return "__default__"

sector_bucket = _normalize_sector_bucket(sector_raw)

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
# interest_exp_raw removed from metrics (not needed)
cash_raw = get_raw(["Cash and cash equivalents (Bn. VND)","Cash"], 0.0)
receivables_raw = get_raw(["Accounts receivable (Bn. VND)","Receivables"], 0.0)
inventories_raw = get_raw(["Net Inventories","Inventories"], 0.0)
current_assets_raw = get_raw(["CURRENT ASSETS (Bn. VND)","Current_Assets"], 0.0)

# ---- Debt & ratios (bounded presentation) ----
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
if pd.notna(dta): dta = min(max(dta, 0.0), 0.999)

dte = safe_div(debt_raw, equity_raw)
if pd.notna(dte): dte = min(max(dte, 0.0), 0.999)

current_ratio = safe_div(current_assets_raw, curr_liab)
quick_ratio   = safe_div((cash_raw or 0.0) + (receivables_raw or 0.0), curr_liab)

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
        st.plotly_chart(fig_rev, width='stretch')
    else:
        st.info("No historical series for this company.")

with col2:
    fig_cap = go.Figure(data=[go.Pie(labels=["Total Debt","Equity"], values=[debt_raw, equity_raw], hole=0.5)])
    fig_cap.update_layout(title="Capital Structure", height=380)
    st.plotly_chart(fig_cap, width='stretch')

st.markdown("### Key Financial Ratios")
key_ratios = pd.DataFrame({
    "Metric": ["ROA","ROE","Debt_to_Assets","Debt_to_Equity","Current_Ratio","Quick_Ratio"],
    "Value": [roa, roe, dta, dte, current_ratio, quick_ratio]
})
key_ratios["Value"] = key_ratios["Value"].apply(fmt_ratio)
st.dataframe(key_ratios, use_container_width=True, hide_index=True)  # dataframe UI vẫn OK

# ===================== B) Default Probability (PD) & Policy (Multi-factor) =====================
st.subheader("B. Default Probability (PD) & Policy Band")

def _sigmoid(z):
    z = float(z)
    if z >= 35: return 1.0
    if z <= -35: return 0.0
    return 1.0 / (1.0 + np.exp(-z))

def _logit(p, eps=1e-9):
    p = float(np.clip(p, eps, 1 - eps))
    return np.log(p / (1 - p))

def _safe(v):
    try:
        return float(v) if pd.notna(v) else np.nan
    except Exception:
        return np.nan

# 1) PD gốc từ model
if hasattr(model, "predict_proba"):
    pd_base = float(model.predict_proba(X_base)[:, 1][0])
else:
    pd_base = float(model.predict(X_base)[0])

# 2) Cấu hình hậu-hiệu chỉnh PD (có thể đặt file models/pd_adjustments.json để override)
DEFAULT_PD_CFG = {
    "exchange_logit_mult": {"UPCOM": 0.40, "HNX": 0.18, "HOSE": 0.00, "HSX": 0.00, "__default__": 0.10},
    "size": {"assets_q40": 0.20, "revenue_q40": 0.10},
    "leverage": {"dta_hi": 0.25, "dte_hi": 0.20, "netde_hi": 0.15},
    "profitability": {"roa_neg": 0.25, "roe_neg": 0.20, "npm_neg": 0.15, "rev_cagr_neg": 0.10},
    "liquidity": {"cr_low": 0.12, "qr_low": 0.08},
    "governance": {"auditor_non_big4": 0.10, "opinion_qualified": 0.30, "filing_delay": 0.12},
    # sector tilt (nâng cấp như bạn gợi ý)
    "sector_tilt": {
        "Real Estate": 0.20, "Materials": 0.10, "Consumer Discretionary": 0.05,
        "Financials": 0.00, "Utilities": -0.05, "Technology": -0.02, "__default__": 0.00
    },
    "pd_floor": {"UPCOM": 0.06, "HNX": 0.04, "HOSE": 0.02, "HSX": 0.02, "__default__": 0.03},
    "pd_cap":   {"default": 0.60}
}

def load_pd_cfg(path="models/pd_adjustments.json"):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        pass
    return DEFAULT_PD_CFG

PD_CFG = load_pd_cfg()

# 3) Tín hiệu rủi ro từ row (không dùng Interest_Coverage/EBITDA_to_Interest)
def _from_row(series, keys, default=np.nan):
    for k in keys:
        if k in series.index and pd.notna(series.get(k)):
            return _safe(series.get(k))
    return default

npm = _from_row(row_model, ["Net_Profit_Margin","net_profit_margin"])
rev_cagr3y = _from_row(row_model, ["Revenue_CAGR_3Y","revenue_cagr_3y","sales_cagr_3y"])
nde = _from_row(row_model, ["Net_Debt_to_Equity","net_debt_to_equity"])
auditor = str(_from_row(row_raw, ["Auditor","Audit_Firm","Auditor_Name"], "") or "")
opinion = str(_from_row(row_raw, ["Audit_Opinion","Opinion"], "") or "")
filing_delay = _from_row(row_raw, ["Filing_Delay_Days","Filing_Delay"], np.nan)

# 4) Phân vị làm mốc size từ tập train
ref = load_train_reference()
ref_use = ref if isinstance(ref, pd.DataFrame) else feats_df

def _q(col, q, fallback=np.nan):
    if (col in ref_use.columns) and ref_use[col].notna().any():
        try: return float(ref_use[col].quantile(q))
        except Exception: return fallback
    return fallback

assets_q40 = _q("Total_Assets", 0.40, np.nan) if "Total_Assets" in ref_use.columns else np.nan
revenue_q40 = _q("Revenue", 0.40, np.nan) if "Revenue" in ref_use.columns else np.nan

flags = {
    "exch_mult": PD_CFG["exchange_logit_mult"].get(exchange, PD_CFG["exchange_logit_mult"]["__default__"]),
    "assets_q40": (assets_raw is not None and np.isfinite(_safe(assets_raw)) and np.isfinite(assets_q40) and float(assets_raw) < assets_q40),
    "revenue_q40": (revenue_raw is not None and np.isfinite(_safe(revenue_raw)) and np.isfinite(revenue_q40) and float(revenue_raw) < revenue_q40),
    "dta_hi": (isinstance(dta, float) and dta > 0.65),
    "dte_hi": (isinstance(dte, float) and dte > 1.0),
    "netde_hi": (isinstance(nde, float) and nde > 0.8),
    "roa_neg": (isinstance(roa, float) and roa < 0.0),
    "roe_neg": (isinstance(roe, float) and roe < 0.0),
    "npm_neg": (isinstance(npm, float) and npm < 0.0),
    "rev_cagr_neg": (isinstance(rev_cagr3y, float) and rev_cagr3y < 0.0),
    "cr_low": (isinstance(current_ratio, float) and current_ratio < 1.0),
    "qr_low": (isinstance(quick_ratio, float) and quick_ratio < 0.8),
    "auditor_non_big4": (auditor != "" and not any(k in auditor.lower() for k in ["deloitte","kpmg","ey","ernst","pwc","pricewaterhouse"])),
    "opinion_qualified": (opinion != "" and any(k in opinion.lower() for k in ["qualified","adverse","disclaimer"])),
    "filing_delay": (isinstance(filing_delay, float) and filing_delay >= 20),
}

logit0 = _logit(pd_base)
adj = 0.0
# Exchange premium
adj += flags["exch_mult"]
# Sector tilt — ưu tiên bucket (nếu không có thì dùng alias, rồi default)
adj += PD_CFG["sector_tilt"].get(sector_bucket, PD_CFG["sector_tilt"].get(sector_alias, PD_CFG["sector_tilt"]["__default__"]))
# Size
if flags["assets_q40"]:  adj += PD_CFG["size"]["assets_q40"]
if flags["revenue_q40"]: adj += PD_CFG["size"]["revenue_q40"]
# Leverage
if flags["dta_hi"]:  adj += PD_CFG["leverage"]["dta_hi"]
if flags["dte_hi"]:  adj += PD_CFG["leverage"]["dte_hi"]
if flags["netde_hi"]: adj += PD_CFG["leverage"]["netde_hi"]
# Profitability & growth
if flags["roa_neg"]:       adj += PD_CFG["profitability"]["roa_neg"]
if flags["roe_neg"]:       adj += PD_CFG["profitability"]["roe_neg"]
if flags["npm_neg"]:       adj += PD_CFG["profitability"]["npm_neg"]
if flags["rev_cagr_neg"]:  adj += PD_CFG["profitability"]["rev_cagr_neg"]
# Liquidity
if flags["cr_low"]: adj += PD_CFG["liquidity"]["cr_low"]
if flags["qr_low"]: adj += PD_CFG["liquidity"]["qr_low"]
# Governance
if flags["auditor_non_big4"]:  adj += PD_CFG["governance"]["auditor_non_big4"]
if flags["opinion_qualified"]: adj += PD_CFG["governance"]["opinion_qualified"]
if flags["filing_delay"]:      adj += PD_CFG["governance"]["filing_delay"]

pd_adj = _sigmoid(logit0 + adj)
pd_floor = PD_CFG["pd_floor"].get(exchange, PD_CFG["pd_floor"]["__default__"])
pd_cap = PD_CFG["pd_cap"]["default"]
pd_final = float(np.clip(pd_adj, pd_floor, pd_cap))

thr = thresholds_for_sector(load_thresholds("models/threshold.json"), sector_raw)
band = classify_pd(pd_final, thr)

c1,c2,c3 = st.columns([1,1,2])
with c1: st.metric("PD (multi-factor adj.)", f"{pd_final:.2%}")
with c2: st.metric("Policy Band", band)
with c3:
    st.markdown(
        f"<span class='small'>Policy: Low &lt; {thr['low']:.0%} • Medium &lt; {thr['medium']:.0%} • "
        f"Floor/Cap: {pd_floor:.0%}/{pd_cap:.0%} • Exchange: {exchange or '-'}</span>", unsafe_allow_html=True
    )

fig_g = go.Figure(go.Indicator(
    mode="gauge+number", value=pd_final*100, number={'suffix': "%"},
    gauge={'axis': {'range': [0,100]},
           'bar': {'color': '#1f77b4'},
           'steps': [{'range':[0,thr['low']*100],'color':'#E8F1FB'},
                     {'range':[thr['low']*100,thr['medium']*100],'color':'#CFE3F7'},
                     {'range':[thr['medium']*100,100],'color':'#F9E3E3'}],
           'threshold': {'line': {'color':'red','width':3},'value':pd_final*100}}
))
fig_g.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig_g, width='stretch')

# ===================== C) Model Explainability (SHAP) — nice chart =====================
st.subheader("C. Model Explainability (SHAP)")

try:
    shap_df = explain_shap(model, X_base, top_n=10)
    if "Impact" in shap_df.columns and "SHAP" not in shap_df.columns:
        shap_df = shap_df.rename(columns={"Impact":"SHAP"})
except Exception:
    shap_df = pd.DataFrame()

DISPLAY_LABEL = {
    "roa": "ROA (Return on Assets)", "roa_ratio": "ROA (Return on Assets)",
    "roe": "ROE (Return on Equity)", "roe_ratio": "ROE (Return on Equity)",
    "debt_to_assets": "Debt / Assets", "debt_assets_ratio": "Debt / Assets",
    "debt_to_equity": "Debt / Equity", "de_ratio": "Debt / Equity",
    "gross_margin": "Gross Margin", "net_profit_margin": "Net Profit Margin",
    "current_ratio": "Current Ratio", "quick_ratio": "Quick Ratio",
    "asset_turnover": "Asset Turnover", "inventory_turnover": "Inventory Turnover",
    "receivables_turnover": "Receivables Turnover",
    "revenue_cagr_3y": "Revenue CAGR (3Y)",
}
def beautify_label(x: str) -> str:
    key = (x or "").strip()
    low = key.lower()
    return DISPLAY_LABEL.get(low, key)

if shap_df.empty:
    st.info("SHAP is not available for this model/input.")
else:
    shap_df["absSHAP"] = shap_df["SHAP"].abs()
    shap_df = shap_df.sort_values("absSHAP", ascending=True).tail(10)
    shap_df["Label"] = shap_df["Feature"].apply(beautify_label)

    fig_sh = go.Figure()
    colors = ["#E24A33" if v < 0 else "#1F77B4" for v in shap_df["SHAP"]]
    fig_sh.add_trace(go.Bar(
        x=shap_df["SHAP"], y=shap_df["Label"],
        orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in shap_df["SHAP"]],
        textposition="outside"
    ))
    fig_sh.update_layout(
        title="Top Feature Contributions (SHAP)",
        xaxis=dict(title="SHAP value → PD"),
        height=420, margin=dict(l=10, r=20, t=40, b=10)
    )
    st.plotly_chart(fig_sh, width='stretch')

# ===================== D) Stress Testing — 4 panels (Sector vs Systemic) =====================
st.subheader("D. Stress Testing — Sector & Systemic Impacts")

# --- Feature alias resolver (đảm bảo multiplier chạm đúng cột mô hình) ---
FEATURE_ALIAS = {
    "ROA": ["roa","roa_ratio","return_on_assets"],
    "ROE": ["roe","roe_ratio","return_on_equity"],
    "Gross_Margin": ["gross_margin","gm","gross_margin_ratio"],
    "Net_Profit_Margin": ["net_profit_margin","npm","net_margin"],
    "Debt_to_Assets": ["debt_to_assets","debt_assets_ratio"],
    "Debt_to_Equity": ["debt_to_equity","dte","de_ratio"],
    "Total_Debt_to_EBITDA": ["debt_to_ebitda","total_debt_ebitda"],
    "Operating_Income_to_Debt": ["operating_income_to_debt","op_income_debt"],
    "Asset_Turnover": ["asset_turnover","at"],
    "Receivables_Turnover": ["receivables_turnover","rt"],
    "Inventory_Turnover": ["inventory_turnover","it"],
    "Revenue_CAGR_3Y": ["revenue_cagr_3y","sales_cagr_3y","rev_cagr3y"],
    "Current_Ratio": ["current_ratio","cr"],
    "Quick_Ratio": ["quick_ratio","qr"],
}

def resolve_feature_alias(X: pd.DataFrame, name: str):
    if name in X.columns: return name
    for alt in FEATURE_ALIAS.get(name, []):
        if alt in X.columns: return alt
    return None

def apply_feature_multipliers_alias(Xrow: pd.DataFrame, fmap: dict) -> pd.DataFrame:
    X = Xrow.copy()
    for f, mult in fmap.items():
        col = resolve_feature_alias(X, f)
        if col is None: 
            continue
        try:
            X.at[X.index[0], col] = float(X.at[X.index[0], col]) * float(mult)
        except Exception:
            pass
    return X

def combine_impacts(base_map: dict, sens: float = 1.0, severity: float = 1.0) -> dict:
    out = {}
    for k, v in base_map.items():
        if v == 1.0: out[k] = 1.0
        else: out[k] = float(np.clip(1.0 + (v - 1.0) * sens * severity, 0.5, 1.8))
    return out

def _pd_from_row(model, Xrow: pd.DataFrame) -> float:
    Xr = align_features_to_model(Xrow, model)
    if hasattr(model, "predict_proba"): return float(model.predict_proba(Xr)[:,1][0])
    return float(model.predict(Xr)[0])

# --- Catalogs (đÃ LOẠI Interest_Coverage / EBITDA_to_Interest) ---
SECTOR_CRISIS_CATALOG = {
    "COVID-19 Pandemic (2020–2021)": {
        "base_impacts": {
            "Net_Profit_Margin": 0.85, "Gross_Margin": 0.90, "ROA": 0.85, "ROE": 0.85,
            "Current_Ratio": 0.95, "Quick_Ratio": 0.95,
            "Debt_to_Assets": 1.05, "Debt_to_Equity": 1.05, "Total_Debt_to_EBITDA": 1.15,
            "Asset_Turnover": 0.90, "Receivables_Turnover": 0.90, "Inventory_Turnover": 0.85,
            "Revenue_CAGR_3Y": 0.85
        },
        "sector_sensitivity": {"Transportation":1.8,"Hospitality & Travel":1.8,"Energy":1.3,
                               "Consumer Discretionary":1.2,"Industrials":1.2,"__default__":1.0}
    },
    "US–China Tariffs (2018–2019; 2025 updates)": {
        "base_impacts": {
            "Gross_Margin": 0.95, "Net_Profit_Margin": 0.93,
            "Asset_Turnover": 0.97, "Receivables_Turnover": 0.95, "Inventory_Turnover": 0.93,
            "Debt_to_Assets": 1.05, "Debt_to_Equity": 1.05
        },
        "sector_sensitivity": {"Materials":1.5,"Technology":1.2,"Consumer Discretionary":1.2,"Industrials":1.2,"__default__":1.0}
    },
    "Supply Chain Disruptions (2021–2022)": {
        "base_impacts": {
            "Inventory_Turnover": 0.85, "Receivables_Turnover": 0.92, "Asset_Turnover": 0.93,
            "Revenue_CAGR_3Y": 0.92, "Gross_Margin": 0.95
        },
        "sector_sensitivity": {"Technology":1.3,"Automotive":1.4,"Industrials":1.2,"Consumer Discretionary":1.2,"__default__":1.0}
    },
    "Energy Price Shock (Europe 2022)": {
        "base_impacts": {
            "Gross_Margin": 0.92, "Net_Profit_Margin": 0.90, "ROA": 0.92,
            "Current_Ratio": 0.93, "Quick_Ratio": 0.93
        },
        "sector_sensitivity": {"Materials":1.3,"Industrials":1.3,"Consumer Staples":1.2,"Utilities":1.2,"__default__":1.0}
    },
    "Tech Valuation Reset (2022–2023)": {
        "base_impacts": {"Net_Profit_Margin":0.95, "ROE":0.93, "ROA":0.95},
        "sector_sensitivity": {"Technology":1.5,"__default__":1.0}
    },
    "Oil Demand Crash (2020)": {
        "base_impacts": {"Gross_Margin":0.92, "Net_Profit_Margin":0.88, "ROA":0.90, "ROE":0.90},
        "sector_sensitivity": {"Energy":1.6,"Transportation":1.3,"__default__":1.0}
    }
}

SYSTEMIC_SHOCKS = {
    "Global Financial Crisis (2008–2009)": {
        "impacts": {"Net_Profit_Margin":0.90,"ROA":0.90,"ROE":0.88,
                    "Current_Ratio":0.95,"Quick_Ratio":0.95,
                    "Debt_to_Assets":1.10,"Operating_Income_to_Debt":0.85,
                    "Revenue_CAGR_3Y":0.90}
    },
    "Interest Rate +300bps": {
        "impacts": {"Debt_to_Equity":1.10, "Debt_to_Assets":1.05, "Operating_Income_to_Debt":0.85}
    },
    "Government Tightening": {
        "impacts": {"Current_Ratio":0.90,"Quick_Ratio":0.90,"ROA":0.90,"Debt_to_Assets":1.10}
    },
    "Market Liquidity Crisis": {
        "impacts": {"Revenue_CAGR_3Y":0.95,"Net_Profit_Margin":0.92}
    }
}

# Baseline for stress
pd_base_for_stress = _pd_from_row(model, X_base)

# Sector impacts
sector_rows = []
for nm, cfg in SECTOR_CRISIS_CATALOG.items():
    sens = cfg["sector_sensitivity"].get(sector_bucket, cfg["sector_sensitivity"].get("__default__", 1.0))
    fmap = combine_impacts(cfg["base_impacts"], sens, 1.0)
    pd_c = _pd_from_row(model, apply_feature_multipliers_alias(X_base, fmap))
    sector_rows.append({"Scenario": nm, "PD": pd_c, "Impact_%": (pd_c - pd_base_for_stress)/pd_base_for_stress*100.0 if pd_base_for_stress>0 else np.nan})
df_sector = pd.DataFrame(sector_rows).sort_values("Impact_%", ascending=True)

# Systemic impacts
sys_rows = []
for nm, cfg in SYSTEMIC_SHOCKS.items():
    fmap = combine_impacts(cfg["impacts"], 1.0, 1.0)
    pd_c = _pd_from_row(model, apply_feature_multipliers_alias(X_base, fmap))
    sys_rows.append({"Scenario": nm, "PD": pd_c, "Impact_%": (pd_c - pd_base_for_stress)/pd_base_for_stress*100.0 if pd_base_for_stress>0 else np.nan})
df_sys = pd.DataFrame(sys_rows).sort_values("Impact_%", ascending=True)

# Monte Carlo tail risk (fallback feats_df)
ref_df = load_train_reference()
try:
    ref_df = ref_df if isinstance(ref_df, pd.DataFrame) else feats_df
    mc = mc_cvar_pd(model, X_base, ref_df, sims=4000, alpha=0.95, clip_q=(0.01,0.99))
    pd_var = float(mc["VaR"]); pd_cvar = float(mc["CVaR"])
except Exception:
    mc = {"PD_sims": np.array([])}; pd_var = np.nan; pd_cvar = np.nan

# Render 4 charts (2x2)
c1, c2 = st.columns(2)
with c1:
    if not df_sector.empty:
        fig_sector = go.Figure()
        fig_sector.add_trace(go.Bar(
            x=df_sector["Scenario"], y=df_sector["Impact_%"],
            text=[f"{v:.1f}%" if np.isfinite(v) else "-" for v in df_sector["Impact_%"]],
            textposition="outside"
        ))
        fig_sector.update_layout(title=f"Sector Impact — ΔPD vs Baseline (%)  ·  Bucket: {sector_bucket}",
                                 yaxis=dict(title="Impact (%)"),
                                 height=360, margin=dict(l=10, r=10, t=40, b=80))
        st.plotly_chart(fig_sector, width='stretch')
    else:
        st.info("No sector crisis impact computed.")

with c2:
    if not df_sys.empty:
        fig_sys = go.Figure()
        fig_sys.add_trace(go.Bar(
            x=df_sys["Scenario"], y=df_sys["Impact_%"],
            text=[f"{v:.1f}%" if np.isfinite(v) else "-" for v in df_sys["Impact_%"]],
            textposition="outside"
        ))
        fig_sys.update_layout(title="Systemic Impact — ΔPD vs Baseline (%)",
                              yaxis=dict(title="Impact (%)"),
                              height=360, margin=dict(l=10, r=10, t=40, b=80))
        st.plotly_chart(fig_sys, width='stretch')
    else:
        st.info("No systemic impact computed.")

c3, c4 = st.columns(2)
with c3:
    try:
        if not shap_df.empty:
            st.plotly_chart(fig_sh, width='stretch')
        else:
            st.info("SHAP is not available for this model/input.")
    except Exception:
        st.info("SHAP is not available for this model/input.")

with c4:
    if isinstance(mc.get("PD_sims"), np.ndarray) and mc["PD_sims"].size:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=mc["PD_sims"], nbinsx=40))
        fig_mc.add_vline(x=pd_var, line_width=2, line_dash="dash", line_color="orange")
        fig_mc.add_vline(x=pd_cvar, line_width=2, line_dash="dot", line_color="red")
        fig_mc.update_layout(title="Monte Carlo — PD sims (VaR 95% orange, CVaR 95% red)",
                             xaxis_title="PD", yaxis_title="Count",
                             height=360, margin=dict(l=10, r=10, t=40, b=40))
        st.plotly_chart(fig_mc, width='stretch')
    else:
        st.info("Monte Carlo distribution unavailable.")

# KPIs
k1, k2, k3 = st.columns(3)
with k1: st.metric("Baseline PD (post-adj in B)", f"{pd_final:.2%}")
with k2: st.metric("Max PD (Sector/Systemic)", f"{max(df_sector['PD'].max() if not df_sector.empty else 0.0, df_sys['PD'].max() if not df_sys.empty else 0.0):.2%}")
with k3: st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}" if np.isfinite(pd_cvar) else "-")

st.caption("Notes: PD is model-based with multi-factor log-odds adjustments (exchange, size, leverage, profitability, liquidity, governance, sector tilt). "
           "Stress impacts use alias-mapped feature multipliers; unknown columns are skipped safely. "
           "Coverage ratios (Interest_Coverage, EBITDA_to_Interest) have been removed from scenarios as requested.")
