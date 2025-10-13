# ============================================
# Corporate Default Risk Scoring — Bank-grade
# Single-page Streamlit app
# ============================================

import os, json, warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==== Utils có sẵn trong repo ====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import preprocess_and_create_features
from utils.feature_selection import select_features_for_model
from utils.model_scoring import load_lgbm_model, model_feature_names, explain_shap
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Page config & styles ----------
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

# ---------- Small helpers ----------
ID_LABEL_COLS = {"Year","Ticker","Sector","Exchange","Default"}

def read_csv_smart(path: str) -> pd.DataFrame:
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
    feats = list(model_feature_names(model) or fallbacks)
    data = {f: float(row.get(f, 0.0)) for f in feats}
    return force_numeric(pd.DataFrame([data], columns=feats))

def align_features_to_model(X_df: pd.DataFrame, model):
    model_feats = list(getattr(model, "feature_name_", []) or [])
    if not model_feats:
        return force_numeric(X_df.copy())
    X = X_df.copy()
    for col in model_feats:
        if col not in X.columns:
            X[col] = 0.0
    return force_numeric(X[model_feats])

def load_train_reference():
    for p in ("models/train_reference.parquet", "models/train_reference.csv"):
        if os.path.exists(p):
            try:
                return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
            except Exception:
                pass
    return None

def bucketize_sector(sector_raw: str) -> str:
    s = (sector_raw or "").lower()
    if any(k in s for k in ["real estate","property","construction"]): return "Real Estate"
    if any(k in s for k in ["steel","material","basic res","cement","mining","metal"]): return "Materials"
    if any(k in s for k in ["energy","oil","gas","coal","petro"]): return "Energy"
    if any(k in s for k in ["bank","finance","insurance","securities"]): return "Financials"
    if any(k in s for k in ["software","it","tech","information"]): return "Technology"
    if any(k in s for k in ["utility","power","water","electric"]): return "Utilities"
    if any(k in s for k in ["staple","food","beverage","agri"]): return "Consumer Staples"
    if any(k in s for k in ["retail","consumer","discretionary","apparel","leisure"]): return "Consumer Discretionary"
    if any(k in s for k in ["industrial","manufacturing","machinery"]): return "Industrials"
    if "tele" in s: return "Telecom"
    if any(k in s for k in ["health","pharma","hospital"]): return "Healthcare"
    if any(k in s for k in ["transport","shipping","airline","airport","logistics"]): return "Transportation"
    if any(k in s for k in ["hotel","hospitality","tourism","travel"]): return "Hospitality & Travel"
    if any(k in s for k in ["auto","automobile","motor"]): return "Automotive"
    if any(k in s for k in ["fish","seafood"]): return "Agriculture & Fisheries"
    return "Other"

# Market microstructure risk weight (sàn)
EXCHANGE_INTENSITY = {"UPCOM": 1.25, "HNX": 1.10, "HOSE": 1.00, "HSX": 1.00}

# ---------- Monte Carlo CVaR ----------
def shrink_cov(cov: np.ndarray, alpha: float = 0.15) -> np.ndarray:
    d = np.diag(np.diag(cov))
    shrunk = (1 - alpha) * cov + alpha * d
    w, V = np.linalg.eigh(shrunk)
    w = np.clip(w, 1e-8, None)
    return (V * w) @ V.T

def mc_cvar_pd(model, Xrow: pd.DataFrame, ref_df: pd.DataFrame,
               sims: int = 4000, alpha: float = 0.95, clip_q=(0.01,0.99)) -> dict:
    assert Xrow.shape[0] == 1
    cols = list(Xrow.columns)
    ref = ref_df[cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
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

# ---------- Plotly wrapper: NO deprecated kwargs ----------
def show_plotly(fig, key: str):
    st.plotly_chart(fig, key=key, config={"displayModeBar": False})

# ---------- Load data & model ----------
@st.cache_data(show_spinner=False)
def load_raw_and_features():
    if not os.path.exists("bctc_final.csv"):
        raise FileNotFoundError("bctc_final.csv not found in repository root.")
    raw = read_csv_smart("bctc_final.csv")
    cleaned = clean_and_log_transform(raw.copy())
    feats = preprocess_and_create_features(cleaned)
    return raw, feats

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_lgbm_model("models/lgbm_model.pkl")
    thresholds = load_thresholds("models/threshold.json")
    return model, thresholds

# ---------- Header ----------
st.title("Corporate Default Risk Scoring")
st.caption("English UI • Single page • LightGBM scoring • SHAP explainability • Sector vs Systemic stress • Monte Carlo CVaR • Bank-grade UI")

# ---------- Data init ----------
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

# ---------- Sidebar ----------
tickers = sorted(feats_df["Ticker"].astype(str).unique().tolist())
ticker = st.sidebar.selectbox("Ticker", tickers, index=0 if tickers else None, key="sb_ticker")

years_avail = sorted(feats_df.loc[feats_df["Ticker"].astype(str)==ticker, "Year"].dropna().astype(int).unique().tolist())
year_idx = len(years_avail)-1 if years_avail else 0
year = st.sidebar.selectbox("Year", years_avail, index=year_idx, key=f"sb_year_{ticker}")

row_model = feats_df[(feats_df["Ticker"].astype(str)==ticker) & (feats_df["Year"]==year)]
if row_model.empty:
    st.warning("No record for selected Ticker & Year.")
    st.stop()
row_model = row_model.iloc[0]

row_raw = raw_df[(raw_df["Ticker"].astype(str)==ticker) & (raw_df["Year"]==year)]
row_raw = row_raw.iloc[0] if not row_raw.empty else pd.Series(dtype="object")

sector_raw = str(row_model.get("Sector","")) if pd.notna(row_model.get("Sector","")) else ""
sector_bucket = bucketize_sector(sector_raw)
exchange = (str(row_model.get("Exchange","")) or "").upper()

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

def safe_div(a, b):
    try:
        return (float(a) / float(b)) if (b not in [0, None, np.nan] and float(b)!=0.0) else np.nan
    except Exception:
        return np.nan

total_liab_raw = (curr_liab or 0.0) + (long_liab or 0.0)
interest_bearing_debt = (short_bor or 0.0) + (long_liab or 0.0)
debt_raw = to_float(row_raw.get("Total_Debt")) if ("Total_Debt" in row_raw.index and pd.notna(row_raw.get("Total_Debt"))) else interest_bearing_debt

roa = safe_div(net_profit_raw, assets_raw)
roe = safe_div(net_profit_raw, equity_raw)
dta = safe_div(total_liab_raw, assets_raw); dta = min(max(dta, 0.0), 0.999) if pd.notna(dta) else np.nan
dte = safe_div(debt_raw, equity_raw);     dte = min(max(dte, 0.0), 0.999) if pd.notna(dte) else np.nan
current_ratio = safe_div(current_assets_raw, curr_liab)
quick_ratio   = safe_div((cash_raw or 0.0) + (receivables_raw or 0.0), curr_liab)

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

# ---------- Model input ----------
X_base = model_align_row(row_model, model, fallbacks=final_features)
X_base = align_features_to_model(X_base, model)

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
        fig_rev.add_trace(go.Scatter(x=rev_series["Year"], y=rev_series["Net_Profit"], name="Net Profit", mode="lines+markers", yaxis="y2"))
        fig_rev.update_layout(
            title="Revenue & Net Profit (multi-year)",
            yaxis=dict(title="Revenue"),
            yaxis2=dict(title="Net Profit", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=380
        )
        show_plotly(fig_rev, "ov_rev")
    else:
        st.info("No historical series for this company.")

with col2:
    fig_cap = go.Figure(data=[go.Pie(labels=["Total Debt","Equity"], values=[debt_raw, equity_raw], hole=0.5)])
    fig_cap.update_layout(title="Capital Structure", height=380)
    show_plotly(fig_cap, "ov_cap")

st.markdown("### Key Financial Ratios")
key_ratios = pd.DataFrame({
    "Metric": ["ROA","ROE","Debt_to_Assets","Debt_to_Equity","Current_Ratio","Quick_Ratio"],
    "Value": [roa, roe, dta, dte, current_ratio, quick_ratio]
})
key_ratios["Value"] = key_ratios["Value"].apply(fmt_ratio)
st.dataframe(key_ratios, use_container_width=True, hide_index=True)

# ===================== B) Default Probability (PD) =====================
st.subheader("B. Default Probability (PD) & Policy Band")

def _logit(p, eps=1e-9): 
    p = float(np.clip(p, eps, 1-eps)); return np.log(p/(1-p))
def _sigmoid(z): 
    z = float(z)
    if z>=35: return 1.0
    if z<=-35: return 0.0
    return 1.0/(1.0+np.exp(-z))

pd_model = float(model.predict_proba(X_base)[:,1][0]) if hasattr(model,"predict_proba") else float(model.predict(X_base)[0])

# Per-ticker overrides: thêm mã rủi ro mặc định
TICKER_OVERRIDES = {
    "HAG": {"logit_boost": 2.20, "severity_boost": 0.50, "pd_floor": 0.45},
    "ROS": {"logit_boost": 1.60, "severity_boost": 0.40, "pd_floor": 0.30},
    # thêm tại đây nếu muốn
}

PD_CFG = {
    "exchange_logit_mult": {"UPCOM": 1.10, "HNX": 0.45, "HOSE": 0.00, "HSX": 0.00, "__default__": 0.20},
    "size": {"assets_q40": 0.35, "revenue_q40": 0.20},
    "leverage": {"dta_hi": 0.50, "dte_hi": 0.40, "netde_hi": 0.35},
    "profitability": {"roa_neg": 0.50, "roe_neg": 0.35, "npm_neg": 0.30, "rev_cagr_neg": 0.25},
    "liquidity": {"cr_low": 0.25, "qr_low": 0.20},
    "governance": {"auditor_non_big4": 0.25, "opinion_qualified": 0.70, "filing_delay": 0.25},
    "sector_tilt": {"Real Estate": 0.60, "Materials": 0.25, "Consumer Discretionary": 0.15,
                    "Financials": 0.00, "Utilities": -0.05, "Technology": 0.00, "__default__": 0.05},
    "pd_floor": {"UPCOM": 0.15, "HNX": 0.08, "HOSE": 0.03, "HSX": 0.03, "__default__": 0.05},
    "pd_cap":   {"default": 0.98}
}

def _get(sr, keys, default=np.nan):
    for k in keys:
        if k in sr.index and pd.notna(sr.get(k)):
            try: return float(sr.get(k))
            except Exception: return default
    return default

npm = _get(row_model, ["Net_Profit_Margin","net_profit_margin"])
rev_cagr3y = _get(row_model, ["Revenue_CAGR_3Y","revenue_cagr_3y","sales_cagr_3y"])
nde = _get(row_model, ["Net_Debt_to_Equity","net_debt_to_equity"])
auditor = str(_get(row_raw, ["Auditor","Audit_Firm","Auditor_Name"], "") or "")
opinion = str(_get(row_raw, ["Audit_Opinion","Opinion"], "") or "")
filing_delay = _get(row_raw, ["Filing_Delay_Days","Filing_Delay"], np.nan)

ref = load_train_reference(); ref_use = ref if isinstance(ref,pd.DataFrame) else feats_df
def _q(col, q, fallback=np.nan):
    if (col in ref_use.columns) and ref_use[col].notna().any():
        try: return float(pd.to_numeric(ref_use[col], errors="coerce").quantile(q))
        except Exception: return fallback
    return fallback
assets_q40 = _q("Total_Assets", 0.40, np.nan) if "Total_Assets" in ref_use.columns else np.nan
revenue_q40 = _q("Revenue", 0.40, np.nan) if "Revenue" in ref_use.columns else np.nan

flags = {
    "exch_mult": PD_CFG["exchange_logit_mult"].get(exchange, PD_CFG["exchange_logit_mult"]["__default__"]),
    "assets_q40": (np.isfinite(assets_raw) and np.isfinite(assets_q40) and assets_raw < assets_q40),
    "revenue_q40": (np.isfinite(revenue_raw) and np.isfinite(revenue_q40) and revenue_raw < revenue_q40),
    "dta_hi": (isinstance(dta, float) and dta > 0.70),
    "dte_hi": (isinstance(dte, float) and dte > 1.5),
    "netde_hi": (isinstance(nde, float) and nde > 1.0),
    "roa_neg": (isinstance(roa, float) and roa < 0.0),
    "roe_neg": (isinstance(roe, float) and roe < 0.0),
    "npm_neg": (isinstance(npm, float) and npm < 0.0),
    "rev_cagr_neg": (isinstance(rev_cagr3y, float) and rev_cagr3y < 0.0),
    "cr_low": (isinstance(current_ratio, float) and current_ratio < 0.9),
    "qr_low": (isinstance(quick_ratio, float) and quick_ratio < 0.7),
    "auditor_non_big4": (auditor != "" and not any(k in auditor.lower() for k in ["deloitte","kpmg","ey","ernst","pwc","pricewaterhouse"])),
    "opinion_qualified": (opinion != "" and any(k in opinion.lower() for k in ["qualified","adverse","disclaimer"])),
    "filing_delay": (isinstance(filing_delay, float) and filing_delay >= 20),
}

risk_intensity = 1.0
for cond, bump in [
    ("dta_hi",0.25), ("dte_hi",0.20), ("netde_hi",0.15),
    ("cr_low",0.15), ("qr_low",0.10),
    ("roa_neg",0.20), ("roe_neg",0.10), ("npm_neg",0.10), ("rev_cagr_neg",0.10),
    ("assets_q40",0.10), ("revenue_q40",0.05)
]:
    if flags[cond]: risk_intensity += bump
if exchange == "UPCOM": risk_intensity += 0.25
risk_intensity = float(np.clip(risk_intensity, 1.0, 2.5))

logit0 = _logit(pd_model)
adj = 0.0
adj += flags["exch_mult"]
adj += PD_CFG["sector_tilt"].get(sector_bucket, PD_CFG["sector_tilt"]["__default__"])
for group_cfg, conds in [
    (PD_CFG["size"], ["assets_q40","revenue_q40"]),
    (PD_CFG["leverage"], ["dta_hi","dte_hi","netde_hi"]),
    (PD_CFG["profitability"], ["roa_neg","roe_neg","npm_neg","rev_cagr_neg"]),
    (PD_CFG["liquidity"], ["cr_low","qr_low"]),
    (PD_CFG["governance"], ["auditor_non_big4","opinion_qualified","filing_delay"]),
]:
    for c in conds:
        if flags[c]: adj += group_cfg[c]

ovr = TICKER_OVERRIDES.get(str(ticker), {})
adj += float(ovr.get("logit_boost", 0.0))
risk_intensity += float(ovr.get("risk_boost", 0.0))
adj *= risk_intensity

pd_floor = float(ovr.get("pd_floor", PD_CFG["pd_floor"].get(exchange, PD_CFG["pd_floor"]["__default__"])))
pd_cap = PD_CFG["pd_cap"]["default"]
pd_final = float(np.clip(_sigmoid(logit0 + adj), pd_floor, pd_cap))

thr = thresholds_for_sector(load_thresholds("models/threshold.json"), sector_raw)
band = classify_pd(pd_final, thr)

c1,c2,c3 = st.columns([1,1,2])
with c1: st.metric("PD (multi-factor, post-adj.)", f"{pd_final:.2%}")
with c2: st.metric("Policy Band", band)
with c3:
    st.markdown(f"<span class='small'>Policy: Low &lt; {thr['low']:.0%} • Medium &lt; {thr['medium']:.0%} • "
                f"Floor/Cap: {pd_floor:.0%}/{pd_cap:.0%} • Exchange: {exchange or '-'}</span>", unsafe_allow_html=True)

gauge = go.Figure(go.Indicator(
    mode="gauge+number", value=pd_final*100, number={'suffix': "%"},
    gauge={'axis': {'range': [0,100]},
           'bar': {'color': '#1f77b4'},
           'steps': [{'range':[0,thr['low']*100],'color':'#E8F1FB'},
                     {'range':[thr['low']*100,thr['medium']*100],'color':'#CFE3F7'},
                     {'range':[thr['medium']*100,100],'color':'#F9E3E3'}],
           'threshold': {'line': {'color':'red','width':3},'value':pd_final*100}}
))
gauge.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
show_plotly(gauge, "pd_gauge")

# ===================== C) SHAP (gọn, chắc) =====================
st.subheader("C. Model Explainability (SHAP)")
fig_sh = None
try:
    shap_raw = explain_shap(model, X_base, top_n=10)
    if shap_raw is None:
        shap_df = pd.DataFrame()
    elif isinstance(shap_raw, pd.Series):
        shap_df = shap_raw.reset_index(); shap_df.columns = ["Feature","SHAP"]
    elif isinstance(shap_raw, (list, tuple, np.ndarray)):
        try: shap_df = pd.DataFrame(shap_raw, columns=["Feature","SHAP"])
        except Exception: shap_df = pd.DataFrame()
    elif isinstance(shap_raw, pd.DataFrame):
        shap_df = shap_raw.copy()
    else:
        shap_df = pd.DataFrame()
except Exception:
    shap_df = pd.DataFrame()

def _pick_col(df: pd.DataFrame, cands):
    lower = {c.lower(): c for c in df.columns}
    for c in cands:
        if c in df.columns: return c
        if c.lower() in lower: return lower[c.lower()]
    return None

if shap_df.empty:
    st.info("SHAP is not available for this model/input.")
else:
    feat_col = _pick_col(shap_df, ["Feature","feature","name","variable"])
    shap_col = _pick_col(shap_df, ["SHAP","shap","impact","value","shap_value"])
    if (feat_col is None) or (shap_col is None):
        st.info("SHAP output detected but columns are not recognizable.")
    else:
        shap_df = shap_df[[feat_col, shap_col]].dropna()
        shap_df[shap_col] = pd.to_numeric(shap_df[shap_col], errors="coerce")
        shap_df = shap_df.dropna()
        if not shap_df.empty:
            shap_df["absSHAP"] = shap_df[shap_col].abs()
            shap_df = shap_df.sort_values("absSHAP", ascending=True).tail(10)
            fig_sh = go.Figure()
            colors = ["#E24A33" if v < 0 else "#1F77B4" for v in shap_df[shap_col]]
            fig_sh.add_trace(go.Bar(
                x=shap_df[shap_col], y=shap_df[feat_col].astype(str),
                orientation="h", marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_df[shap_col]], textposition="outside"
            ))
            fig_sh.update_layout(title="Top Feature Contributions (SHAP)",
                                 xaxis=dict(title="SHAP value → PD"),
                                 height=420, margin=dict(l=10, r=20, t=40, b=10))
            show_plotly(fig_sh, "shap_top")

# ===================== D) Stress Testing — Sector & Systemic (auto severity) =====================
st.subheader("D. Stress Testing — Sector & Systemic Impacts")

# alias resolver + fallback
def _norm(s:str)->str: return ''.join(ch for ch in s.lower() if ch.isalnum())
MODEL_COL_MAP = { _norm(c): c for c in X_base.columns }
FEATURE_ALIAS = {
    "ROA": ["roa","return_on_assets"], "ROE": ["roe","return_on_equity"],
    "Gross_Margin": ["gross_margin","gm","grossmarginratio"],
    "Net_Profit_Margin": ["net_profit_margin","npm","netmargin"],
    "Debt_to_Assets": ["debt_to_assets","debtratio","debtassetsratio"],
    "Debt_to_Equity": ["debt_to_equity","dte","deratio"],
    "Total_Debt_to_EBITDA": ["debt_to_ebitda","total_debt_ebitda"],
    "Operating_Income_to_Debt": ["operating_income_to_debt","op_income_debt"],
    "Asset_Turnover": ["asset_turnover","at"],
    "Receivables_Turnover": ["receivables_turnover","rt"],
    "Inventory_Turnover": ["inventory_turnover","it"],
    "Revenue_CAGR_3Y": ["revenue_cagr_3y","sales_cagr_3y","rev_cagr3y"],
    "Current_Ratio": ["current_ratio","cr"], "Quick_Ratio": ["quick_ratio","qr"],
}
def resolve_feature(name:str):
    if name in X_base.columns: return name
    for alt in FEATURE_ALIAS.get(name, []):
        col = MODEL_COL_MAP.get(_norm(alt))
        if col: return col
    base = _norm(name)
    for k, v in MODEL_COL_MAP.items():
        if base in k or k in base: return v
    return None
def apply_multipliers(Xrow: pd.DataFrame, fmap: dict):
    X = Xrow.copy(); hit=False
    for f, mult in fmap.items():
        col = resolve_feature(f)
        if not col: continue
        hit=True
        try: X.at[X.index[0], col] = float(X.at[X.index[0], col]) * float(mult)
        except Exception: pass
    return X, hit
def _pd(model, Xrow):
    Xr = align_features_to_model(Xrow, model)
    return float(model.predict_proba(Xr)[:,1][0]) if hasattr(model,"predict_proba") else float(model.predict(Xr)[0])

# Libraries (loại coverage ratios)
SECTOR_CRISES = {
    "Materials": [
        ("Steel Price Collapse", {"Gross_Margin":0.88,"Net_Profit_Margin":0.85,"ROA":0.90,"ROE":0.90,"Revenue_CAGR_3Y":0.92,"Asset_Turnover":0.95,"Debt_to_Assets":1.05,"Debt_to_Equity":1.05}),
        ("Energy Cost Surge", {"Gross_Margin":0.92,"Net_Profit_Margin":0.90,"ROA":0.92,"Current_Ratio":0.93,"Quick_Ratio":0.93}),
        ("Supply Chain Disruptions", {"Inventory_Turnover":0.85,"Receivables_Turnover":0.92,"Asset_Turnover":0.93,"Revenue_CAGR_3Y":0.92})
    ],
    "Energy": [("Oil Demand Crash", {"Gross_Margin":0.92,"Net_Profit_Margin":0.88,"ROA":0.90,"ROE":0.90})],
    "Real Estate": [
        ("Credit Tightening", {"Debt_to_Assets":1.12,"Debt_to_Equity":1.12,"Current_Ratio":0.92,"Quick_Ratio":0.90,"ROA":0.92}),
        ("Property Price Correction", {"Net_Profit_Margin":0.88,"Revenue_CAGR_3Y":0.88,"ROE":0.90})
    ],
    "Technology": [
        ("Valuation Reset", {"Net_Profit_Margin":0.95,"ROE":0.93,"ROA":0.95}),
        ("Supply Chain Disruptions", {"Inventory_Turnover":0.88,"Receivables_Turnover":0.92,"Asset_Turnover":0.94,"Revenue_CAGR_3Y":0.93}),
        ("US–China Tariffs (export)", {"Gross_Margin":0.95,"Net_Profit_Margin":0.93,"Asset_Turnover":0.97,"Receivables_Turnover":0.95})
    ],
    "Consumer Discretionary": [("COVID Demand Shock", {"Revenue_CAGR_3Y":0.85,"Gross_Margin":0.92,"Net_Profit_Margin":0.85,"ROA":0.88,"ROE":0.88})],
    "Consumer Staples": [("Energy Price Shock", {"Gross_Margin":0.94,"ROA":0.96})],
    "Industrials": [("Logistics/Supply Chain", {"Asset_Turnover":0.93,"Inventory_Turnover":0.88,"Receivables_Turnover":0.92})],
    "Utilities": [("Regulatory Tightening", {"ROA":0.95,"ROE":0.95})],
    "Financials": [("Credit Loss Cycle", {"ROE":0.92,"ROA":0.94})],
    "Healthcare": [("Reimbursement Pressure", {"Net_Profit_Margin":0.95,"ROA":0.96})],
    "Telecom": [("Capex Cycle Upswing", {"ROE":0.95,"ROA":0.96})],
    "Transportation": [("COVID Travel Collapse", {"Revenue_CAGR_3Y":0.80,"Asset_Turnover":0.90,"ROA":0.85})],
    "Hospitality & Travel": [("Tourism Freeze", {"Revenue_CAGR_3Y":0.78,"Gross_Margin":0.90,"ROA":0.85})],
    "Agriculture & Fisheries": [("Export Shock", {"Revenue_CAGR_3Y":0.88,"Gross_Margin":0.92})],
    "Automotive": [("Semiconductor Shortage", {"Inventory_Turnover":0.85,"Asset_Turnover":0.92})],
    "Other": [("Generic Sector Shock", {"ROA":0.95,"Net_Profit_Margin":0.95,"Revenue_CAGR_3Y":0.95})]
}
SYSTEMIC_CRISES = [
    ("Global Financial Crisis", {"Net_Profit_Margin":0.90,"ROA":0.90,"ROE":0.88,"Current_Ratio":0.95,"Quick_Ratio":0.95,"Debt_to_Assets":1.10,"Operating_Income_to_Debt":0.85,"Revenue_CAGR_3Y":0.90}),
    ("Interest Rate +300bps", {"Debt_to_Equity":1.12,"Debt_to_Assets":1.06,"Operating_Income_to_Debt":0.85}),
    ("Government Tightening", {"Current_Ratio":0.90,"Quick_Ratio":0.90,"ROA":0.90,"Debt_to_Assets":1.10}),
    ("Market Liquidity Crisis", {"Revenue_CAGR_3Y":0.95,"Net_Profit_Margin":0.92}),
    ("US–China Tariffs (broad)", {"Gross_Margin":0.96,"Net_Profit_Margin":0.95,"Asset_Turnover":0.98})
]

# Auto severity theo hồ sơ DN + sector + sàn + per-ticker
base_sev = 1.15
if sector_bucket in ["Real Estate","Materials","Consumer Discretionary"]: base_sev += 0.10
if exchange == "UPCOM": base_sev += 0.10
sev_ticker = float(TICKER_OVERRIDES.get(str(ticker), {}).get("severity_boost", 0.0))
risk_multiplier = 1.0 + (risk_intensity - 1.0)
SEVERITY = float(np.clip((1.0 + sev_ticker) * base_sev * risk_multiplier, 1.0, 2.8))

st.caption(f"Sector raw: {sector_raw or '-'} → Bucket: **{sector_bucket}** • Severity used: **×{SEVERITY:.2f}**")

pd_base_stress = _pd(model, X_base)
logit_base = _logit(pd_base_stress)

def _scale_map(base_map: dict, k: float) -> dict:
    return {feat: float(np.clip(1.0 + (mult - 1.0) * k, 0.5, 1.9)) for feat, mult in base_map.items()}

def scenario_pd(fmap, sys_weight=1.0):
    fmap_scaled = _scale_map(fmap, SEVERITY)
    Xs, hit = apply_multipliers(X_base, fmap_scaled)
    if hit:
        return _pd(model, Xs), True
    # fallback: logit bump nếu không chạm cột nào
    bump = 0.7 * SEVERITY * sys_weight
    return _sigmoid(logit_base + bump), False

# Sector
rows_sector = []
for nm, fmap in SECTOR_CRISES.get(sector_bucket, SECTOR_CRISES["Other"]):
    pd_sc, _ = scenario_pd(fmap, 1.0)
    rows_sector.append({"Scenario": nm, "PD": pd_sc, "Impact_%": (pd_sc - pd_base_stress)/pd_base_stress*100.0 if pd_base_stress>0 else np.nan})
df_sector = pd.DataFrame(rows_sector).sort_values("Impact_%", ascending=True)

# Systemic
rows_sys = []
for nm, fmap in SYSTEMIC_CRISES:
    pd_sc, _ = scenario_pd(fmap, 0.8)
    rows_sys.append({"Scenario": nm, "PD": pd_sc, "Impact_%": (pd_sc - pd_base_stress)/pd_base_stress*100.0 if pd_base_stress>0 else np.nan})
df_sys = pd.DataFrame(rows_sys).sort_values("Impact_%", ascending=True)

# Monte Carlo
ref_df = load_train_reference()
try:
    ref_df = ref_df if isinstance(ref_df, pd.DataFrame) else feats_df
    mc = mc_cvar_pd(model, X_base, ref_df, sims=4000, alpha=0.95, clip_q=(0.01,0.99))
    pd_var = float(mc["VaR"]); pd_cvar = float(mc["CVaR"])
except Exception:
    mc = {"PD_sims": np.array([])}; pd_var = np.nan; pd_cvar = np.nan

# Charts
c1, c2 = st.columns(2)
with c1:
    if not df_sector.empty:
        f1 = go.Figure()
        f1.add_trace(go.Bar(x=df_sector["Scenario"], y=df_sector["Impact_%"],
                            text=[f"{v:.1f}%" if np.isfinite(v) else "-" for v in df_sector["Impact_%"]],
                            textposition="outside"))
        f1.update_layout(title=f"Sector Impact — ΔPD vs Baseline (%) • {sector_bucket}",
                         yaxis=dict(title="Impact (%)"),
                         height=360, margin=dict(l=10, r=10, t=40, b=80))
        show_plotly(f1, "chart_sector_impact")
with c2:
    if not df_sys.empty:
        f2 = go.Figure()
        f2.add_trace(go.Bar(x=df_sys["Scenario"], y=df_sys["Impact_%"],
                            text=[f"{v:.1f}%" if np.isfinite(v) else "-" for v in df_sys["Impact_%"]],
                            textposition="outside"))
        f2.update_layout(title="Systemic Impact — ΔPD vs Baseline (%)",
                         yaxis=dict(title="Impact (%)"),
                         height=360, margin=dict(l=10, r=10, t=40, b=80))
        show_plotly(f2, "chart_systemic_impact")

if isinstance(mc.get("PD_sims"), np.ndarray) and mc["PD_sims"].size:
    f3 = go.Figure()
    f3.add_trace(go.Histogram(x=mc["PD_sims"], nbinsx=40))
    f3.add_vline(x=pd_var, line_width=2, line_dash="dash", line_color="orange")
    f3.add_vline(x=pd_cvar, line_width=2, line_dash="dot", line_color="red")
    f3.update_layout(title="Monte Carlo — PD sims (VaR 95% orange, CVaR 95% red)",
                     xaxis_title="PD", yaxis_title="Count",
                     height=360, margin=dict(l=10, r=10, t=40, b=40))
    show_plotly(f3, "chart_mc_pd")

k1, k2, k3 = st.columns(3)
with k1: st.metric("Baseline PD (post-adj in B)", f"{pd_final:.2%}")
with k2: st.metric("Max PD under crises",
                   f"{max(df_sector['PD'].max() if not df_sector.empty else 0.0, df_sys['PD'].max() if not df_sys.empty else 0.0):.2%}")
with k3: st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}" if np.isfinite(pd_cvar) else "-")
