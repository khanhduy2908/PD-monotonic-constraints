# app.py — Corporate Default Risk Scoring (Bank-grade, single page)
# ---------------------------------------------------------------
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ==== Utils from your repo ====
from utils.data_cleaning import clean_and_log_transform
from utils.feature_engineering import preprocess_and_create_features
from utils.feature_selection import select_features_for_model
from utils.model_scoring import load_lgbm_model, model_feature_names, explain_shap
from utils.policy import load_thresholds, thresholds_for_sector, classify_pd

# ===================== Page config & style =====================
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
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            if df.shape[1] == 0:
                raise ValueError("CSV has no columns (empty or bad delimiter).")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to read {path}: {last_err}")

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
    """Map row -> 1xN theo danh sách feature kỳ vọng của model; thêm cột thiếu = 0; bỏ cột thừa."""
    m_feats = model_feature_names(model)
    feats = list(m_feats) if m_feats else list(fallbacks)
    data = {f: float(row.get(f, 0.0)) for f in feats}
    X = pd.DataFrame([data], columns=feats)
    return force_numeric(X)

def align_features_to_model(X_df: pd.DataFrame, model):
    """Đảm bảo X_df có đúng ĐẦY ĐỦ & THỨ TỰ cột như khi training model (fix LightGBM shape error)."""
    # LightGBM sklearn exposes feature_name_ (list[str])
    model_features = list(getattr(model, "feature_name_", []) or [])
    if not model_features:
        # fallback: giữ nguyên X_df (trường hợp model không có feature_name_)
        return force_numeric(X_df.copy())
    X = X_df.copy()
    # add missing cols with 0
    for col in model_features:
        if col not in X.columns:
            X[col] = 0.0
    # drop extras and reorder
    X = X[model_features]
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

# ===================== Stress testing (self-contained) =====================
def sector_alias_map(sector_raw: str) -> str:
    s = (sector_raw or "").lower()
    if any(k in s for k in ["tech","information","software","it"]): return "Technology"
    if "tele" in s: return "Telecom"
    if any(k in s for k in ["material","metal","mining","cement"]): return "Materials"
    if any(k in s for k in ["energy","oil","gas","coal"]): return "Energy"
    if any(k in s for k in ["bank","finance","insurance","securities"]): return "Financials"
    if any(k in s for k in ["real estate","property","construction"]): return "Real Estate"
    if any(k in s for k in ["industrial","manufacturing","machinery"]): return "Industrials"
    if any(k in s for k in ["consumer","retail","food","beverage"]): return "Consumer"
    if any(k in s for k in ["utilit"]): return "Utilities"
    return "__default__"

EXCHANGE_INTENSITY = {"UPCOM": 0.6, "HNX": 1.0, "HOSE": 1.0}

SECTOR_CRISIS = {
    "Technology":   {"ROA": 0.70, "ROE": 0.70, "Revenue_CAGR_3Y": 0.70, "EBITDA_to_Interest": 0.70},
    "Telecom":      {"ROA": 0.75, "EBITDA_to_Interest": 0.70, "Revenue_CAGR_3Y": 0.75},
    "Materials":    {"Net_Profit_Margin": 0.75, "ROA": 0.75, "Debt_to_Assets": 1.15},
    "Energy":       {"Net_Profit_Margin": 0.75, "ROE": 0.75, "Debt_to_Assets": 1.15},
    "Financials":   {"ROE": 0.60, "Interest_Coverage": 0.70, "Current_Ratio": 0.85},
    "Real Estate":  {"Debt_to_Assets": 1.30, "Current_Ratio": 0.80, "Quick_Ratio": 0.80, "EBITDA_to_Interest": 0.70},
    "Industrials":  {"Asset_Turnover": 0.75, "EBITDA_to_Interest": 0.75, "ROA": 0.80},
    "Consumer":     {"Net_Profit_Margin": 0.75, "Revenue_CAGR_3Y": 0.80, "Debt_to_Equity": 1.10},
    "Utilities":    {"EBITDA_to_Interest": 0.75, "ROA": 0.85},
    "__default__":  {"ROA": 0.80, "EBITDA_to_Interest": 0.80, "Revenue_CAGR_3Y": 0.85},
}
RISK_UP = {"Debt_to_Assets","Debt_to_Equity","Total_Debt_to_EBITDA","Net_Debt_to_Equity","Long_Term_Debt_to_Assets"}
RISK_DOWN = {"ROA","ROE","Current_Ratio","Quick_Ratio","Interest_Coverage","EBITDA_to_Interest","Operating_Income_to_Debt"}

def systemic_sigma_for(sector_alias: str) -> float:
    return 2.0 if sector_alias in {"Financials","Real Estate"} else 1.8

def apply_sector_crisis_row(Xrow: pd.DataFrame, sector_alias: str, exch_intensity: float) -> pd.DataFrame:
    assert Xrow.shape[0] == 1
    spec = SECTOR_CRISIS.get(sector_alias, SECTOR_CRISIS["__default__"])
    Xs = Xrow.copy()
    for f, mult in spec.items():
        if f in Xs.columns:
            Xs[f] = float(Xs[f].iloc[0]) * (mult * exch_intensity)
    return Xs

def apply_systemic_sigma_row(Xrow: pd.DataFrame, reference_df: pd.DataFrame, k_sigma: float) -> pd.DataFrame:
    assert Xrow.shape[0] == 1
    feats = list(Xrow.columns)
    stats = compute_feature_stats(reference_df, feats)
    Xs = Xrow.copy()
    for f in feats:
        if f not in stats.index: continue
        s = stats.loc[f,"std"]
        if not np.isfinite(s) or s == 0: continue
        v = float(Xs[f].iloc[0])
        if f in RISK_UP:
            Xs[f] = v + k_sigma * float(s)
        elif f in RISK_DOWN:
            Xs[f] = v - k_sigma * float(s)
    return Xs

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
    X = force_numeric(X)
    X = align_features_to_model(X, model)  # ALIGN HERE
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
    raw = read_csv_smart("bctc_final.csv")            # RAW for Overview
    cleaned = clean_and_log_transform(raw.copy())     # pipeline for model
    feats = preprocess_and_create_features(cleaned)
    return raw, feats

@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = load_lgbm_model("models/lgbm_model.pkl")
    thresholds = load_thresholds("models/threshold.json")
    return model, thresholds

# ===================== Page header =====================
st.title("Corporate Default Risk Scoring")
st.caption("English UI • Single page • LightGBM scoring • SHAP • Sector-specific & Systemic stress • Monte Carlo CVaR")

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
exchange = (str(row_model.get("Exchange","")) or "").upper()
ex_intensity = EXCHANGE_INTENSITY.get(exchange, 1.0)

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
debt_raw  = get_raw(["Total_Debt"], (curr_liab + long_liab + short_bor))

revenue_raw = get_raw(["Net Sales","Revenue"])
net_profit_raw = get_raw(["Net Profit For the Year","Net_Profit"])
oper_profit_raw = get_raw(["Operating Profit/Loss","Operating_Profit"])
interest_exp_raw = get_raw(["Interest Expenses","Interest_Expenses"], 0.0)
cash_raw = get_raw(["Cash and cash equivalents (Bn. VND)","Cash"], 0.0)
receivables_raw = get_raw(["Accounts receivable (Bn. VND)","Receivables"], 0.0)
inventories_raw = get_raw(["Net Inventories","Inventories"], 0.0)
current_assets_raw = get_raw(["CURRENT ASSETS (Bn. VND)","Current_Assets"], 0.0)

roa = (net_profit_raw / assets_raw) if (assets_raw and assets_raw != 0) else np.nan
roe = (net_profit_raw / equity_raw) if (equity_raw and equity_raw != 0) else np.nan
dta = (debt_raw / assets_raw) if (assets_raw and assets_raw != 0) else np.nan
dte = (debt_raw / equity_raw) if (equity_raw and equity_raw != 0) else np.nan
current_ratio = (current_assets_raw / curr_liab) if curr_liab else np.nan
quick_ratio = ((cash_raw + receivables_raw) / curr_liab) if curr_liab else np.nan
interest_coverage = (oper_profit_raw / interest_exp_raw) if interest_exp_raw else np.nan
ebitda_to_interest = np.nan  # add if Depreciation & Amortisation available

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
# X_base theo feature của model (đã scale/feature engineered từ feats_df)
X_base = model_align_row(row_model, model, fallbacks=final_features)
# align lần nữa để chắc chắn (đề phòng model có 229 features)
X_base = align_features_to_model(X_base, model)
features_order = list(X_base.columns)

# ===================== A) Company Financial Overview =====================
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

# ===================== D) Stress Testing (no baseline) =====================
st.subheader("D. Stress Testing")

# Reference for stress (covariance & σ)
reference = load_train_reference()
if reference is None:
    # dùng toàn bộ feature dataset làm baseline cho systemic/MC khi không có train_reference
    reference = feats_df.copy()

# Align strictly (giữa X_base & reference)
common_cols = [c for c in X_base.columns if c in reference.columns]
if not common_cols:
    # nếu reference không có cột chung (hiếm), fallback dùng chính X_base columns
    common_cols = list(X_base.columns)
reference = reference[common_cols].copy()
X_base_comm = X_base[common_cols].copy()

# 1) Sector-specific crisis
try:
    X_sector = apply_sector_crisis_row(X_base_comm, sector_alias=sector_alias, exch_intensity=ex_intensity)
    X_sector = align_features_to_model(X_sector, model)
    pd_sector = float(model.predict_proba(X_sector)[:,1][0]) if hasattr(model,"predict_proba") else float(model.predict(X_sector)[0])
except Exception as e:
    st.error(f"Sector Crisis failed: {type(e).__name__} — {e}")
    pd_sector = np.nan

# 2) Systemic crisis (σ–based)
try:
    k_sys = systemic_sigma_for(sector_alias)
    X_sys = apply_systemic_sigma_row(X_base_comm, reference_df=reference, k_sigma=k_sys)
    X_sys = align_features_to_model(X_sys, model)
    pd_sys = float(model.predict_proba(X_sys)[:,1][0]) if hasattr(model,"predict_proba") else float(model.predict(X_sys)[0])
except Exception as e:
    st.error(f"Systemic Crisis failed: {type(e).__name__} — {e}")
    pd_sys = np.nan

# 3) Monte Carlo CVaR 95%
try:
    # mc_cvar_pd đã align bên trong, nhưng để chắc chắn, truyền vào X_base_comm (common features)
    mc = mc_cvar_pd(model, X_base_comm, reference_df=reference, sims=5000, alpha=0.95)
    pd_var, pd_cvar = mc["VaR"], mc["CVaR"]
except Exception as e:
    st.error(f"Monte Carlo CVaR failed: {type(e).__name__} — {e}")
    mc = {"PD_sims": np.array([])}; pd_var = pd_cvar = np.nan

# Layout: 2x2 — 2 bar charts + histogram + metrics
a,b = st.columns(2)
with a:
    st.markdown(f"**Sector Crisis — {sector_alias}**")
    figA = go.Figure()
    figA.add_trace(go.Bar(x=[f"{sector_alias} Crisis"], y=[pd_sector]))
    figA.update_layout(yaxis=dict(tickformat=".0%"), height=300)
    st.plotly_chart(figA, use_container_width=True)

with b:
    st.markdown(f"**Systemic Crisis ({k_sys:.1f}σ)**")
    figB = go.Figure()
    figB.add_trace(go.Bar(x=[f"Systemic {k_sys:.1f}σ"], y=[pd_sys]))
    figB.update_layout(yaxis=dict(tickformat=".0%"), height=300)
    st.plotly_chart(figB, use_container_width=True)

c,d = st.columns(2)
with c:
    st.markdown("**Monte Carlo CVaR 95%**")
    if isinstance(mc.get("PD_sims"), np.ndarray) and mc["PD_sims"].size:
        hist = np.histogram(mc["PD_sims"], bins=40)
        centers = (hist[1][1:]+hist[1][:-1])/2
        figC = go.Figure()
        figC.add_trace(go.Bar(x=centers, y=hist[0]))
        figC.add_vline(x=pd_var, line_width=2, line_dash="dash", line_color="red")
        figC.add_vline(x=pd_cvar, line_width=2, line_dash="dot", line_color="black")
        figC.update_layout(title="PD distribution (VaR 95% red, CVaR 95% black)",
                           xaxis_title="PD", yaxis_title="Frequency", height=300)
        st.plotly_chart(figC, use_container_width=True)
    else:
        st.info("Monte Carlo distribution unavailable.")

with d:
    st.metric("Sector Crisis PD", f"{pd_sector:.2%}" if np.isfinite(pd_sector) else "-")
    st.metric("Systemic Crisis PD", f"{pd_sys:.2%}" if np.isfinite(pd_sys) else "-")
    st.metric("VaR 95% (PD)", f"{pd_var:.2%}" if np.isfinite(pd_var) else "-")
    st.metric("CVaR 95% (PD)", f"{pd_cvar:.2%}" if np.isfinite(pd_cvar) else "-")

# ===================== F) Executive Summary =====================
st.subheader("F. Executive Summary")
bullets = [
    f"Ticker {ticker}, Year {int(year)}, Sector '{sector_raw or '-'}', Exchange '{exchange or '-'}'.",
    f"PD: {pd_base:.2%} → Policy band: {band}.",
    f"Under sector-specific crisis: PD = {pd_sector:.2%}. Under systemic {k_sys:.1f}σ: PD = {pd_sys:.2%}.",
    f"Monte Carlo tail risk: VaR95 = {pd_var:.2%}, CVaR95 = {pd_cvar:.2%}."
]
st.write("\n".join(f"- {x}" for x in bullets))
st.caption("© Corporate Risk Analytics — single page portal")
