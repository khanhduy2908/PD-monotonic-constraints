import streamlit as st
import pandas as pd
import shap
import numpy as np
from modules.ui_components import app_header, section, metric_row
from modules.data_utils import get_record_by_ticker_year, list_tickers
from modules.model_utils import load_model_and_assets, prepare_input_row, scale_numeric, predict_proba_label

def render(state):
    app_header("⚡ Scoring", "Tra cứu theo Ticker & Year, không cần train lại")
    df_master = state.get("df_master")
    if df_master is None or df_master.empty:
        st.warning("Chưa có dữ liệu. Vào mục **Data Ingestion** để nạp dữ liệu trước.")
        return

    model, scaler, features, threshold, constraints = load_model_and_assets()

    tickers = list_tickers(df_master)
    colA, colB = st.columns([2,1])
    with colA:
        ticker = st.selectbox("Ticker", options=tickers, index=0 if tickers else None)
    with colB:
        years = sorted(df_master[df_master["Ticker"]==ticker]["Year"].unique()) if ticker else []
        year = st.selectbox("Year", options=years, index=len(years)-1 if years else 0)

    if st.button("Dự báo", type="primary", use_container_width=True):
        row = get_record_by_ticker_year(df_master, ticker, year)
        if row.empty:
            st.error("Không tìm thấy bản ghi phù hợp.")
            return
        X = prepare_input_row(row, features)
        X_scaled = scale_numeric(X, scaler=scaler)
        proba, label = predict_proba_label(model, X_scaled, threshold)

        metric_row({
            "Default Probability": f"{proba:.2%}",
            "Prediction": "🚨 High Risk" if label==1 else "✅ Low Risk",
            "Threshold": f"{threshold:.2f}"
        })

        # SHAP explanation (fallback to feature_importances if SHAP fails)
        try:
            with st.expander("🔍 Giải thích mô hình (SHAP)", expanded=False):
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_scaled)
                # pick class 1 if list
                vals = sv[1][0] if isinstance(sv, list) else sv[0]
                import plotly.express as px
                import pandas as pd
                s = pd.Series(vals, index=X.columns).sort_values(key=lambda x: x.abs(), ascending=False)[:12]
                fig = px.bar(x=s.values, y=s.index, orientation="h", template="plotly_white",
                             labels={"x":"SHAP value","y":"Feature"}, title="Top Contributors")
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("SHAP explanation unavailable; showing model feature importances instead.")
            try:
                import plotly.express as px
                s = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)[-12:]
                fig = px.bar(x=s.values, y=s.index, orientation="h", template="plotly_white",
                             labels={"x":"Importance","y":"Feature"}, title="Top Feature Importances (approx)")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Không thể hiển thị giải thích mô hình.")
